"""Use the actual mixed worker/sampler/accept and real TP Gloo, with checked CPU Triton pointer operations."""
from collections import Counter
from datetime import timedelta
from hashlib import sha256
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve()
KERNELS = ROOT / 'python/sglang/kernels/ops'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


device = module('device_fixture_v203', next(HERE.parent.glob('v156_*/device_cpu.py')))
ref, env = device.ref, device.env
fx = module('pointer_v203', next(HERE.parent.glob('v70_*/cpu_check.py')))
pair = module('pairmax_v203', next(HERE.parent.glob('v91_*/cpu_check.py')))


def host_kernel(path, kernel_name, wrapper):
    pids, (body, launch) = fx.program(path, (kernel_name, wrapper))
    ns = body.__globals__
    ns['tl'].program_id = lambda axis: fx.array(np.int32(pids[axis]))
    ns['tl'].max = pair.pair_max
    ns['libdevice'] = NS(tanh=lambda x: fx.array(np.tanh(x)))
    ns['torch'] = torch
    ns['triton'].cdiv = lambda x, y: (x+y-1)//y

    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                kwargs.pop('num_warps', None)
                arguments = [fx.Pointer(v.detach().numpy()) if isinstance(v,torch.Tensor) else v for v in args]
                if kernel_name=='_softcap_row_argmax_kernel':
                    arguments[4]=fx.array(np.int32(arguments[4]))
                for index in itertools.product(*(range(n) for n in grid)):
                    pids[:len(index)] = index
                    body(*arguments, **kwargs)
            return run
    ns[kernel_name] = Kernel()
    return launch


def build_processor(group):
    arg = host_kernel(KERNELS/'sampling/softcap_row_argmax.py','_softcap_row_argmax_kernel','softcap_row_argmax')
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)/'softcap.py'
        src.write_bytes(subprocess.check_output(['git','-C',str(ROOT),'show',':python/sglang/kernels/ops/activation/softcap.py']))
        full = host_kernel(src,'_gathered_softcap_logits_kernel','gathered_softcap_logits')
    env.update(get_parallel=lambda:ref.PAR, softcap_row_argmax=arg, gathered_softcap_logits=full)
    ref.PAR.tp_group = group
    proc = device.build('layers/logits_processor.py','LogitsProcessor',
            {'_get_logits','is_deferred_shard','finish_deferred_gather','deferred_argmax','_gather_softcap_fused'})
    head=proc()
    head.do_tensor_parallel_all_gather=True
    head.do_tensor_parallel_all_gather_dp_attn=False
    head._gather_dp_attn_hidden_states=lambda h,meta:(h,h)
    head.logit_scale=None
    head._compute_lm_head=lambda h,_head,_bias,out=None:h
    head._can_gather_softcap_fused=lambda *args:True  # device capability substituted on CPU
    return head


def raw_logits(rows, cols, valid, first_tail, width, size, with_nan):
    idx = torch.arange(cols)
    vals = ((idx[None,:]*29 + torch.arange(rows)[:,None]*61) % 173).float()/31-2
    for r in range(rows):
        mode = (r-first_tail)%8
        if mode == 0:  # Equal winners within a shard, across first/last shard, and beyond one native tile.
            vals[r,::width] = 12
            if width>4096: vals[r,4097]=12
        elif mode == 1:
            vals[r,1] = 1e20
            vals[r,width] = float('inf')
        elif mode == 2: vals[r].fill_(float('-inf'))
        elif mode == 3:
            vals[r].fill_(-0.0)
            vals[r,::3]=0.0
        elif mode == 4:
            vals[r].fill_(-10000)
            vals[r,valid-1]=10
        elif mode == 5:
            vals[r, max(0,valid-width//2)]=7
        if with_nan and r>=first_tail and mode in (1,4,6):
            # Lowest valid NaN and finite highest values can be in different blocks and ranks.
            pos = 1 if mode==1 else (width + 4096 if mode==4 and width>4096 else width+1)
            vals[r,pos]=float('nan')
            vals[r,valid-1]=float('nan')
    vals[:,valid:]=float('nan')  # every invalid vocabulary lane poisoned even on finite tests
    return vals


def worker(rank, size, store, result_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=size,timeout=timedelta(seconds=60))
    events=[]
    def gather(out, val):
        # Gloo worker writes normal (non-inference) scratch, then caller thread copies back.
        with torch.inference_mode(False):
            parts=[torch.empty(tuple(val.shape),dtype=val.dtype) for _ in range(size)]
            dist.all_gather(parts,val.contiguous().clone())
        out.view(size,*val.shape).copy_(torch.stack(parts))
        events.append(tuple(val.shape))
    group=NS(world_size=size,rank_in_group=rank,all_gather_into_tensor=gather)
    proc=build_processor(group)
    count=Counter()
    try:
        with torch.inference_mode():
            for (shard,short,tails,iteration) in [(17,3,2,2),(128,1,3,1),(4111,7,2,4),(7,7,4,7)]:
                for with_nan in ([False,True] if 'nan' in sys.argv[2:] else [False]):
                    valid=shard*size-short
                    df,p,tail,mixed,fb,*_=device.prepare('eager',tails,iteration,True)
                    prompts=len(p.reqs);q=sum(p.extend_lens);tw=tails*8
                    ref.PAR.tp_group=group
                    proc.defer_verify_gather=True;proc.final_logit_softcapping=25;proc.vocab_size=valid
                    df._target_logits_processor=proc
                    tail.sampling_info.vocab_size=p.sampling_info.vocab_size=valid
                    whole=raw_logits(prompts+tw,shard*size,valid,prompts,shard,size,with_nan)
                    local=whole[:,rank*shard:(rank+1)*shard].contiguous()
                    metadata=NS(mixed_verify_sharded_logits=False,mixed_verify_logits_indices=torch.arange(len(local)),
                                forward_mode=ref.Mode.EXTEND,next_token_logits_buffer=None)
                    events.clear()
                    capped=proc._get_logits(local,None,metadata)
                    assert events==[(len(local),shard)]
                    metadata.mixed_verify_sharded_logits=True
                    events.clear()
                    selected=proc._get_logits(local,None,metadata)
                    assert selected is local and not events
                    count['native_producer_calls']+=2
                    # Existing pretarget policy must send a penalized request to ordinary serving,
                    # whether the prompt or a tail owns the penalty. No head traffic may start.
                    assert df.can_settle_mixed_verify(p,tail)
                    for cohort in (p,tail):
                        for key,bad in (('logit_bias',torch.ones(1)),('acc_linear_penalties',torch.zeros(1)),
                                        ('acc_additive_penalties',torch.zeros(1)),('acc_scaling_penalties',torch.ones(1))):
                            si=cohort.sampling_info; old=getattr(si,key,None);setattr(si,key,bad)
                            try:assert not df.can_settle_mixed_verify(p,tail) and not events
                            finally:setattr(si,key,old)
                            count['native_penalty_admission_vetoes']+=1
                    gold=torch.from_numpy(np.tanh(whole[:,:valid].numpy()/np.float32(25))*np.float32(25))
                    torch.testing.assert_close(capped,gold,rtol=0,atol=0,equal_nan=True)
                    prompt_bias=torch.zeros_like(capped[:prompts])
                    biased_ids=(capped[:prompts].argmax(-1)+1)%valid
                    prompt_bias.scatter_(1,biased_ids[:,None],1000)
                    ordinary_info=p.sampling_info
                    del ordinary_info.apply_logits_bias  # restore published SamplingBatchInfo implementation
                    for key in ('acc_additive_penalties','logit_bias'):
                        setattr(ordinary_info,key,prompt_bias)
                        try:
                            assert not df.can_settle_mixed_verify(p,tail)
                            actual_ids=df.target_worker.model_runner.sample(ref.LO(next_token_logits=capped[:prompts].clone()),fb)
                            assert torch.equal(actual_ids,biased_ids)
                        finally:setattr(ordinary_info,key,None)
                        count['ordinary_penalized_prompt_sampler_calls']+=1
                    winners=gold[prompts:].argmax(-1).view(tails,8)
                    candidates=torch.full((tails,8),2,dtype=torch.long)
                    candidates[:,1:]=winners[:,:-1]
                    desired=[]
                    for i in range(tails):
                        n=(iteration+i*4-1)%8+1; desired.append(n)
                        if n<8:candidates[i,n]=(candidates[i,n]+1)%valid
                    positions=torch.cat([fb.positions,(tail.seq_lens[:,None]+torch.arange(8)).flatten()])
                    hidden=torch.arange((q+tw)*40).view(q+tw,40).bfloat16()
                    snapshots=[]
                    for is_local in (False,True):
                        events.clear()
                        target=device.Result(logits_output=device.ref.LO(next_token_logits=local if is_local else capped,hidden_states=hidden),
                                             kv_loc_plan=ref.Plan(mixed),can_run_cuda_graph=iteration%2==0)
                        result=df.settle_mixed_verify(target_out=target,prompt_forward_batch=fb,decode_sampling_info=tail.sampling_info,
                                       candidates=candidates,prefix_lens=tail.seq_lens,positions=positions)
                        a,b,next_input=result
                        assert events==([(prompts,shard),(tw,2)] if is_local else []),events
                        assert a.next_token_ids.tolist()==gold[:prompts].argmax(-1).tolist()
                        assert b.accept_lens.tolist()==desired,(with_nan,rank,shard,b.accept_lens,desired)
                        expected_bonus=winners[torch.arange(tails),torch.tensor(desired)-1]
                        assert torch.equal(next_input.bonus_tokens,torch.cat((a.next_token_ids,expected_bonus)))
                        assert torch.equal(next_input.new_seq_lens,torch.cat((fb.seq_lens,tail.seq_lens+torch.tensor(desired))))
                        snapshots.append((a.next_token_ids,b.next_token_ids,b.accept_lens,next_input.new_seq_lens,next_input.bonus_tokens))
                    assert all(torch.equal(a,b) for a,b in zip(*snapshots))
                    count['per_rank_case_pairs']+=1;count['native_prompt_and_accept_calls']+=2
                    count['compared_logits']+=gold.numel()
                    count['tail_target_rows']+=tw
                    count['posted_prompt_vocab_values']+=prompts*shard
                    count['posted_tail_pair_values']+=tw*2
                    count['nan_tail_target_rows']+=int(torch.isnan(gold[prompts:]).any(-1).sum())
            if rank==0:Path(result_path).write_text(json.dumps(dict(count)))
    finally:dist.destroy_process_group()


def main():
    output={}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            mp.spawn(worker,args=(size,str(folder/'init'),str(folder/'result.json')),nprocs=size,join=True)
            output[size]=json.loads((folder/'result.json').read_text())
    paths=['srt/layers/logits_processor.py','srt/speculative/dflash_worker_v2.py','srt/layers/sampler.py','srt/model_executor/model_runner.py',
           'kernels/ops/sampling/softcap_row_argmax.py','kernels/ops/activation/softcap.py']
    return dict(torch=torch.__version__,actual_gloo=True,h200=False,same_rows_seen_by_each_rank=output,
                files={p:sha256((ROOT/'python/sglang'/p).read_bytes() if (ROOT/'python/sglang'/p).exists() else subprocess.check_output(['git','-C',str(ROOT),'show',':python/sglang/'+p])).hexdigest() for p in paths},
                simulations=['checked CPU execution of Triton load/store/reduction bodies, NumPy FP32 tanh','synthetic target logits, physical slots and model taps'])

if __name__=='__main__':print(json.dumps(main(),indent=2))
