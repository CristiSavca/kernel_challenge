"""Join real scheduler and mixed worker settlement to the physical five-layer draft sidecar."""
from contextlib import nullcontext
import hashlib
from importlib import util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import torch
from torch.nn.modules import module as hooks

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
SRT=STAGE/'python/sglang/srt'
MANIFEST=json.loads((HERE/'validation.json').read_bytes())


def load(name,path):
    spec=util.spec_from_file_location(name,path)
    module=util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    assert hashlib.sha256((HERE/'patch.diff').read_bytes()).hexdigest()==MANIFEST['patch_sha256']
    previous=load('v253_physical',HERE.parent/'v253_window_backed_cache_transfers/broad_writer_check.py')
    previous.source=lambda version,path:(STAGE/path).read_text()
    saved=previous.collect_sidecar()
    owner, original_plan=saved['worker'],saved['plan']
    outer=owner.draft_model_runner.token_to_kv_pool
    leaf=saved['sidecar'].device_pool
    assert leaf is outer.swa_kv_pool and saved['host_domain']=='swa'
    test=load('mixed_native',HERE.parent/'v160_mixed_dflash_serving/independent_round.py')
    limits=load('published_limits',SRT/'arg_groups/prefill_buffer_ceiling.py')
    projection=load('published_projection',SRT/'layers/dflash_context_projection.py')
    test.pl.mixed_prefill_graph_fits=test.admit['mixed_prefill_graph_fits']=limits.mixed_prefill_graph_fits
    test.pl.PrefillCudaGraphRunner.mixed_dflash_token_capacity=16384
    test.bridge.env.update(module_hooks=hooks,dflash_sp_prefill_scope=projection.dflash_sp_prefill_scope)
    test.ref.env.update(project_dflash_prefill_shard=projection.project_dflash_prefill_shard,
                        project_paired_draft_kv=projection.project_paired_draft_kv)
    previous.symbol((SRT/'models/muse_spark.py').read_text(),'_pack_muse_prompt_taps',test.ref.env)
    test.ref.Graph._muse_prompt_taps=None
    # Replace the historical test plane's convenience prefix writer with the
    # native CPU dispatch; its terminal tensor store remains instrumented.
    from types import MethodType
    prefix=previous.symbol((SRT/'mem_cache/memory_pool.py').read_text(),'set_kv_buffer_prefix_valid',
                           dict(torch=torch,_is_cuda=False,_is_hip=False),'MHATokenToKVPool')
    leaf.set_kv_buffer_prefix_valid=MethodType(prefix,leaf)
    test.sched.module('sglang.srt.batch_invariant_ops',is_batch_invariant_mode_enabled=lambda:False)
    test.ref.env['get_forward']=lambda:NS(set=lambda name,value:None,scoped=lambda **kwargs:nullcontext())
    test.ref.env['layernorm_sp'].sp_gather_selected_rows=lambda x,ids,n:x[:n][ids]
    test.ref.PAR.tp_group=NS(world_size=1,rank_in_group=0)
    test.settle.env['IdSpaceKind']=saved['kind']
    names=['_context_kv_pool','_prefill_kv_helper','_append_paired_prompt_kv']
    worker_path='speculative/dflash_worker_v2.py'
    for name in names:
        method=previous.symbol((SRT/worker_path).read_text(),name,test.settle.env,'DFlashWorkerV2')
        setattr(test.settle.DF,name,method)
    originals=test.settle.prepare
    collected={}
    same_source={}
    def prepare(backend,tails,iteration,veto):
        result=originals(backend,tails,iteration,veto)
        worker,prompt,decode,schedule,_,_,entries,steps=result
        worker._sp_prefill_context=worker._target_logits_processor=None
        worker.draft_model_runner=owner.draft_model_runner
        worker.draft_model=owner.draft_model
        worker.draft_model.project_target_hidden=lambda x:x[:,:8].contiguous()
        worker.draft_model.fc=NS(out_features=8)
        def simulate_fc(taps, count, group, processor, full_capture):
            assert full_capture and taps.ndim==2 and count<=len(taps)
            if route=='raw':
                return projection.project_dflash_prefill_shard(taps,count,group,processor,full_capture)
            return taps[:count,:8].contiguous()
        def simulate_paired(hidden,count,group):
            if route!='projected':
                return projection.project_paired_draft_kv(hidden,count,group)
            return torch.stack([torch.stack(layer.self_attn.kv_proj_only(hidden),dim=1)
                                for layer in worker.draft_model.layers],dim=1).reshape(len(hidden),-1)
        test.ref.env.update(project_dflash_prefill_shard=simulate_fc,project_paired_draft_kv=simulate_paired)
        for layer in worker.draft_model.layers:
            layer.self_attn.kv_size=8
            layer.self_attn.v_scale=None
        def writes(plan,reader,*,kind):
            assert reader is owner.draft_model_runner.kv_index_translator and kind==saved['kind'].SLIDING_WINDOW
            native=type(original_plan)(source=original_plan._source,req_pool_indices=None,seq_lens=None,seq_lens_cpu=None,
                                       write_virtual=plan.slots)
            return native.write_ids(reader,kind=kind)
        test.ref.Plan.write_ids=writes
        loc=schedule.out_cache_loc
        fresh=torch.randperm(294,generator=torch.Generator().manual_seed(53+iteration+211*epoch))[:len(loc)]+1
        mapping=outer.full_to_swa_index_mapping
        # All addresses are visible to the target. The draft has a physically distinct slot per live row.
        key=(backend,route,iteration)
        if not epoch:
            same_source[key]=(loc.clone(),fresh.clone())
        else:
            old_loc,old_fresh=same_source[key]
            assert torch.equal(loc,old_loc) and not torch.equal(fresh,old_fresh)
        mapping[loc]=fresh
        assert not torch.equal(fresh,loc)
        for i in range(5):
            for t in leaf.get_kv_buffer(i):t.fill_(-913)
        q=sum(prompt.extend_lens)
        mask=torch.cat([torch.ones(q,dtype=torch.bool),
                        (torch.arange(8)<torch.tensor([1+(iteration+3*i)%8 for i in range(tails)])[:,None]).flatten()])
        target=worker.target_worker
        call=target.forward_batch_generation
        def target_call(*args,**kwargs):
            fb=kwargs['forward_batch']
            assert fb.mixed_verify_is_causal and fb.input_ids.numel()==len(loc)
            out=call(*args,**kwargs)
            assert torch.equal(fb.out_cache_loc,loc)
            out_h=out.logits_output.hidden_states
            h=test.ref.body_tokens(fb.input_ids)[0]
            assert out.logits_output.dflash_prefill_context_done==(route!='raw')
            assert out.logits_output.dflash_prefill_kv_projected==(route=='projected')
            planes=[]
            for layer in worker.draft_model.layers:
                attn=layer.self_attn
                k,v=attn.kv_proj_only(h)
                planes.append((k,v))
            if route=='projected':
                reference=torch.stack([torch.stack((k,v),dim=1) for k,v in planes],dim=1).reshape(len(h),-1)
                assert torch.equal(out_h[:len(h)],reference)
            else:
                assert torch.equal(out_h[:len(h),:8],h)
            expected=[]
            for i,(k,v) in enumerate(planes):
                a=worker.draft_model.layers[i].self_attn
                pair=[]
                for t,values in zip(leaf.get_kv_buffer(i),(a.apply_k_rope(fb.positions,a.apply_k_norm(k)),v)):
                    gold=torch.full_like(t,-913)
                    gold[fresh[mask]]=values[mask].view(-1,1,8)
                    pair.append(gold)
                expected.append(pair)
            collected.update(expected=expected,live=int(mask.sum()),dead=int((~mask).sum()),prompt=q,
                             repeated_slots=loc.tolist(),fresh=fresh.tolist(),entry=fb.forward_mode)
            return out
        target.forward_batch_generation=target_call
        return result
    test.settle.prepare=prepare
    results=[]
    for epoch in (0,1):
        for backend in ('eager','FULL','BREAKABLE'):
            for route in ('raw','fc','projected'):
                for iteration in (0,3,7):
                    with torch.inference_mode():
                        report=test.replay(backend,2,iteration,iteration!=3)
                    for layer,pair in enumerate(collected['expected']):
                        for got,gold in zip(leaf.get_kv_buffer(layer),pair):
                            assert torch.equal(got.view(torch.int16),gold.view(torch.int16)),(backend,route,iteration,layer)
                    results.append(dict(epoch=epoch,backend=backend,route=route,iteration=iteration,prompt=collected['prompt'],
                                      committed=collected['live']-collected['prompt'],rejected=collected['dead'],next_tick_checked=True))
    files=[worker_path,'speculative/dflash_mixed_forward.py','speculative/dflash_mixed_graph_admission.py',
           'managers/scheduler.py','managers/tp_worker.py','managers/scheduler_components/mixed_dflash_admission.py',
           'model_executor/runner/prefill_cuda_graph_runner.py','models/muse_spark.py','layers/dflash_context_projection.py',
           'mem_cache/kv_loc_plan.py','mem_cache/kv_index_translator.py','mem_cache/swa_memory_pool.py']
    hashes={}
    for file in files:
        physical=(SRT/file).read_bytes()
        expected=subprocess.check_output(['git','-C',STAGE,'show',MANIFEST['tree']+':python/sglang/srt/'+file])
        assert physical==expected
        hashes[file]=hashlib.sha256(physical).hexdigest()
    return dict(tree=MANIFEST['tree'],patch=MANIFEST['patch_sha256'],results=results,source_sha256=hashes,
                compared_BF16_words=len(results)*sum(t.numel() for layer in range(5) for t in leaf.get_kv_buffer(layer)),
                accepted_verify_rows=sum(row['committed'] for row in results),rejected_verify_rows=sum(row['rejected'] for row in results),
                prompt_rows=sum(row['prompt'] for row in results),same_virtual_new_swa_pairs=len(same_source),
                native_cpu_prefix_writer=True,
                limits='Single-rank CPU serving control flow. Synthetic target layers, draft proposal and projection arithmetic. CUDA graph replay emulated.',hardware=False)

if __name__=='__main__':
    torch.set_num_threads(1)
    print(json.dumps(main(),indent=2))
