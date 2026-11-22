"""V235/V237 processor, full-graph output slice and actual DFLASH consumption branch on CPU."""
import argparse
import ast
from contextlib import nullcontext
import dataclasses
from functools import cache
from hashlib import sha1, sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch


DIR=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate",type=Path,default=DIR)
parser.add_argument("--stage",type=Path)
parser.add_argument("--output",type=Path)
parser.add_argument("--expect-sha",help="require this exact candidate patch SHA-256 before replay")
args=parser.parse_args()
candidate=args.candidate.resolve()
spec=importlib.util.spec_from_file_location("graph_reads",DIR.parent/"v217_captured_verify_context/independent_model_entry.py")
u=importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
baseline=DIR.parent/"v235_peer_mixed_sliding_prefill"
base_manifest=json.loads((baseline/"validation.json").read_bytes())
manifest=json.loads((candidate/"validation.json").read_bytes())
parent_bytes=(baseline/"patch.diff").read_bytes()
child_bytes=(candidate/"patch.diff").read_bytes()
parent_sha=sha256(parent_bytes).hexdigest()
child_sha=sha256(child_bytes).hexdigest()
if parent_sha!=base_manifest["patch_sha256"] or child_sha!=manifest["patch_sha256"]:
    raise ValueError("full patch digest differs from the frozen validation manifest")
if not manifest["base"]==base_manifest["base"]==u.PIN:
    raise ValueError("candidate and V235 require the same pinned serving base")
if args.expect_sha and args.expect_sha!=child_sha:
    raise ValueError("requested patch digest differs from the selected candidate")
u.PARENT=parent_bytes.decode()
u.CHILD=child_bytes.decode()
u.STAGE=args.stage.resolve() if args.stage else None
LP="layers/logits_processor.py"
GPU="model_executor/runner/decode_cuda_graph_runner.py"
WORKER="speculative/dflash_worker_v2.py"
SCALE=.17677669529663687


@cache
def read(path,child):
    result=u.read_source(path,child)
    if child and u.STAGE:
        relative=u.SRT+path
        contents=(u.STAGE/relative).read_bytes()
        assert contents==result.encode(),relative
        oid=subprocess.check_output(["git","-C",u.STAGE,"rev-parse",manifest["tree"]+":"+relative],text=True).strip()
        assert sha1(f"blob {len(contents)}\0".encode()+contents).hexdigest()==oid,relative
    return result


@cache
def parsed(path,child):
    return ast.parse(read(path,child))


@cache
def node(path,name,child,cls=None):
    body=parsed(path,child).body
    if cls:
        body=next(n for n in body if getattr(n,"name",None)==cls).body
    return next(n for n in body if getattr(n,"name",None)==name)


def bind(path,name,env,child,cls=None):
    n=node(path,name,child,cls)
    n.decorator_list=[]
    exec("from __future__ import annotations\n"+ast.unparse(n),env)
    return env[name]


class FB(NS):
    def __getattr__(self,name):
        return None


class Mode:
    def __init__(self,name): self.name=name
    def is_target_verify(self): return self.name=="TARGET_VERIFY"
    def is_extend(self): return self.name=="EXTEND"
    def is_draft_extend_v2(self): return False
    def is_dllm_extend(self): return False
    def is_decode_or_idle(self): return False


def softcap(x,cap):
    return (x/cap).tanh()*cap


class Group:
    world_size=2
    rank_in_group=0
    def all_gather_into_tensor(self,out,local):
        out.copy_(local.repeat(self.world_size,1))


def capture_gate(child, active_hook):
    cls=type("FullCudaGraphBackend",(),{})
    module_name="sglang.srt.model_executor.runner_backend.full_cuda_graph_backend"
    module=ModuleType(module_name);module.FullCudaGraphBackend=cls
    other_name="sglang.srt.batch_invariant_ops"
    other=ModuleType(other_name);other.is_batch_invariant_mode_enabled=lambda:False
    target=torch.nn.Module()
    target.model=torch.nn.Module()
    target.model.language_model=torch.nn.Identity()
    target.logits_processor=torch.nn.Identity()
    if active_hook:
        target.logits_processor.register_forward_hook(lambda m,a,out:out)
    fc=torch.nn.Linear(4,4,bias=False)
    group=NS(world_size=2,rank_in_group=0)
    ctx=NS(group=group,fc=fc,target=target,logits_processor=target.logits_processor)
    runner=NS(backend=cls(),ragged_verify_mode=False,enable_pdmux=False,enable_two_batch_overlap=False,
              enable_torch_compile=False,max_num_token=12,model_runner=NS(capture_head_hooks=[],capture_tail_hooks=[]))
    obj=NS(context=ctx,ready=lambda:True,local_weight=None,raw=None,weight=fc.weight)
    gate=bind("layers/dflash_context_projection.py","prepare_capture",dict(torch=torch,_agree=lambda b,g:b),child,"DFlashVerifyFCGraph")
    with patch.dict(sys.modules,{module_name:module,other_name:other}):
        eligible=gate(obj,runner)
    assert (eligible is None)==active_hook
    return None if eligible is None else eligible.raw


def definitions(child):
    flags=NS(sp_active=False,scoped=lambda **kw:nullcontext(),set=lambda *a:None)
    genv=dict(torch=torch,dataclasses=dataclasses,ForwardBatch=FB,ForwardMode=NS(IDLE=Mode("IDLE")),
              _is_cpu=False,_is_npu=False,_autotune_run_lm_head=None,get_forward=lambda:flags,
              _trace_e2e_logits=lambda *a,**k:None,layernorm_sp=NS(maybe_exit_gather=lambda **kw:(kw["hidden_states"],kw["hidden_states_before_norm"]),runs_sp=lambda mode:False),
              DpPaddingMode=NS(SUM_LEN=0),
              get_parallel=lambda:NS(tp_group=Group()),LoRABatchLayout=NS(TP_GLOBAL=0,DP_LOCAL=1),
              envs=NS(SGLANG_TRACE_LOGITS_E2E_SYNC=NS(get=lambda:False),SGLANG_LOG_DECODE_GRAPH_KEY=NS(get=lambda:False)))
    out_class=node(LP,"LogitsProcessorOutput",child)
    exec("from __future__ import annotations\n"+ast.unparse(out_class),genv)
    class Meta(NS):
        pass
    Meta.from_forward_batch=classmethod(bind(LP,"from_forward_batch",genv,child,"LogitsMetadata"))
    genv["LogitsMetadata"]=Meta
    class Processor:
        pass
    genv["LogitsProcessor"]=Processor
    for method in ("forward","_get_logits","_get_pruned_states","_get_hidden_states_to_store",
                   "_get_last_hidden_states_to_store","_can_gather_softcap_fused",
                   "finish_deferred_gather","deferred_argmax","_gather_softcap_fused","is_deferred_shard"):
        setattr(Processor,method,bind(LP,method,genv,child,"LogitsProcessor"))
    def winners(x,c,w,offset,*,scale=None,first_nan=False):
        values=softcap(x[:,:w] if scale is None else x[:,:w]*scale,c)
        values, ids=values.max(dim=1)
        return torch.stack([values,ids+offset],dim=1)
    genv["softcap_row_argmax"]=winners
    genv["gathered_softcap_logits"]=lambda g,c,out:out.copy_(softcap(g.transpose(0,1).reshape(out.shape[0],-1)[:,:out.shape[1]],c))
    p=Processor()
    p.do_tensor_parallel_all_gather=True
    p.use_attn_tp_group=p.do_tensor_parallel_all_gather_dp_attn=False
    p.final_logit_softcapping=20.
    p.defer_verify_gather=True
    p.vocab_size=33
    p.expected_aux_hidden_states=None
    p.keep_last_hidden_states=False
    p._gather_dp_attn_hidden_states=lambda h,m:(h,h)
    p._can_use_tp_lm_head_all_to_all=lambda *a:False
    p._compute_lm_head=lambda h,*a,**kw:h.clone()
    genv.update(device_timer_ctx=lambda *a:nullcontext(),
                SharedReadEnds=NS(PRE_REPLAY=1,IN_REPLAY=2,POST_REPLAY=3))
    execute=bind(GPU,"execute",genv,child,"DecodeCudaGraphRunner")
    skip=bind(WORKER,"_can_skip_verify_logits",genv,child,"DFlashWorkerV2")
    # Isolate the actual consumer between the real target result and grammar-mask construction.
    # This includes full-logit fallback, in-place/out-of-place operations, tag clearing and rank-local winner selection.
    method=node(WORKER,"verify_proposed_decode_block",child,"DFlashWorkerV2")
    start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="logits_output" and ast.unparse(n.value)=="target_out.logits_output")
    end=next(i for i,n in enumerate(method.body[start:],start) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="grammar_mask")
    body="\n".join(ast.unparse(n) for n in method.body[start:end])
    def consume(output,batch,info):
        ns=dict(genv,self=NS(_target_logits_processor=p,_can_skip_verify_logits=lambda b,s:skip(NS(_selector_sample=None),b,s)),
                target_out=NS(logits_output=output,can_run_cuda_graph=True),batch=batch,sampling_info=info)
        exec(body,ns)
        return ns["target_predict"],ns["logits_output"]
    def replay(output,batch,rows):
        backend=NS(replay_session=lambda:nullcontext(),replay=lambda *a:output)
        runner=NS(model_runner=NS(device_timer=None),_resolve_shared_read_ends=lambda *a:None,
                  _replay_attn_backend=lambda:None,model_context=lambda:nullcontext(),backend=backend,
                  load_batch=lambda *a:None,_replay_graph_key=None,_process_output_after_replay=lambda out,b:out,
                  is_dllm=False,raw_num_token=rows)
        return execute(runner,batch)
    return p,replay,consume


def sample_info(route):
    flags=NS(is_all_greedy=route!="sample",has_custom_logit_processor=route=="custom",
             penalizer_orchestrator=NS(is_required=route=="penalty"),sampling_mask_batch_indices=None,
             acc_linear_penalties=None,grammar_mask=None,logit_bias=None)
    if route=="bias": flags.logit_bias=torch.tensor([0.])
    return flags


def trial(child,kind,raw,route):
    p,replay,consume=definitions(child)
    p.logit_scale=SCALE
    seen=[]
    if kind=="observer":
        original=p._can_gather_softcap_fused
        def observing(logits,*args):
            seen.append(logits.clone())
            return original(logits,*args)
        p._can_gather_softcap_fused=observing
    if kind=="unit": p.logit_scale=1.
    if kind=="zero": p.logit_scale=0.
    if kind=="no_scale": p.logit_scale=None
    if kind=="subnormal": p.logit_scale=float(torch.finfo(torch.float32).tiny)/2
    if kind=="override":
        previous=p._get_logits
        p._get_logits=lambda *a,**kw:previous(*a,**kw)
    fmode=Mode("EXTEND" if kind=="mixed" else "TARGET_VERIFY")
    workspace=None if kind=="eager" else capture_gate(child,kind=="processor_hook")
    batch=FB(forward_mode=fmode,dflash_verify_raw_buffer=workspace,
             capture_hidden_mode=NS(is_full=lambda:False,need_capture=lambda:False),
             return_logprob=False,mixed_verify_sharded_logits=kind=="mixed",
             mixed_verify_logits_indices=torch.arange(len(raw)) if kind=="mixed" else None)
    if kind=="mixed":
        batch.extend_seq_lens_cpu=[len(raw)]
        batch.extend_logprob_start_lens_cpu=[0]
        batch.extend_seq_lens=torch.tensor([len(raw)])
    output=p.forward(None,raw,None,batch)
    tag=getattr(output,"dflash_verify_logit_scale",None)
    marked=child and kind in ("base","unit")
    assert (tag is not None)==marked,(kind,tag)
    native_scale=p.logit_scale
    expected=raw if native_scale is None else raw*native_scale
    assert torch.equal(output.next_token_logits,raw if marked else expected)
    if seen:
        assert len(seen)==1 and torch.equal(seen[0],expected)
    view=replay(output,batch,8)
    assert view.next_token_logits.shape[0]==8 and view.next_token_logits.data_ptr()==output.next_token_logits.data_ptr()
    assert getattr(view,"dflash_verify_logit_scale",None)==tag
    before=output.next_token_logits.clone()
    p.logit_scale=13.0 # A captured operand carries its original scalar through replay.
    livebatch=NS(has_grammar=route=="grammar",return_logprob=route=="logprob")
    result,finished=consume(view,livebatch,sample_info(route))
    oracle=softcap(expected[:8].repeat(1,2)[:,:p.vocab_size],20)
    if route=="greedy":
        assert torch.equal(result,oracle.argmax(dim=1)) and finished.next_token_logits is None
    else:
        assert result is None and torch.equal(finished.next_token_logits,oracle)
    assert getattr(finished,"dflash_verify_logit_scale",None) is None
    assert torch.equal(before,output.next_token_logits)
    return result if result is not None else finished.next_token_logits,marked


if __name__=="__main__":
    torch.set_num_threads(1)
    raw=torch.arange(12*17,dtype=torch.float32).reshape(12,17)%21-12
    raw[0,:]=-20
    raw[0,:2]=torch.tensor([0x41000011,0x41000012],dtype=torch.int32).view(torch.float32)
    raw[1,:]=-100
    raw[1,[2,9]]=10000
    counts={}
    with torch.inference_mode(),patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True):
        for kind in ("base","eager","mixed","observer","override","processor_hook","unit","zero","no_scale","subnormal"):
            for route in ("greedy","sample","custom","bias","penalty","grammar","logprob"):
                old,_=trial(False,kind,raw,route)
                new,marked=trial(True,kind,raw,route)
                assert torch.equal(old,new)
                counts[kind]=counts.get(kind,0)+1
    out={"passed":True,"pairs":sum(counts.values()),"by_producer":counts,
         "candidate":manifest.get("candidate",candidate.name),"candidate_patch":child_sha,
         "tree":manifest["tree"],"base":u.PIN,"parent_patch":parent_sha,"rows_retained":8,"capture_rows":12,
         "sources":{label:{p:sha256(read(p,child).encode()).hexdigest() for p in
                    (LP,GPU,WORKER,"layers/dflash_context_projection.py")} for label,child in
                    (("V235",False),("candidate",True))},
         "scope":"CPU execution of gated workspace allocation, pinned processor, full-graph output slicer and the actual DFLASH verification consumer block. Mock TP transport, CPU softcap/winner and simulated hardware capability."}
    (args.output or candidate/"independent_graph.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="sources"}))
