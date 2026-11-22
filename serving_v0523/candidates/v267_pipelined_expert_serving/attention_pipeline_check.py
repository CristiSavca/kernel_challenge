"""Run consecutive QKV and output SP calls against the native path on real TP groups."""
import ast
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
SRT = Path(sys.argv[1]).resolve() / "python/sglang/srt"
QKV = SRT/"layers/communication/sp_half_qkv.py"
OUT = SRT/"layers/communication/sp_attention_return.py"
SP = SRT/"layers/layernorm_sp.py"
prior = runpy.run_path(str(HERE.parent/"v264_pipelined_attention_return/check_native.py"))
base = prior["base"]


def reconstruct_return_control(tmp):
    """Make a read-only release control without touching the source checkout's index or objects."""
    src=HERE.parent/"v264_pipelined_attention_return"
    m=json.loads((src/"validation.json").read_bytes())
    d=Path(tmp)
    objects=d/"objects"
    objects.mkdir()
    repo=SRT.parents[2]
    alt=subprocess.check_output(["git","-C",str(repo),"rev-parse","--git-path","objects"],text=True).strip()
    env={**os.environ,"GIT_INDEX_FILE":str(d/"parent-index"),"GIT_OBJECT_DIRECTORY":str(objects),
         "GIT_ALTERNATE_OBJECT_DIRECTORIES":str((repo/alt).resolve())}
    def git(*args):
        return subprocess.check_output(["git","-C",str(repo),*args],env=env)
    git("read-tree", m["base"])
    git("apply","--cached","--whitespace=error-all", str(src/"patch.diff"))
    assert git("write-tree").decode().strip()==m["tree"]
    for path in ("layers/communication/sp_attention_return.py","layers/layernorm_sp.py"):
        (d/Path(path).name).write_bytes(git("show", m["tree"]+":python/sglang/srt/"+path))
    return m["patch_sha256"]


def precision_readout(old,new):
    """Use a fixed synthetic vocabulary and construct a separate two-logit sensitivity witness."""
    def ordered(t):
        bits=t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits<0,-(bits&32767),bits)
    old_order,new_order=ordered(old),ordered(new)
    delta=(new_order-old_order).abs()
    index=torch.nonzero(delta)
    head=torch.randn((64,old.shape[-1]),generator=torch.Generator().manual_seed(267)) / old.shape[-1]**0.5
    old_logits=old.float() @ head.t()
    new_logits=new.float() @ head.t()
    ret=dict(max_bf16_ulp=int(delta.max()),fixed_head_compared_rows=old.shape[0],
             fixed_head_argmax_different=int((old_logits.argmax(-1)!=new_logits.argmax(-1)).sum()),
             fixed_head_max_logit_abs=float((old_logits-new_logits).abs().max()))
    if index.numel():
        row,col=index[0].tolist()
        a,b=float(old[row,col]),float(new[row,col])
        midpoint=(a+b)/2
        ret["constructed_two_logit_example"]=dict(owner_row=row,feature=col,old_value=a,new_value=b,
                        constant_competitor=midpoint,old_argmax=int(midpoint>a),new_argmax=int(midpoint>b))
        assert int(midpoint>a)!=int(midpoint>b)
    return ret


def worker(rank, size, tmp):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+str(Path(tmp)/f"tp{size}"), rank=rank,
                            world_size=size, timeout=timedelta(minutes=5))
    schedule = prior["Schedule"]()
    group = prior["Group"](size,rank,schedule)
    comm = group.pynccl_comm
    state = NS(tokens=0, specialized=False)
    install, load, Method = base["install"], base["load"], base["LinearMethod"]
    parallel = lambda: NS(tp_group=group,attn_tp_group=group)
    def register(fake_impl):
        def decorate(fn):
            op = torch.library.custom_op("v267_probe::"+fn.__name__, mutates_args=(),device_types="cpu")(fn)
            op.register_fake(fake_impl)
            return op
        return decorate
    env=dict(torch=torch, get_parallel=parallel, get_lora=lambda:NS(enable_lora=False),
             get_exec=lambda:NS(deterministic=NS(enable_deterministic_inference=False)),register_custom_op=register)
    load(QKV,[n.name for n in ast.parse(QKV.read_bytes()).body if isinstance(n,ast.FunctionDef)],env)
    qmod=install("sglang.srt.layers.communication.sp_half_qkv",env)
    outenv={"torch":torch}
    load(OUT,("striped_attention_return","striped_return_eligible"),outenv)
    omod=install("sglang.srt.layers.communication.sp_attention_return",outenv)
    sp=dict(torch=torch,os=os,get_parallel=parallel,sp_num_tokens=lambda:state.tokens,
            sp_fused_matmul_eligible=lambda _:False,ceil_align=lambda n,s:((n+s-1)//s)*s)
    load(SP,("column_parallel_g_matmul","sp_exit_gather","row_parallel_gbar_matmul","reduce_scatter_token_rows"),sp)
    install("sglang.srt.layers.layernorm_sp",sp)
    parent_sp=dict(sp)
    load(Path(tmp)/"layernorm_sp.py",("row_parallel_gbar_matmul","reduce_scatter_token_rows"),parent_sp)
    parent_op=dict(torch=torch)
    load(Path(tmp)/"sp_attention_return.py",("striped_return_eligible","striped_attention_return"),parent_op)
    parent_module=NS(**parent_op)
    backend=NS(is_cutedsl=lambda:state.specialized,is_gemv=lambda:False)
    install("sglang.srt.layers.quantization.unquant",dict(UnquantizedLinearMethod=Method,get_bf16_gemm_backend=lambda:backend,
                                                          bf16_gemm_dispatch=torch.nn.functional.linear))
    install("sglang.srt.batch_invariant_ops",dict(is_batch_invariant_mode_enabled=lambda:False))
    install("sglang.srt.environ",dict(envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False))))
    original_allreduce=dist.all_reduce
    votes=[]
    def counted_vote(tensor,*args,**kwargs):
        if tensor.dtype==torch.int32:
            assert tensor.shape==(2,) and kwargs["op"]==dist.ReduceOp.MIN
            before=tensor.tolist()
            original_allreduce(tensor,*args,**kwargs)
            votes.append([before,tensor.tolist()])
            return
        return original_allreduce(tensor,*args,**kwargs)
    streams=[]
    def make_stream(device):
        streams.append(prior["Stream"](schedule,"transfer"))
        return streams[-1]
    compute=schedule.compute
    # Successful joins for each endpoint must refer to the same side stream instance.
    original_use=schedule.use
    uses=[]
    def observed_use(stream):
        uses.append(stream)
        return original_use(stream)
    totals=dict(pairs=0,QKV_BF16=0,owner_BF16=0,q_fused=0,o_fused=0,shared_forward=0,
                startup_vectors=0,allocated_streams=0,failed_joins=0,recovered_pairs=0)
    geometries=[]
    precision=[]
    __import__("torch._dynamo")
    with ExitStack() as mocks, torch.no_grad():
        for obj,name,value in ((torch.cuda,"get_device_capability",lambda dev:(9,0)),(torch.cuda,"Stream",make_stream),
                               (torch.cuda,"current_stream",lambda device: schedule.current),
                               (torch.cuda,"stream",observed_use),(dist,"all_reduce",counted_vote)):
            mocks.enter_context(patch.object(obj,name,value))
        for want_q,want_o in ((True,True),(False,True),(True,False),(False,False)):
            # New model instances exercise all four independent rank votes.
            expected_q=want_q and size>=4 and size%2==0
            q=NS(weight=torch.ones(32,32,dtype=torch.bfloat16),quant_method=Method(),gather_output=False,bias=None)
            o=NS(weight=torch.ones(256,16,dtype=torch.bfloat16),quant_method=Method(),tp_group=group,bias=None,
                 _sp_unfused_collective=True)
            if not want_q and rank==0:
                q.gather_output=True
            if not want_o and rank==size-1:
                o.bias=torch.zeros(256,dtype=torch.bfloat16)
            before_votes=len(votes)
            before_streams=len(streams)
            comm.group_shapes.clear()
            env["enable_sp_attention_pipeline"]([q],[o],group)
            assert len(votes)==before_votes+1 and votes[-1][1]==[int(expected_q),int(want_o)]
            assert len(streams)-before_streams==int(expected_q or want_o)
            assert len(comm.group_shapes)==2*int(expected_q)
            assert (getattr(q,"_sp_half_qkv",None) is q.quant_method)==expected_q
            assert (getattr(o,"_sp_return_method",None) is o.quant_method)==want_o
            if expected_q and want_o:
                assert o._sp_return_stream is group.sp_qkv_cohort_stream is streams[-1]
            totals["startup_vectors"]+=1
            totals["allocated_streams"]+=len(streams)-before_streams
            q.gather_output=False
            o.bias=None
            # Perturbed runtime shapes reuse the same startup-authorized stream.
            modes=["ar","rs"]
            if want_q and want_o:
                modes+=["q_reload","o_reload","special","partial_switch","q_failure","o_failure","recovery"]
                if size==4:
                    modes+=["unscaled"]
            for mode in modes:
                state.specialized=mode=="special"
                group.mode="switch" if mode=="partial_switch" else "ar"
                os.environ["NCCL_NVLS_ENABLE"]="0" if mode=="rs" else "1"
                local,width=(2048,8192) if mode=="unscaled" else (512,256)
                group.width=width
                drop=0 if mode=="unscaled" else size-1
                state.tokens=local*size-drop
                gen=torch.Generator().manual_seed(26700+rank)
                x=torch.randint(-2,3,(local,32),generator=gen).to(torch.bfloat16)/8
                q.weight=torch.randint(-2,3,(32,32),generator=gen).to(torch.bfloat16)/8
                o.weight=torch.randint(-2,3,(width,16),generator=gen).to(torch.bfloat16)/8
                original_q,original_o=q.quant_method,o.quant_method
                if mode=="q_reload": q.quant_method=Method()
                if mode=="o_reload": o.quant_method=Method()
                qs=NS(**{k:v for k,v in vars(q).items() if not k.startswith("_sp_")})
                os_=NS(**{k:v for k,v in vars(o).items() if k not in ("_sp_return_stream","_sp_return_method")})
                def transform(proj):
                    # Per-row, rank-local nonlinear operations stand in for the attention body.
                    return (proj[:,:16] * torch.sigmoid(proj[:,16:])).contiguous()
                ref_q=sp["column_parallel_g_matmul"](qs,x,None)
                ref_o=sp["row_parallel_gbar_matmul"](os_,transform(ref_q),None).clone()
                q_fast=expected_q and mode!="q_reload"
                o_fast=want_o and mode not in ("o_reload","special","partial_switch")
                schedule.trace.clear()
                comm.group_shapes.clear()
                uses.clear()
                real_mm=torch.mm
                visits=[]
                def mm(a,b,out):
                    visits.append(tuple(out.shape))
                    if mode=="q_failure" and q_fast and len(visits)==1:
                        raise RuntimeError("QKV projection failure")
                    if mode=="o_failure" and o_fast and out.shape[-1]==width//2 and visits.count(tuple(out.shape))==2:
                        raise RuntimeError("attention return failure")
                    return real_mm(a,b,out=out)
                # Only the output geometry gate is scaled in the small fixtures.
                gate=outenv["striped_return_eligible"] if mode=="unscaled" else lambda n,w:w>=256
                try:
                    with patch.object(torch,"mm",mm),patch.object(omod,"striped_return_eligible",gate):
                        got_q=sp["column_parallel_g_matmul"](q,x,None)
                        got_o=sp["row_parallel_gbar_matmul"](o,transform(got_q),None)
                except RuntimeError as e:
                    assert (mode=="q_failure" and q_fast and str(e)=="QKV projection failure") or (
                            mode=="o_failure" and o_fast and str(e)=="attention return failure")
                    assert schedule.trace[-1]==("wait","compute","transfer")
                    assert comm.disabled and comm.ops is None and schedule.current is compute
                    totals["failed_joins"]+=1
                else:
                    assert not (mode=="q_failure" and q_fast) and not (mode=="o_failure" and o_fast)
                    assert torch.equal(got_q.view(torch.int16),ref_q.view(torch.int16))
                    different=int((got_o.view(torch.int16)!=ref_o.view(torch.int16)).sum())
                    assert len(comm.group_shapes)==2*int(q_fast)
                    assert len(uses)==int(q_fast)+int(o_fast)
                    assert all(s is streams[-1] for s in uses)
                    waits=[a for a in schedule.trace if a[0]=="wait"]
                    assert waits==[("wait","transfer","compute"),("wait","compute","transfer")]*(int(q_fast)+int(o_fast))
                    totals["pairs"]+=1
                    totals["QKV_BF16"]+=got_q.numel()
                    totals["owner_BF16"]+=got_o.numel()
                    totals["q_fused"]+=q_fast
                    totals["o_fused"]+=o_fast
                    totals["shared_forward"]+=q_fast and o_fast
                    totals["recovered_pairs"]+=mode=="recovery"
                    if mode=="unscaled": geometries.append([size,local,32,32,16,width,got_o.shape[0]])
                    # Some Gloo message sizes can change BF16 addition order. Isolate
                    # that from a change in GEMM arithmetic or in the composed stream.
                    any_diff=torch.tensor([different],dtype=torch.int64)
                    original_allreduce(any_diff,op=dist.ReduceOp.MAX)
                    if any_diff.item():
                        assert o_fast
                        hidden=transform(ref_q)
                        full=torch.nn.functional.linear(hidden,o.weight)
                        parts=[real_mm(hidden,w.t()) for w in o.weight.chunk(2)]
                        local_diff=int((full.view(torch.int16)!=torch.cat(parts,dim=1).view(torch.int16)).sum())
                        # Feed identical native partials through full vs half-width collectives.
                        full_owner=sp["reduce_scatter_token_rows"](full.clone(),group)
                        stripe_control=torch.cat([sp["reduce_scatter_token_rows"](p.contiguous(),group)
                                                  for p in full.chunk(2,dim=1)],dim=1)
                        actual_control=torch.cat([sp["reduce_scatter_token_rows"](p,group) for p in parts],dim=1)
                        assert torch.equal(full_owner.view(torch.int16),ref_o.view(torch.int16))
                        assert torch.equal(got_o.view(torch.int16),actual_control.view(torch.int16))
                        with patch.dict(sys.modules,{"sglang.srt.layers.layernorm_sp":NS(**parent_sp),
                                "sglang.srt.layers.communication.sp_attention_return":parent_module}), \
                             patch.object(parent_module,"striped_return_eligible",gate):
                            standalone=parent_sp["row_parallel_gbar_matmul"](o,hidden,None)
                        assert torch.equal(standalone.view(torch.int16),got_o.view(torch.int16))
                        same_input_mismatch=int((stripe_control.view(torch.int16)!=full_owner.view(torch.int16)).sum())
                        if local_diff==0:
                            assert torch.equal(stripe_control.view(torch.int16),got_o.view(torch.int16))
                        precision.append(dict(tp=size,rank=rank,mode=mode,q_fused=q_fast,local_product_differences=local_diff,
                                              owner_differences=different,controlled_same_input_owner_differences=same_input_mismatch,
                                              standalone_V264_owner_differences_from_native=different,
                                              owner_max_abs=float((got_o.float()-ref_o.float()).abs().max()),
                                              **precision_readout(ref_o,got_o)))
                finally:
                    q.quant_method,o.quant_method=original_q,original_o
    (Path(tmp)/f"{size}-{rank}.json").write_text(json.dumps(dict(totals=totals,geometry=geometries,votes=votes,precision=precision)))
    dist.destroy_process_group()


if __name__=="__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0" if sys.platform=="darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        parent_digest=reconstruct_return_control(tmp)
        for size in (2,4,8): mp.spawn(worker,args=(size,tmp),nprocs=size,join=True)
        records=[json.loads(f.read_bytes()) for f in Path(tmp).glob("*.json")]
    totals={k:sum(r["totals"][k] for r in records) for k in records[0]["totals"]}
    stress=[c for r in records for c in r["precision"]]
    print(json.dumps(dict(patch_sha256=hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest(),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (QKV,OUT,SP)},
        tp=[2,4,8],standalone_V264_sha256=parent_digest,rank_count=len(records),**totals,
        owner_different=sum(c["owner_differences"] for c in stress),
        local_product_different=sum(c["local_product_differences"] for c in stress),
        same_input_full_vs_half_collective_different=sum(c["controlled_same_input_owner_differences"] for c in stress),
        max_owner_abs=max((c["owner_max_abs"] for c in stress),default=0),
        max_owner_bf16_ulp=max((c["max_bf16_ulp"] for c in stress),default=0),
        fixed_head_rows_on_controlled_cases=sum(c["fixed_head_compared_rows"] for c in stress),
        fixed_head_argmax_different=sum(c["fixed_head_argmax_different"] for c in stress),
        precision_cases=stress,
        unscaled_dispatch_geometries=[r["geometry"][0] for r in records if r["geometry"]],
        scope="Real Gloo startup, P2P, all-gather, all-reduce, reduce-scatter and CPU BF16 projections. Small cases scale the O size predicate. Stream ordering is modeled. Between projections arithmetic is a synthetic rank-local function. No H200."),
        indent=2))
