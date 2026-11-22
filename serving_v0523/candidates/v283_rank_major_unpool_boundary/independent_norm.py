"""Compile the original and candidate single-CTA bodies and execute both via their real Python dispatch."""
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional,Tuple
from unittest.mock import patch

import torch

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
cu=Path("python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh")
py=Path("python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py")
base="297e54fd105e974cdf401e5b4d20253a22c16f90"
readbase=lambda p:subprocess.check_output(["git","-C",stage,"show",base+":"+str(p)]).decode()
before=readbase(cu)
after=(stage/cu).read_text()
pinned_py=readbase(py)
candidate_py=(stage/py).read_text()
def hashtext(s):return hashlib.sha256(s.encode()).hexdigest()
def cuda(src):
    body=src[src.index("// One CTA per row;"):src.index("// Multi-CTA path")]
    signature="__global__ void __launch_bounds__(kNumThreads)"
    assert body.count(signature)==1
    body=body.replace(signature,"Step")
    f="cta::reduce_sum<kNumThreads>(sum_sq, smem)"
    g="cta::reduce_sum<kNumThreads>(pre_sum_sq, kNumWarps == 1 ? smem : smem + kNumWarps)"
    assert f in body and g in body
    return body.replace(f,"co_await Reduce{sum_sq}").replace(g,"co_await Reduce{pre_sum_sq}")

class Params(C.Structure):
    _fields_=[(n,C.c_void_p) for n in ('branch','stream','output','new_stream','router','post_weight','pre_weight','alpha','beta')]+[
        ('post_eps',C.c_float),('pre_eps',C.c_float),('m',C.c_int),('channel',C.c_bool),('preweight',C.c_bool),('pre_bf16',C.c_bool)]

def dispatch(src):
    tree=ast.parse(src)
    keep=[x for x in tree.body if ((isinstance(x,ast.FunctionDef) and x.name in ("_launch_config","cluster_size_for_num_tokens","_cluster_size_for_shape","_cluster_num_threads","_cluster_launch_config","fused_norm_residual"))
        or (isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id in ["_VEC","_THREAD_CHOICES","_SMALL_M","_SMALL_M_THREAD_CHOICES","_CLUSTER_SIZES"]))]
    ns={"torch":torch,"Optional":Optional,"Tuple":Tuple}
    exec(compile(ast.Module(keep,[]),"live_python","exec"),ns)
    return ns

def raw(t):return None if t is None else t.data_ptr()
def aswords(t):return t.view(torch.int16 if t.dtype==torch.bfloat16 else torch.int32)
records=[];declines=[];cluster_cases=[];cluster_errors=[];mismatch=0
with tempfile.TemporaryDirectory(prefix="okc-v283-norm-") as tmp:
    temp=Path(tmp)
    common=before[before.index("struct NormResidualParams"):before.index("// One CTA per row;")]
    assert common==after[after.index("struct NormResidualParams"):after.index("// One CTA per row;")]
    (temp/"common.inc").write_text(common);(temp/"before.inc").write_text(cuda(before));(temp/"after.inc").write_text(cuda(after))
    altered="const int rank = idx / shard_vecs;"
    assert cuda(after).count(altered)==1
    (temp/"wrong.inc").write_text(cuda(after).replace(altered,"const int rank = (idx / shard_vecs + 1) % kColumnRanks;"))
    subprocess.run(["clang++","-std=c++20","-O1","-ffp-contract=fast","-fPIC","-shared","-I",str(temp),str(here/"independent_norm.cpp"),"-o",str(temp/"norm.so")],check=True)
    lib=C.CDLL(str(temp/"norm.so"))
    native=lib.norm_row
    native.argtypes=[C.c_int]*4+[Params];native.restype=C.c_int
    original,candidate=dispatch(pinned_py),dispatch(candidate_py)
    rng=torch.Generator().manual_seed(93003)
    torch.set_num_threads(1)
    def rand(*shape):return torch.randn(*shape,generator=rng)
    for m in (1,8,128,191,192,256):
        hid=8192
        plain=rand(m,hid).bfloat16()
        stream=rand(m,hid)
        boundaries=set([0,1,hid-1]+[v+delta for v in range(1024,hid,1024) for delta in (-1,0,1)])
        # Negative and positive zero, two NaN payloads, infinities and smallest subnormals.
        tags=(0,0x8000,0x7fc1,0xffc2,0x7f80,0xff80,0x0001,0x8001)
        for i,col in enumerate(sorted(boundaries)):
            if m>1:plain[1,col].view(torch.int16).fill_((tags[i%len(tags)]+32768)%65536-32768)
        plain[0].view(torch.int16)[::2]=-32768
        plain[0].view(torch.int16)[1::2]=0
        stream[0,::2]=-0.;stream[0,1::2]=0.
        post=rand(hid).bfloat16(); pre=rand(hid).bfloat16()
        ga,gb=rand(hid),rand(hid)
        selected=sorted(set((0,min(m-1,1),m//2,m-1)))
        for mode in (0,1,2,3):
            use_scale=mode in (0,3);router=mode in (0,2)
            args=dict(post_eps=3e-6,pre_eps=7e-6,post_norm_is_input_scale=use_scale,
                post_weight=None if mode==1 else post,pre_weight=None if mode==1 else pre,
                alpha=None if mode==1 else ga[:1] if mode==2 else ga,beta=None if mode==1 else gb[:1] if mode==2 else gb,
                want_router=router,pre_norm_bf16=mode==2)
            def call(ns,branch,how,error=RuntimeError):
                visits=[]
                def module(h,t,pref,cs,*ranks):
                    r=ranks[0] if ranks else 1
                    visits.append((h,t,pref,cs,r))
                    assert h==8192 and cs==1 and t==(256 if m<192 else 512) and pref==(m<192)
                    if how=="decline" and r>1:raise error("deliberate rank-major JIT refusal")
                    format_id= -1 if how=="original" else -4 if how=="wrong" else r
                    class Bound:
                        def __getattr__(self,name):
                            assert name==("run_scale" if use_scale else "run_std")+("_router" if router else "")
                            def run(b,s,out,new,rt,pw,prew,a,beta,posteps,preeps,bf):
                                if r==1:
                                    assert b.shape==plain.shape and torch.equal(aswords(b),aswords(plain))
                                else:assert b.shape==(r,m,hid//r) and b.is_contiguous()
                                pp=Params(*[raw(x) for x in (b,s,out,new,rt,pw,prew,a,beta)],posteps,preeps,m,a is not None and a.numel()>1,prew is not None,bf)
                                for row in selected:assert native(t,format_id,mode,row,pp)==0
                            return run
                    return Bound()
                with patch.dict(ns,_jit_module=module),patch.object(torch.cuda,"get_device_capability",return_value=(9,0)):
                    result=ns["fused_norm_residual"](branch,stream,**args)
                return [aswords(x[selected]).clone() for x in result if x is not None],visits
            ref,_=call(original,plain,"original")
            for r in (1,4,8):
                rank=plain if r==1 else plain.reshape(m,r,-1).transpose(0,1).contiguous()
                got,visits=call(candidate,rank,"candidate")
                assert len(got)==len(ref) and all(torch.equal(a,b) for a,b in zip(ref,got))
                records.append((m,r,mode,visits[0][1],len(selected),sum(x.numel() for x in got)))
                if r==4 and m in (8,192) and mode==0:
                    wrong,_=call(candidate,rank,"wrong")
                    mismatch+=sum(int(torch.count_nonzero(a!=b)) for a,b in zip(ref,wrong))
                if r in (4,8) and m in (8,192) and mode in (0,1):
                    err=OSError if mode==0 else RuntimeError
                    recovered,visits=call(candidate,rank,"decline",err)
                    assert [v[-1] for v in visits]==[r,1] and all(torch.equal(a,b) for a,b in zip(ref,recovered))
                    declines.append((m,r,mode,err.__name__))
    assert mismatch
    # The two wrappers must call the same original cluster entry with exactly the same 2D bytes.
    for m,r,arch,requested in ((8,4,9,2),(16,8,9,4),(32,4,9,8),(64,8,9,16),(8,8,10,None),(16,4,10,None),(32,8,10,None),(64,4,10,None)):
        h=8192
        b=rand(m,h).bfloat16()
        b[0,:6]=torch.tensor([0,-32768,0x7fc1,-63,1,-32767],dtype=torch.int16).view(torch.bfloat16)
        s=rand(m,h)
        rank=b.reshape(m,r,h//r).transpose(0,1).contiguous()
        def cluster_call(ns,branch,options):
            calls=[]
            def module(*key):
                assert len(key)==4 and key[-1]>1
                class Cluster:
                    def run_scale_router(self,in_b,in_s,out,new,router,*rest):
                        assert in_b.is_contiguous() and in_b.shape==b.shape and torch.equal(aswords(in_b),aswords(b))
                        assert in_s.is_set_to(s)
                        calls.append(key)
                        # Fingerprint the exact inputs reaching the previously selected native operator.
                        out.copy_(in_b);new.copy_(in_s+in_b.float());router.copy_(in_b.float())
                return Cluster()
            with patch.dict(ns,_jit_module=module),patch.object(torch.cuda,"get_device_capability",return_value=(arch,0)):
                out=ns["fused_norm_residual"](branch,s,post_eps=1e-6,pre_eps=1e-6,post_norm_is_input_scale=True,want_router=True,**options)
            return [aswords(t) for t in out],calls
        options=dict(cluster_size=requested)
        a,ac=cluster_call(original,b,options);z,zc=cluster_call(candidate,rank,options)
        assert ac==zc and len(ac)==1 and all(torch.equal(v,w) for v,w in zip(a,z))
        cluster_cases.append(dict(rows=m,ranks=r,capability=arch,explicit_cluster=requested,original_module_key=ac[0]))
    for options in (dict(cluster_size=2,pre_norm_bf16=True),dict(cluster_size=2,launch_config=(256,True)),dict(cluster_size=3)):
        reasons=[]
        for ns,tensor in ((original,b),(candidate,rank)):
            try:cluster_call(ns,tensor,options)
            except ValueError as exc:reasons.append(str(exc))
            else:raise AssertionError("invalid cluster configuration admitted")
        assert len(reasons)==2 and reasons[0]==reasons[1]
        cluster_errors.append(reasons[0])
    import _ctypes
    _ctypes.dlclose(lib._handle)
print(json.dumps(dict(source_hashes={"native":hashtext(before),"candidate":hashtext(after),"native_dispatch":hashtext(pinned_py),"candidate_dispatch":hashtext(candidate_py)},
    hidden_size=8192,token_counts=sorted({x[0] for x in records}),ranks=sorted({x[1] for x in records}),mode_cases=len(records),BF16_and_FP32_words=sum(x[-1] for x in records),
    cases_by_thread_count={t:sum(x[3]==t for x in records) for t in (256,512)},fallbacks=declines,
    cluster_fallbacks=cluster_cases,cluster_validation_errors=cluster_errors,
    circularly_shifted_shard_negative_control_words_changed=mismatch,
    scope="Identical compiled production C++ before/after bodies. A host coroutine per CUDA thread reproduces the two CTA reductions using the pinned Hopper XOR tree. Native square root, BF16 conversion, memory primitives and PDL are emulated, so this compares layout and operation/order using one host floating-point implementation. No GPU or H200 absolute-output claim."),indent=2))
