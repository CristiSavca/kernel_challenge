"""Trace V206's actual model and expert dispatch using shape-only device operations."""
from __future__ import annotations
import ast
import contextlib
import hashlib
import json
import math
import re
import subprocess
import sys
from enum import IntEnum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
REPO = CAND.parent / "sglang"
V = json.loads((HERE / "validation.json").read_text())
PATCH = (HERE / "patch.diff").read_bytes()
assert hashlib.sha256(PATCH).hexdigest() == V["patch_sha256"]
SECTIONS = {s.splitlines()[0].split()[1][2:]:s.splitlines(keepends=True)
            for s in PATCH.decode().split("diff --git ")[1:]}
SRT = "python/sglang/srt/"
LAYERS = SRT + "layers/muse_spark/"


def source(path):
    base = subprocess.run(["git","-C",str(REPO),"show",V["base"]+":"+path],
                          capture_output=True)
    lines = base.stdout.decode().splitlines(keepends=True)
    result = []
    pos = 0
    active = False
    for line in SECTIONS.get(path, ()):
        if line.startswith("@@"):
            start = max(0,int(re.match(r"@@ -(\d+)",line)[1])-1)
            result.extend(lines[pos:start])
            pos, active = start, True
        elif active and line[0] in " -":
            assert lines[pos] == line[1:]
            if line[0] == " ":
                result.append(line[1:])
            pos += 1
        elif active and line.startswith("+"):
            result.append(line[1:])
    return "".join(result + lines[pos:])


def extract(path, names, cls=None):
    nodes = ast.parse(source(path)).body
    if cls:
        nodes = next(n.body for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls)
    funcs = [n for n in nodes if getattr(n,"name",None) in names]
    assert len(funcs)==len(names)
    module = ast.Module(funcs,[])
    scope = dict(globals())
    exec(compile(module,path,"exec"),scope)
    return {n:scope[n] for n in names}


def module(name):
    m=ModuleType(name)
    sys.modules[name]=m
    return m


class Tensor:
    def __init__(self,*shape,dtype="bf16"):
        self.shape,self.dtype,self.device = shape,dtype,"cuda"
    def new_empty(self,shape):
        return Tensor(*(shape if isinstance(shape,tuple) else (shape,)),dtype=self.dtype)
    new_zeros = new_empty
    def view(self,*shape):
        return Tensor(*shape,dtype=self.dtype)
    def __getitem__(self,key):
        return self.new_empty(key.stop)
    def numel(self):
        return math.prod(self.shape)
    def is_contiguous(self):
        return True
    def data_ptr(self):
        return id(self)
    def float(self):
        return Tensor(*self.shape,dtype="float32")


trace=[]
def mark(name):
    trace.append(name)
def mark_return(name,value):
    mark(name)
    return value


class IndexedKernel:
    def __init__(self,name):
        self.name=name
    def __getitem__(self,grid):
        return lambda *a,**kw:mark(self.name)


ctx=NS(sp_active=False,compile=False,rows=0)
stream=NS(wait_stream=lambda other:None)
torch=NS(bfloat16="bf16",float16="float16",int32="int32",int64="int64",float32="float32",
         contiguous_format="contiguous",empty_like=lambda x, dtype=None, **k:Tensor(*x.shape,dtype=dtype or x.dtype),
         compiler=NS(is_compiling=lambda:ctx.compile),cuda=NS(is_available=lambda:True,
         current_stream=lambda:stream,is_current_stream_capturing=lambda:True,
         get_device_capability=lambda device:(9,0),stream=lambda _:contextlib.nullcontext()))
triton=NS(next_power_of_2=lambda n:1<<(n-1).bit_length())
layernorm_sp=NS(sp_num_tokens=lambda:ctx.rows,
    sp_exit_gather=lambda x,n:Tensor(n,*x.shape[1:],dtype=x.dtype),
    reduce_scatter_token_rows=lambda x,g:mark_return("sp_reduce_scatter",Tensor(x.shape[0]//8,x.shape[1])))
get_parallel=lambda:NS(tp_group=NS(ca_comm=None),attn_dp_size=1)
get_forward=lambda:ctx
get_global_experts_capturer=lambda:None
is_dp_attention_enabled=lambda:False
_is_cpu=False
_use_split_bf16_router=lambda n:False
bf16_gemm_dispatch=lambda x,w,b:Tensor(x.shape[0],w.shape[0])
gemm_ag_supported=lambda *a:False
tensor_model_parallel_all_reduce=lambda x:mark_return("all_reduce",x)
is_in_breakable_cuda_graph=lambda:ctx.sp_active
_MAX_DECODE_SLOTS=2048
_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256
_GROUPED_COMPUTE="fixture"
_expert_scaled_silu_module=lambda:NS(run=lambda *a:mark("legacy_silu"))
_plan_grouped_decode=IndexedKernel("plan")
_gather_grouped_decode=IndexedKernel("gather")
fused_post_expert_norm_weighted_sum=lambda a,s,**kw:mark_return("finalize",kw.get("out",Tensor(s.shape[0],4096)))
h13=LAYERS+"decode_gate_up.py"
h2=LAYERS+"decode_mma_pair.py"
hp=LAYERS+"muse_spark_humming_moe.py"
fp=LAYERS+"muse_spark_fp4_moe.py"
model=SRT+"models/muse_spark.py"
name="sglang.srt.layers.muse_spark."
for filename,subset in ((h13,["run_gate_up"]),(h2,["run_down"]),(hp,["humming_decode_eligible","humming_grouped_decode_outputs"])):
    mod=module(name+Path(filename).stem)
    vars(mod).update(extract(filename,subset))
humming=module("humming")
vendor=module("humming.layer")
vendor.HummingMethod=NS(forward_layer=lambda obj,**kw:mark("vendor_"+kw["sublayer_name"]))
humming.ops=NS(launch_kernel=lambda **kw:mark("small_W2_M"+str(kw["configs"])))
mod.local_humming_expert_ids=lambda layer,ids:ids
mod.humming_expert_outputs=lambda layer,x,ids:mark_return("indexed",(x,ids))

class Gate:
    weight=Tensor(256,8192,dtype="float32")
    def __call__(self,x):
        return Tensor(x.shape[0],256)
def route(_,logits,k,always,*,want_packed):
    rows=logits.shape[0]
    return Tensor(rows,k),Tensor(rows,k,dtype="int32"),None

def tune(cand):
    cfg=json.loads((next(CAND.glob(cand)) / "config.json").read_text())["configs"]
    return [(item["bounds"][1],item["block"][0]) for item in cfg.values()]


ex_methods=extract(fp,["_forward_w4a16","forward"],"MuseSpark1xFp4Experts")
Experts=type("Experts",(),dict(ex_methods,__call__=ex_methods["forward"]))
mo_methods=extract(model,["forward","_forward_tp","_forward_experts","_reduce_and_unpool",
                          "_forward_sp_experts","_forward_sp_replicated_experts"],"MuseSpark1xMoE")
MoE=type("MoE",(),mo_methods)
ForwardMode=extract(SRT+"model_executor/forward_batch_info.py",["ForwardMode"])["ForwardMode"]
md=MoE()
e=md.experts=Experts()
md.finalize_ar_workspace=None
md.moe_tp_size,md.tp_size,md.top_k=1,8,8
md.pre_expert_proj=NS(weight=Tensor(4096,8192))
md.post_expert_proj=NS(weight=Tensor(8192,4096))
md.pre_expert_norm=lambda x:x
md.gate=Gate()
md._side_stream=lambda:stream
md._routing_op=lambda:route
md.nvfp4_checkpoint=md.sp_replicate_expert_input=md.router_in_fp32=True
e.use_w4a16=e.use_humming=True
e.post_expert_norm=e._post_expert_norm=object()
e._post_expert_norm_weight=Tensor(4096)
e._post_expert_norm_eps,e._post_expert_norm_weight_first=1e-6,True
e._expert_map=object()
e.intermediate_size_per_partition,e.num_local_experts,e.global_num_experts=4096,32,256
e.w2_global_scale=e.w13_global_scale=Tensor(32)
e.w13_weight=e.w13_weight_scale=Tensor(32,256,16384)
e.w2_weight=e.w2_weight_scale=Tensor(32,256,8192)
e._grouped_humming_and_combine=lambda *a:mark("prompt_grouped")
tunings=dict(w13=tune("v191_*"),w2=tune("v189_*"))
addresses=tuple((t.data_ptr(),t.shape) for t in (e.w13_weight,e.w13_weight_scale))
launcher=NS(run=lambda inputs,output,keys,offsets,scales,maps,handle,shared,width,sms:
            mark_return("small_W13_M"+str(width),True))
choices={w:(None,1,90000,w) for _,w in tunings["w13"]}
result=[]
for mode,rows,sp,change,expected in (
    ("DECODE",1,False,"","small_W13_M8"),
    ("DECODE",32,False,"","small_W13_M8"),
    ("TARGET_VERIFY",8,False,"","small_W13_M8"),
    ("TARGET_VERIFY",112,False,"","small_W13_M8"),
    ("TARGET_VERIFY",113,False,"","small_W13_M16"),
    ("TARGET_VERIFY",128,False,"","small_W13_M16"),
    ("TARGET_VERIFY",216,False,"","small_W13_M16"),
    ("TARGET_VERIFY",217,False,"","small_W13_M24"),
    ("TARGET_VERIFY",256,False,"","small_W13_M24"),
    ("TARGET_VERIFY",257,False,"","indexed"),
    ("TARGET_VERIFY",128,False,"no_bank","vendor_w13"),
    ("TARGET_VERIFY",256,False,"no_w2","vendor_w2"),
    ("TARGET_VERIFY",128,False,"compiling","indexed"),
    ("EXTEND",128,False,"","indexed"),
    ("MIXED",128,False,"","indexed"),
    ("EXTEND",16384,True,"","prompt_grouped"),
    ("MIXED",16384,True,"","prompt_grouped"),
):
    trace.clear()
    ctx.sp_active,ctx.compile,ctx.rows=sp,change=="compiling",rows
    e.humming_decode_tunings=tunings
    e.humming_decode_gate_up=None if change=="no_bank" else (launcher,None,addresses,choices,132)
    e.humming_decode_down={} if change=="no_w2" else {w:w for _,w in tunings["w2"]}
    local=rows//8 if sp else rows
    out=md.forward(Tensor(local,8192),forward_batch=NS(forward_mode=ForwardMode[mode]))
    assert out.shape==(local,8192) and expected in trace,(mode,rows,change,trace)
    if expected.startswith("small_W13"):
        assert trace==["plan","gather",expected,expected.replace("13","2"),"finalize","all_reduce"],trace
    if expected in ("indexed","prompt_grouped"):
        assert all("small_" not in x for x in trace)
    result.append(dict(mode=mode,rows=rows,sp=sp,guard=change or None,trace=list(trace)))
print(json.dumps(dict(patch_sha256=V["patch_sha256"],cases=result,
   boundary="Actual model/expert methods, eligibility, Humming tier selection, optional W13/W2 wrappers and ForwardMode enum. Tensor allocation, projections/router, graph/stream primitives, Triton launches, CUDA launchers, collectives, finalizer and fallback compute are shape-only test doubles; no GPU or numerical claim."),indent=2))
