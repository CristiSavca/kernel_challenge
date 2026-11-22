"""Run the standard grouped W13/W2 callback, including live sorting, parent fallbacks and workspace aliases."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as S
from unittest.mock import patch
import torch
here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
src=stage/"python/sglang/srt/layers/muse_spark"
hum=src/"muse_spark_humming_moe.py";w4=src/"muse_spark_fp4_moe.py"
def fn(file,name,cls=None):
    body=ast.parse(file.read_bytes()).body
    if cls:body=next(c.body for c in body if getattr(c,"name",None)==cls)
    node=next(f for f in body if getattr(f,"name",None)==name)
    node.decorator_list=[]
    return ast.unparse(node)
def add(src_,variables):exec(compile("from __future__ import annotations\n"+src_,"production-source","exec"),variables)
def module(name,**attrs):
    obj=ModuleType(name);vars(obj).update(attrs);sys.modules[name]=obj;return obj
utils=module("sglang.kernels.jit.utils",load_jit=None);gate=module("sglang.srt.layers.muse_spark.decode_gate_up")
exec(compile((src/"decode_gate_up.py").read_bytes(),"decode_gate_up.py","exec"),gate.__dict__)
torch.set_num_threads(1);g=torch.Generator().manual_seed(197)
E,H,I=8,128,128
def rand(*shape):return torch.randn(shape,generator=g).to(torch.bfloat16)
scale=rand(E);w2scale=rand(E)
gate_diag=rand(E,H);up_diag=rand(E,H);down_diag=rand(E,H)
events=[];out_checks=0;declines=0;roles=[]
def silu(x,out):
    lhs,rhs=x.float().chunk(2,dim=-1)
    out.copy_((lhs/(1.0+torch.exp(-lhs))*rhs).to(torch.bfloat16))
module("sglang.kernels.ops.activation.activation",silu_and_mul=silu)
def ref_w13(x,e):
    return torch.cat((x.float()*gate_diag[e].float(),x.float()*up_diag[e].float()),dim=-1).to(torch.bfloat16)
class Humming:
    @staticmethod
    def forward_layer(layer,*,inputs,outputs,expert_layout,compute_config,tuning_config,sublayer_name):
        assert inputs.data_ptr()!=outputs.data_ptr() and expert_layout.dtype==torch.int32
        assert inputs.is_contiguous() and outputs.is_contiguous()
        assert int(expert_layout[0])==0 and int(expert_layout[-1])==len(inputs)
        if sublayer_name=="w2" and events and events[-1][0]=="paired":
            assert outputs.data_ptr()==events[-1][1] and inputs.data_ptr()==events[-1][2]
        events.append((sublayer_name,inputs.data_ptr(),outputs.data_ptr()))
        for e,(start,end) in enumerate(zip(expert_layout[:-1],expert_layout[1:])):
            if start==end:continue
            x=inputs[start:end]
            projection=ref_w13(x,e) if sublayer_name=="w13" else (x.float()*down_diag[e].float()).bfloat16()
            outputs[start:end].copy_(projection)
module("humming.layer",HummingMethod=Humming)
class Launch:
    def run_prompt(self,inputs,acts,ids,offsets,factors,maps,handle,shared,width,sms):
        roles.append(("prompt",width))
        assert ids.dtype==torch.int32 and factors is scale and maps is prompt_maps and handle==84 and width==176
        assert not (inputs.data_ptr()==acts.data_ptr())
        if mode=="reject":return False
        for e,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
            if start==end:continue
            assert (ids[start:end]==e).all()
            gates=(ref_w13(inputs[start:end],e)*factors[e]).bfloat16()
            silu(gates,acts[start:end])
        events.append(("paired",inputs.data_ptr(),acts.data_ptr()))
        return True
    def run(self,inputs,acts,keys,offsets,factors,maps,handle,shared,width,sms):
        # Verify the distinct buffer/handle from preparation and genuine sorted-key decoding in a small decode.
        assert maps is decode_maps and handle==width+100 and factors is scale and width in (8,16,24)
        roles.append(("decode",width))
        bits=(len(inputs)-1).bit_length()
        for e,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
            if a==b:continue
            assert ((keys[a:b]>>bits)-1==e).all()
            silu((ref_w13(inputs[a:b],e)*factors[e]).bfloat16(),acts[a:b])
        return True

probe=Launch();prompt_maps=object();decode_maps=object()
layer=S(w13_weight=torch.empty(2,dtype=torch.int32),w13_weight_scale=torch.empty(3,dtype=torch.uint8),
    w13_global_scale=scale,w2_global_scale=w2scale,num_local_experts=E,intermediate_size_per_partition=I,
    humming_grouped_tunings={"w13":"P","w2":"W2"})
bindings=tuple((x.data_ptr(),x.shape) for x in (layer.w13_weight,layer.w13_weight_scale))
def pstate(mode):
    if mode=="disabled":return None
    addrs=bindings if mode!="moved" else ((17,(2,)),bindings[1])
    kernels={} if mode=="missing" else {"P":(None,84,128000,176)}
    return probe,prompt_maps,addrs,kernels,132
layer.humming_decode_gate_up=(probe,decode_maps,bindings,{w:(None,w+100,90000,w) for w in (8,16,24)},132)
env=dict(torch=torch,_GROUPED_COMPUTE="GROUPED",humming_expert_outputs=lambda *_:("indexed",None))
add(fn(hum,"humming_grouped_expert_outputs"),env)
run=env["humming_grouped_expert_outputs"]
counts={}
for n,dt,pattern in [(8192,torch.int32,"mixed"),(16384,torch.int64,"mixed"),(8311,torch.int32,"skewed"),
                     (8193,torch.int64,"remote"),(32,torch.int32,"local"),(17,torch.int64,"mixed")]:
    x=rand(n,H)
    ids=torch.full((n,8),-1,dtype=dt)
    if pattern=="mixed":ids[::13,:4]=torch.randint(E,(len(x[::13]),4),generator=g,dtype=dt)
    elif pattern=="skewed":ids[::7,0]=E-1
    elif pattern=="local":ids[:]=torch.randint(E,ids.shape,generator=g,dtype=dt)
    # Compare every output and every inactive or padded route map with the ordinary implementation.
    ref=None
    for mode in ("disabled","paired","reject","missing","moved"):
        events.clear();layer.humming_prompt_gate_up=pstate(mode)
        y,mapped=run(layer,x,ids)
        if ref is None:ref=(y.clone(),mapped.clone())
        else:assert torch.equal(y,ref[0]) and torch.equal(mapped,ref[1])
        names=[e[0] for e in events]
        assert names==([] if pattern=="remote" else ["paired","w2"] if mode=="paired" else ["w13","w2"]),names
        if pattern!="remote":
            assert mapped.shape==ids.shape and torch.all(mapped[ids<0]==-1)
            for e in range(E):
                rows,slot=torch.where(ids==e)
                if not len(rows):continue
                activated=x.new_empty((len(rows),I))
                silu((ref_w13(x[rows],e)*scale[e]).bfloat16(),activated)
                expected=((activated.float()*down_diag[e].float()).bfloat16()*w2scale[e]).bfloat16()
                assert torch.equal(y[mapped[rows,slot]],expected)
                out_checks+=expected.numel()
        counts[mode]=counts.get(mode,0)+1
# One layer, one pair of packed banks, two descriptor kinds. Keep both mode tables live.
mode="paired";layer.humming_prompt_gate_up=pstate(mode);roles.clear()
x=rand(8,H);indices=torch.zeros((8,1),dtype=torch.int32)
run(layer,x,indices)
for w in (8,16,24):
    keys=torch.arange(8,dtype=torch.int32)+8
    offsets=torch.tensor([0]+[8]*E,dtype=torch.int32)
    got=gate.run_gate_up(layer,x,keys,offsets,w)
    expect=x.new_empty(x.shape);silu((ref_w13(x,0)*scale[0]).bfloat16(),expect)
    assert torch.equal(got,expect)
run(layer,x,indices)
assert roles==[("prompt",176),("decode",8),("decode",16),("decode",24),("prompt",176)]
print(json.dumps(dict(physical_gpu=False,standard_grouped_callback="humming_grouped_expert_outputs",
    geometry_cases=6,mode_cases=counts,bf16_output_components=out_checks,alias_and_postscale_preserved=True,
    real_host_sort_gather_and_multiply=True,mode_sequence=roles,delta_scope="CPU host path; projection and device epilogue emulated.",
    grouped_source_sha256=hashlib.sha256(hum.read_bytes()).hexdigest()),indent=2))
