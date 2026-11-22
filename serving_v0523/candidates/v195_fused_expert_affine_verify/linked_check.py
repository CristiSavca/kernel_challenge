"""Pass BF16 bytes from the integrated attention interpreter through the routed decoder layer on CPU."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as S

import numpy as np

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
arrays=json.loads(Path(sys.argv[2]).read_text())
raw=arrays if isinstance(arrays,list) else next(v for v in arrays.values() if isinstance(v,list) and np.asarray(v).size==256)
bits=np.asarray(raw,dtype="<u2").reshape(8,32)
assert bits.any()
vals=(bits.astype("u4")<<16).view("f4")
assert np.isfinite(vals).all()
old=here.parent/"v66_humming_grouped_decode/cpu_check.py"
sys.argv=[str(old),str(stage)]
oracle=old.read_text().split("routing_cases = kernel_cases")[0]
oracle=oracle.replace("    def numel(self):","    def data_ptr(self):\n        return self.ctypes.data\n\n    def numel(self):")
env={"__file__":str(old)}
exec(compile(oracle,str(old),"exec"),env)
scope=env["scope"];tensor=env["tensor"];bf16=env["bf16"];sha=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
fixture=here.parent/"v191_decode_gate_up_epilogue/dispatch_check.py"
exec(compile(fixture.read_text().split('fixture=prev.read_text()')[0],str(fixture),"exec"),{"__file__":str(fixture)})
src=stage/"python/sglang/srt/models/muse_spark.py"
env["extract"](src,("forward",),scope,klass="MuseSpark1xDecoderLayer")
forward=scope["forward"]
n=32
rng=np.random.default_rng(195)
rids=tensor(np.array([[int((i+j)%5) if j%3 else -1 for j in range(8)] for i in range(8)],dtype="i4"))
scores=tensor(rng.random((8,8),dtype="f4"));norm=tensor(bf16(rng.uniform(.5,1.5,n)))
t='{"use_stream_k":false,"use_tma_c":false}'
layer=S(global_num_experts=40,num_local_experts=5,_expert_map=None,intermediate_size_per_partition=n,
    humming_decode_tunings={key:[(128,t)] for key in ("w13","w2")},w13=tensor(bf16(rng.normal(0,.4,(5,2*n,n)))),
    w2=tensor(bf16(rng.normal(0,.4,(5,n,n)))),w13_global_scale=tensor(bf16(rng.random(5)+.5)),
    w2_global_scale=tensor(bf16(rng.random(5)+.5)),launches=[])
layer.w13_weight=layer.w13;layer.w13_weight_scale=layer.w13_global_scale
bindings=tuple((x.data_ptr(),x.shape) for x in (layer.w13_weight,layer.w13_weight_scale))
batch=object();log=[];x=tensor(vals);projection_returns=[]
class Gate:
    def run(self,in_,act,keys,offsets,scale,maps,handle,shared,width,sms):
        assert not np.shares_memory(in_,act)
        buf=in_.new_empty((len(in_),2*n))
        env["Humming"].forward_layer(layer,inputs=in_,outputs=buf,expert_layout=offsets,tuning_config=t,
            compute_config=scope["_GROUPED_COMPUTE"],sublayer_name="w13")
        env["silu_scaled"](buf,act,keys,offsets,scale,64);log.append("paired gate/up")
        return True
state=(Gate(),None,bindings,{t:(None,1,82944,8)},132)
def attention(positions,attn_input,token_batch):
    assert token_batch is batch;log.append("replayed attention output");return x
def boundary(original,v,**options):
    assert v is x;return v,original,None
def mlp(v,router,token_batch):
    assert v is x and token_batch is batch
    down,row_map,ids=scope["humming_grouped_decode_outputs"](layer,v,rids)
    log.append("routed finalization")
    return scope["fused_post_expert_norm_weighted_sum"](down,scores,norm,weight_first=True,
        slot_expert_ids=ids,slot_row_indices=row_map,out=v.new_empty(v.shape),expert_output_scales=layer.w2_global_scale)
module=S(self_attn=attention,_router_dtype=None,operator_dtype="bf16",_fused_attn_boundary=boundary,
    is_moe_layer=True,mlp=mlp,_fused_ffn_boundary=None,post_feedforward_layernorm=lambda a:a,
    post_feedforward_residual_gate=lambda a,b:b)
for mode in (None,state):
    layer.humming_decode_gate_up=mode
    out,_=forward(module,None,x,x,batch)
    projection_returns.append(np.copy(out))
assert log==["replayed attention output","routed finalization","replayed attention output","paired gate/up","routed finalization"]
assert np.array_equal(*projection_returns) and np.isfinite(projection_returns[0]).all() and projection_returns[0].any()
print(json.dumps(dict(attention_bf16_sha256=sha(bits),linked_rows=8,linked_channels=32,
    accepted_routed_rows=int((rids>=0).sum()),source_decoder_forward="MuseSpark1xDecoderLayer.forward",forward_calls=2,
    ordinary_and_paired_outputs_identical=True,output_sha256=sha(projection_returns[0].astype("<f4")),
    calls=log,scope="Exact attention output slice; native layer control and routed planner/finalizer; CPU GEMM/SiLU boundary and identity post-attention norm. No H200 model run."),indent=2))
