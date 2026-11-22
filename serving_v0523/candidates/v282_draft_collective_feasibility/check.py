"""Test a block-owned draft decoder against the assembled V279 equations and native reduction dispatch."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch
import torch.nn.functional as F

here=Path(__file__).resolve().parent
serving=here.parents[1]
repo=serving/"sglang"
candidate=here.parent/"v279_multicast_expert_serving"
baseline=json.loads((candidate/"validation.json").read_text())
base=baseline["base"]
paths=["models/dflash.py","models/muse_spark.py","distributed/parallel_state.py","layers/linear.py","speculative/dflash_worker_v2.py"]
with tempfile.TemporaryDirectory() as scratch:
    obj=Path(scratch)/"obj"; obj.mkdir()
    alt=subprocess.check_output(["git","-C",repo,"rev-parse","--absolute-git-dir"],text=True).strip()+"/objects"
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(scratch)/"index"),GIT_OBJECT_DIRECTORY=str(obj),GIT_ALTERNATE_OBJECT_DIRECTORIES=alt)
    def git(*args,data=None):
        return subprocess.check_output(["git","-C",repo,*args],input=data,env=env)
    git("read-tree",base)
    git("apply","--cached","--whitespace=error-all","-",data=(candidate/"patch.diff").read_bytes())
    tree=git("write-tree").decode().strip()
    assert tree==baseline["tree"]
    source={p:git("show",tree+":python/sglang/srt/"+p).decode() for p in paths}

def method(path,cls,name):
    tree=ast.parse(source[path]); body=tree.body if cls is None else next(x.body for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    return next(x for x in body if isinstance(x,ast.FunctionDef) and x.name==name)

def ordered(src,*need):
    cursor=0
    for item in need:
        cursor=src.index(item,cursor)+len(item)

ordered(ast.unparse(method("models/dflash.py","DFlashDecoderLayer","forward")),"self.input_layernorm","self.attention_conv.prepare",
        "self.self_attn(","self.attention_conv.finish","self.post_attention_layernorm","self.mlp_conv.prepare","self.mlp(","self.mlp_conv.finish")
ordered(ast.unparse(method("models/muse_spark.py","MuseSpark1xDecoderLayer","forward")),"self.self_attn(","self._fused_attn_boundary(","self.mlp(","self._fused_ffn_boundary(")
comm=source["distributed/parallel_state.py"]
assert "self.ca_comm.should_custom_ar" in comm and "pynccl_comm.reduce_scatter(output, input)" in comm and "self.all_reduce(input.clone())" in comm
draft=source["models/dflash.py"]
assert "self.o_proj = RowParallelLinear(" in draft and "self.down_proj = RowParallelLinear(" in draft and "self.kernel_projection = nn.Linear(" in draft
conv=method("models/dflash.py",None,"_grouped_conv");conv.decorator_list=[]
scope=dict(torch=torch,F=F)
exec(compile(ast.Module([conv],[]),"pinned_convolution","exec"),scope)
convolve=scope["_grouped_conv"]
rng=torch.Generator().manual_seed(331)
torch.set_num_threads(1)
def bf(*shape):
    return torch.randn(*shape,generator=rng).bfloat16()
def norm(x):
    return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
def add(a,b):
    summed=(a.float()+b.float()).bfloat16()
    return norm(summed),summed
def prepare(x,w,b):
    coeff=F.linear(x,w).view(-1,2,2,8)
    return convolve(x,coeff[:,0],b[0],8,8,4,2),coeff[:,1]
def reduce(xs):
    return torch.stack(xs).float().sum(0).bfloat16()
def equal(a,b):
    assert torch.equal(a.view(torch.int16),b.view(torch.int16))
    return a.numel()

dense_receipt=here.parent/"v262_decode_phase_budget/check.json"
post_overlap=[]
for bench in json.loads(dense_receipt.read_bytes())["cases"]:
    post_us=10*bench["native_dense_raw_us"]["pooler"]/4
    period=bench["estimated_period_ms"]
    post_overlap.append(dict(users=bench["users"],raw_native_H200_pooler_median_us=bench["native_dense_raw_us"]["pooler"],
        eligible_post_filter_weight_MiB=160,quarter_size_times_ten_proxy_us=post_us,illustrative_full_overlap_multiplier=period/(period-post_us/1000)))
geometry=[]
cases=[]
tp=8
for B in (1,4,16,32):
    M=B*8
    n=(B+tp-1)//tp
    bounds=[(min(B,i*n)*8,min(B,(i+1)*n)*8) for i in range(tp)]
    H=32
    full=bf(M,H); residual=full.clone(); folded=norm(full)
    owned=[(folded[lo:hi].clone(),residual[lo:hi].clone()) for lo,hi in bounds]
    checked=0
    for layer in range(10):
        weight=bf(32,H); coeff_base=bf(2,2,H)
        native,k=prepare(folded,weight,coeff_base)
        local=[]
        for (lo,hi),(rows,r) in zip(bounds,owned):
            if hi==lo:continue
            y,kl=prepare(rows,weight,coeff_base)
            checked+=equal(y,native[lo:hi])+equal(kl,k[lo:hi])
            local.append(y)
        checked+=equal(torch.cat(local),native)
        # Synthetic rank projections reproduce the native BF16 rank-partial interface.
        partial=[F.linear(native,bf(H,H)) for _ in range(tp)]
        combined=reduce(partial)
        native=convolve(combined,k,coeff_base[1],8,8,4,2)
        folded,residual=add(native,residual)
        new=[]
        for (lo,hi),(rows,r) in zip(bounds,owned):
            if hi==lo:
                new.append((rows,r));continue
            # Evenly block-padded transport logically receives this exact row interval.
            result=reduce([p[lo:hi] for p in partial])
            _,kl=prepare(rows,weight,coeff_base)
            z=convolve(result,kl,coeff_base[1],8,8,4,2)
            x,r=add(z,r)
            checked+=equal(x,folded[lo:hi])+equal(r,residual[lo:hi])
            new.append((x,r))
        owned=new
    cases.append(dict(users=B,live_owners=sum(a<b for a,b in bounds),BF16_words_compared=checked,ten_sublayers=True))
    rows=n*8*tp
    payload=lambda r:r*8192*2*(tp-1)/tp/2**20
    perfilter=2*M*8192*2048
    geometry.append(dict(users=B,rows=M,padded_collective_rows=rows,baseline_ring_sent_MiB_10_allreduces=20*payload(M),
        owner_ring_sent_MiB_10_RS_11_AG=21*payload(rows),nine_owned_ring_sent_MiB_10_RS_10_AG=20*payload(rows),
        V123_extra_ring_sent_MiB_ten_coeff_gathers=10*payload(M)/4,
        V279_filter_work_GFLOP_per_rank=10*perfilter/1e9,wholeblock_owner_max_rank_filter_work_GFLOP=10*2*n*8*8192*2048/1e9,
        V123_filter_work_GFLOP_per_rank=10*perfilter/8/1e9,critical_fullbank_MiB=320,V123_fullbank_MiB=40))

x=bf(8,2048)
fp=x.float().sum(0).bfloat16()
while x.shape[0]>1:x=(x[::2]+x[1::2]).bfloat16()
mismatch=(fp.view(torch.int16)!=x[0].view(torch.int16)).count_nonzero().item()
assert mismatch>0
print(json.dumps(dict(assembled_tree=tree,source_sha256={p:hashlib.sha256(s.encode()).hexdigest() for p,s in source.items()},
    post_filter_overlap=post_overlap,dense_benchmark_receipt_sha256=hashlib.sha256(dense_receipt.read_bytes()).hexdigest(),torch_version=str(torch.__version__),native_conv_source=ast.unparse(conv),cases=cases,geometry=geometry,
    control_BF16_vs_FP32_tree_mismatches=mismatch,control_values=2048,
    scope="CPU BF16 original convolution. Matches after identical rank-sum ordering. Synthetic rank projections, simulated collective transport; hardware reduction order, traffic and time unmeasured."),indent=2))
