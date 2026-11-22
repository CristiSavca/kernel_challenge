"""Run the pinned first-layer grouped convolution and verify its guaranteed row classes."""
import ast
import itertools
import json
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

SERVING = Path(__file__).resolve().parents[2]
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def pinned(name):
    return subprocess.check_output(["git", "-C", str(SERVING/"sglang"), "show", f"{BASE}:python/sglang/{name}"]).decode()


def extract(source, function, names):
    fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == function)
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module([fn], type_ignores=[]))
    exec(compile(module, function, "exec"), names)
    return names[function]


def project(x, weight):
    return F.linear(x, weight)


def rotate(x):
    # A small NeoX RoPE fixture with the checkpoint's theta.
    half = x.shape[-1] // 2
    freq = 500000. ** (-torch.arange(half) / half)
    phase = torch.arange(x.shape[1])[:, None] * freq[None, :]
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first*phase.cos()-second*phase.sin(), second*phase.cos()+first*phase.sin()), dim=-1).to(x.dtype)


def trial(conv, batch, width, seed):
    rng = torch.Generator().manual_seed(seed)
    def sample(*shape):
        return torch.randn(shape, generator=rng).bfloat16()
    hidden, group_size, taps = 32, 4, 2
    groups = hidden // group_size
    mask = sample(hidden)
    anchors = sample(batch, hidden)
    unique = torch.cat((anchors, mask[None]), dim=0)
    classes = torch.full((batch,width),batch,dtype=torch.long)
    classes[:,0] = torch.arange(batch)
    dense = unique[classes].reshape(-1,hidden)
    normalized = F.rms_norm(dense, (hidden,), eps=1e-6)
    norm_unique = F.rms_norm(unique, (hidden,), eps=1e-6)
    weight = sample(2*taps*groups,hidden)
    coeff = project(normalized,weight)
    compact_coeff = project(norm_unique,weight)[classes.flatten()]
    assert torch.equal(coeff,compact_coeff)
    kernel = sample(taps,hidden)
    def pre(rows):
        return conv(normalized,rows.reshape(-1,2,taps,groups)[:,0],kernel,width,groups,group_size,taps)
    h = pre(coeff).view(batch,width,hidden)
    assert torch.equal(h,pre(compact_coeff).view_as(h))
    assert torch.equal(h[:,2:],h[:1,2:3].expand(batch,width-2,hidden))
    assert torch.all((h[:,1]!=h[:,2]).any(-1))
    representative = torch.cat((h[:,0],h[:,1],h[:1,2]),dim=0)
    qclass = torch.full_like(classes,2*batch)
    qclass[:,0],qclass[:,1] = torch.arange(batch),batch+torch.arange(batch)
    qkv_weight = sample(3*hidden,hidden)
    qkv = project(h,qkv_weight)
    assert torch.equal(qkv,project(representative,qkv_weight)[qclass])
    q,k,v = qkv.chunk(3,dim=-1)
    q = rotate(F.rms_norm(q,(hidden,),eps=1e-6))
    k = rotate(F.rms_norm(k,(hidden,),eps=1e-6))
    prefix_k, prefix_v = sample(batch,9,hidden),sample(batch,9,hidden)
    attended = F.scaled_dot_product_attention(
        q[:,None], torch.cat((prefix_k,k),dim=1)[:,None], torch.cat((prefix_v,v),dim=1)[:,None], is_causal=False
    )[:,0]
    assert torch.all((q[:,2]!=q[:,3]).any(-1))
    assert torch.all((attended[:,2]!=attended[:,3]).any(-1))
    return batch


def bank_traffic():
    # Official reference geometry, TP8, one BF16 read per useful bank element.
    hidden,intermediate,hq,hkv,dim,tp = 8192,28672,64,8,128,8
    n_qkv = (hq+2*hkv)*dim//tp
    filters = 2*2*(hidden//16)*hidden*2
    qkv = n_qkv*hidden*2
    per_layer = dict(qkv=qkv, attention_out=hq*dim//tp*hidden*2,
                     gate_up=2*intermediate//tp*hidden*2, down=intermediate//tp*hidden*2,
                     attention_filter=filters,mlp_filter=filters)
    flop_saving = (8-2)*filters + (8-3)*qkv
    assert sum(per_layer.values())*5==(1340<<20)
    return {"tp_size":tp,"per_layer_bf16_weight_mib":{k:v//(1<<20) for k,v in per_layer.items()},
            "five_layer_weight_mib":5*sum(per_layer.values())//(1<<20),
            "two_affected_banks_mib":(filters+qkv)//(1<<20),"eliminated_weight_bytes":0,
            "one_request_useful_flops_removed":flop_saving,
            "one_request_fraction_of_five_layer_linear_flops_removed":flop_saving/(8*5*sum(per_layer.values())),
            "one_rank_single_pass_affected_weight_bytes_over_peak_HBM_seconds":(filters+qkv)/4.8e12}


def main():
    torch.set_num_threads(1)
    model = pinned("srt/models/dflash.py")
    conv = extract(model,"_grouped_conv",{"F":F,"torch":torch})
    stock = extract(pinned("kernels/ops/gemm/hopper_bf16_gemv.py"),"use_hopper_bf16_gemv",
                    {"_is_sm12x":lambda:False,"_MAX_K":65536,"_MAX_N":140000})
    for m in (7,8,16,32,128,256):
        for n,k in ((1280,8192),(8192,1024),(7168,8192),(8192,3584),(2048,8192)):
            assert not stock(m,n,k)
    cases = list(itertools.product((1,3,7),(4,8,12),range(8)))
    requests = sum(trial(conv,*case) for case in cases)
    run = json.loads((SERVING/"runs/abd217baf0ed49fd.json").read_text())["result"]
    one = next(row for row in run["rungs"] if row["concurrency"]==1)
    return {"base":BASE,"torch_cpu":torch.__version__,"synthetic_bf16_batches":len(cases),
            "requests":requests,"native_hopper_gemv_shapes_rejected":30,
            "pre_rope_qkv_compressed_matches_full":True,"post_rope_and_attention_counterexamples":requests,
            "v64_rung_one_user":{k:one[k] for k in ("acceptance","tok_s_per_user_p50","tok_s_per_user_e2e","reps")},
            **bank_traffic()}


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
