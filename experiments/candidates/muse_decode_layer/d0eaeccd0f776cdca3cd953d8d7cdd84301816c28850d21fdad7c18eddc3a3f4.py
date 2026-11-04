# okc: problem=muse_decode_layer
# Combined candidate: licensed compact expert alignment plus our V5 attention stage.
# Alignment base: public_alignment_v6_licensed.py, SHA-256 5eb7205b63185fb60ba2a33b53b697d6cc176cff68eb541f1f5c3ef8696e5112.
# Its executable AST matches passed check 41920b5cf4bf423f, frozen SHA-256
# 27d8c9ad081d0d11c1ede983ad3c08d1a8ffe36d4a84762814f4316bb598d2bf.
# Attention stage: window_attention_v5.py, SHA-256
# d360fe102cc3b662cbeaf5474f236bee790f2b64e6c175724c56ae2e9c5abf63.
# The combined behavior has not been GPU-validated; the base's measured speedup
# does not establish accuracy or performance after the attention change.
# MIT License
#
# Copyright (c) 2026 Rohan Arun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Expert alignment/finalizer adapted from Auto-research-agent (MIT), commit
# 018b7a102348ea3b73701e8be52d24f82d8c91ee, GPT-6.1 Sol MoE candidate.
# Original source and licenses retained under third_party/auto_research/.
# SGLang baseline portions retain their original Apache-2.0 terms.
# Integration: receives rotated global IDs; local expert bank starts at zero.
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  fused FP8 cache store            FP8 queries for FA3; BF16 queries for the grouped window path
  grouped window attention         sliding decode and small-batch verify: complete causal window,
                                   split reduction fused with BF16 output norm and sigmoid gate
  flash_attn_with_kvcache (FA3)    full attention and larger verify batches, followed by the
                                   original FusedRMSNormGate per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


@triton.jit
def _single_align(Ids, Local, Sorted, Experts, Padded,
                  T:tl.constexpr, E:tl.constexpr, START:tl.constexpr, BM:tl.constexpr):
    slot=tl.arange(0,T)
    ids=tl.load(Ids+slot)
    valid=(ids>=START)&(ids<START+E)
    rank=tl.cumsum(valid.to(tl.int32))-1
    lane=tl.arange(0,BM)
    value=tl.where(lane[None,:]==0,slot[:,None],T)
    tl.store(Sorted+rank[:,None]*BM+lane[None,:],value,valid[:,None])
    tl.store(Experts+rank,ids-START,valid)
    tl.store(Local+slot,tl.where(valid,ids-START,-1))
    tl.store(Padded,tl.sum(valid.to(tl.int32),0)*BM)

@triton.jit
def _collect(Ids, Local, Pairs, Counts, M:tl.constexpr, T:tl.constexpr, E:tl.constexpr, START:tl.constexpr, R:tl.constexpr):
    e=tl.program_id(0)
    r=tl.arange(0,R)
    ids=tl.load(Ids+r,r<M*T,other=-1)
    valid=(ids==e+START)&(r<M*T)
    rank=tl.cumsum(valid.to(tl.int32))-1
    tl.store(Pairs+e*M+rank,r,valid)
    tl.store(Counts+e,tl.sum(valid.to(tl.int32),0))
    if e==0:
        local=tl.where((ids>=START)&(ids<START+E),ids-START,-1)
        tl.store(Local+r,local,r<M*T)

@triton.jit
def _align(Pairs, Counts, Sorted, Experts, Padded, M:tl.constexpr, T:tl.constexpr, E:tl.constexpr, BM:tl.constexpr, R:tl.constexpr, RE:tl.constexpr, RB:tl.constexpr):
    e=tl.program_id(0)
    ee=tl.arange(0,RE)
    c=tl.load(Counts+ee,ee<E,other=0)
    padded=tl.cdiv(c,BM)*BM
    offset=tl.sum(tl.where(ee<e,padded,0),0)
    count=tl.load(Counts+e)
    r=tl.arange(0,R)
    pair=tl.load(Pairs+e*M+r,r<count,other=M*T)
    tl.store(Sorted+offset+r,pair,r<tl.cdiv(count,BM)*BM)
    b=tl.arange(0,RB)
    tl.store(Experts+offset//BM+b,e,b<tl.cdiv(count,BM))
    if e==0:
        tl.store(Padded,tl.sum(padded,0))

@triton.jit
def _norm_sum(Y, Ids, Gate, Norm, Out, H:tl.constexpr, T:tl.constexpr, BH:tl.constexpr):
    token=tl.program_id(0)
    h=tl.arange(0,BH)
    slot=tl.arange(0,T)
    ids=tl.load(Ids+token*T+slot)
    g=tl.load(Gate+token*T+slot)
    n=tl.load(Norm+h,h<H,other=0).to(tl.float32)
    y=tl.load(Y+(token*T+slot[:,None])*H+h[None,:],(ids[:,None]>=0)&(h[None,:]<H),other=0).to(tl.float32)
    y=(y*n[None,:]).to(tl.bfloat16).to(tl.float32)
    inv=tl.rsqrt(tl.sum(y*y,1)/H+1.e-8)
    z=tl.sum(y*(inv*g)[:,None],0)
    tl.store(Out+token*H+h,z,h<H)

def _experts(x, topk_weights, topk_ids, state):
    bank=state['bank']
    m,h=x.shape
    n=bank['size_n'];t=TOP_K;e=LOCAL_EXPERTS
    bm=8 if m<=64 else 16
    dev=x.device
    local=torch.empty((m,t),device=dev,dtype=torch.int32)
    capacity=t*bm if m==1 else triton.cdiv(m*t+e*(bm-1),bm)*bm
    sorted_ids=torch.empty((capacity,),device=dev,dtype=torch.int32)
    block_experts=torch.empty((capacity//bm,),device=dev,dtype=torch.int32)
    n_padded=torch.empty((),device=dev,dtype=torch.int32)
    if m==1:
        _single_align[(1,)](topk_ids,local,sorted_ids,block_experts,n_padded,
                           t,e,0,bm,num_warps=4)
    else:
        pairs=torch.empty((e*m,),device=dev,dtype=torch.int32)
        counts=torch.empty((e,),device=dev,dtype=torch.int32)
        _collect[(e,)](topk_ids,local,pairs,counts,m,t,e,0,triton.next_power_of_2(m*t))
        _align[(e,)](pairs,counts,sorted_ids,block_experts,n_padded,m,t,e,bm,
                     triton.next_power_of_2(triton.cdiv(m,bm)*bm),triton.next_power_of_2(e),
                     triton.next_power_of_2(triton.cdiv(m,bm)),num_warps=4)
    gate=topk_weights.to(torch.float32).contiguous()
    c1=torch.empty((m*t,2*n),device=dev,dtype=x.dtype)
    c2=torch.empty((m*t,n),device=dev,dtype=x.dtype)
    c3=torch.empty((m*t,h),device=dev,dtype=x.dtype)
    common=dict(moe_block_size=bm,mul_topk_weights=False,is_ep=True,
                b_q_type=scalar_types.float4_e2m1f,is_k_full=True,use_atomic_add=False,
                use_fp32_reduce=True,is_zp_float=False)
    ws=bank['workspace']
    c1=moe_wna16_marlin_gemm(x,c1,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,ws,
        sorted_ids,block_experts,n_padded,gate,top_k=t,size_m=m,size_n=2*n,size_k=h,**common)
    silu_and_mul(c1,c2,expert_ids=local.view(-1),expert_step=1)
    y=moe_wna16_marlin_gemm(c2,c3,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,ws,
        sorted_ids,block_experts,n_padded,gate,top_k=1,size_m=m*t,size_n=h,size_k=n,**common)
    out=torch.empty_like(x)
    _norm_sum[(m,)](y,local,gate,state['post_expert_w'],out,h,t,triton.next_power_of_2(h),num_warps=8)
    return out



@triton.jit
def _prepare_qkv(QKV, Q_OUT, POS, COS_SIN, LOC, KC, VC, KS, VS,
                 STRIDE: tl.constexpr, ROWS: tl.constexpr,
                 HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 ROPE: tl.constexpr, SCALED: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, D)
    heads: tl.constexpr = HQ + 2 * HKV
    token, head = rows // heads, rows % heads
    is_q = head < HQ
    is_k = (head >= HQ) & (head < HQ + HKV)
    is_v = head >= HQ + HKV
    mask = rows[:, None] < ROWS
    src = token[:, None] * STRIDE + head[:, None] * D + cols[None, :]
    raw = tl.load(QKV + src, mask=mask, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(raw * raw, axis=1) / D + 1.0e-5)
    norm = (raw * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        pair = tl.load(QKV + token[:, None] * STRIDE + head[:, None] * D
                       + (cols[None, :] ^ 1), mask=mask, other=0).to(tl.float32)
        pair = (pair * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POS + token, mask=rows < ROWS, other=0).to(tl.int64)
        c = tl.load(COS_SIN + pos[:, None] * D + cols[None, :] // 2,
                    mask=mask, other=1)
        s = tl.load(COS_SIN + pos[:, None] * D + D // 2 + cols[None, :] // 2,
                    mask=mask, other=0)
        norm = tl.where(cols[None, :] % 2 == 0,
                        norm * c - pair * s, norm * c + pair * s)
        norm = norm.to(tl.bfloat16).to(tl.float32)
    q_dst = (token[:, None] * HQ + head[:, None]) * D + cols[None, :]
    tl.store(Q_OUT + q_dst, norm, mask=mask & is_q[:, None])
    value = tl.where(is_v[:, None], raw, norm)
    if SCALED:
        k_scale = tl.load(KS)
        v_scale = tl.load(VS)
        scale = tl.where(is_v, v_scale, k_scale)
        value = (value / scale[:, None]).to(tl.bfloat16).to(tl.float32)
    slot = tl.load(LOC + token, mask=rows < ROWS, other=0).to(tl.int64)
    kh = tl.where(is_v, head - HQ - HKV, head - HQ)
    dst = (slot[:, None] * HKV + kh[:, None]) * D + cols[None, :]
    tl.store(KC + dst, value, mask=mask & is_k[:, None])
    tl.store(VC + dst, value, mask=mask & is_v[:, None])


# Attention-only integration of our passed standalone V3, check 05f77382a0c84254,
# SHA-256 103624d509012ab92fd2f3a2237827bc739a542a0eae5a333802d3cf3712aa60,
# into passed decoder V2, check a28df47ba0584bed,
# SHA-256 010c881e46e411735adef5aff33713d060106c9a4abef3cc5a4986043c221197.
# The two kernels below are copied verbatim. They cover every causal key in
# the window, including the WINDOW + LENGTH - 1 union for verify blocks.
# Their output already includes BF16 attention rounding, RMSNorm and gating.
@triton.jit
def _window_attention(Q, KC, VC, TABLE, SEQ, KS, VS, PART, LSE,
                      HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                      LENGTH: tl.constexpr, GROUP: tl.constexpr,
                      PAGE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
                      WINDOW: tl.constexpr, SPLITS: tl.constexpr,
                      CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      SCALE: tl.constexpr):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    qm = tl.arange(0, BM)
    rd = tl.arange(0, D)
    q_token = qm // GROUP
    q_head = kv_head * GROUP + qm % GROUP
    q = tl.load(Q + ((batch * LENGTH + q_token[:, None]) * HQ
                     + q_head[:, None]) * D + rd[None, :],
                mask=q_token[:, None] < LENGTH, other=0).to(tl.bfloat16)
    seq = tl.load(SEQ + batch)
    first = tl.maximum(0, seq - WINDOW + 1)
    query_pos = seq + q_token
    k_scale = tl.load(KS)
    v_scale = tl.load(VS)
    scale2 = SCALE * k_scale * 1.4426950408889634
    m = tl.full((BM,), -1.0e6, tl.float32)
    l = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for step in range(tl.cdiv(CHUNK, BN)):
        n = first + split * CHUNK + step * BN + tl.arange(0, BN)
        k_live = n < seq + LENGTH
        physical = tl.load(TABLE + batch * TABLE_STRIDE + n // PAGE,
                           mask=k_live, other=0).to(tl.int64)
        cache_row = physical * PAGE + n % PAGE
        k = tl.load(KC + (cache_row[None, :] * HKV + kv_head) * D + rd[:, None],
                    mask=k_live[None, :], other=0.0).to(tl.bfloat16)
        v = tl.load(VC + (cache_row[:, None] * HKV + kv_head) * D + rd[None, :],
                    mask=k_live[:, None], other=0.0).to(tl.bfloat16)
        score = tl.dot(q, k).to(tl.float32) * scale2
        live = (q_token[:, None] < LENGTH) & k_live[None, :] \
            & (n[None, :] <= query_pos[:, None]) \
            & (n[None, :] > query_pos[:, None] - WINDOW)
        score = tl.where(live, score, -1.0e6)
        new_m = tl.maximum(m, tl.max(score, axis=1))
        alpha = tl.exp2(m - new_m)
        prob = tl.where(live, tl.exp2(score - new_m[:, None]), 0.0)
        acc = acc * alpha[:, None]
        acc += tl.dot(prob.to(tl.bfloat16), v)
        l = l * alpha + tl.sum(prob, axis=1)
        m = new_m
    inv_l = 1.0 / tl.maximum(l, 1.0e-30)
    normalized = acc * (inv_l * v_scale)[:, None]
    part_base = (((batch * HKV + kv_head) * SPLITS + split) * BM + qm) * D
    tl.store(PART + part_base[:, None] + rd[None, :], normalized,
             mask=q_token[:, None] < LENGTH)
    logsum = tl.where(l > 0, m + tl.log2(l), -float("inf"))
    tl.store(LSE + ((batch * HKV + kv_head) * SPLITS + split) * BM + qm,
             logsum, mask=q_token < LENGTH)


@triton.jit
def _window_combine_norm_gate(PART, LSE, QKV, OUT,
                              HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                              LENGTH: tl.constexpr, GROUP: tl.constexpr,
                              SPLITS: tl.constexpr, BM: tl.constexpr,
                              QKV_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    head = row % HQ
    token = row // HQ
    batch = token // LENGTH
    token_j = token % LENGTH
    kv_head = head // GROUP
    qm = token_j * GROUP + head % GROUP
    splits = tl.arange(0, SPLITS)
    rd = tl.arange(0, D)
    lse = tl.load(LSE + ((batch * HKV + kv_head) * SPLITS + splits) * BM + qm)
    maximum = tl.max(lse, axis=0)
    weight = tl.exp2(lse - maximum)
    part = tl.load(PART + (((batch * HKV + kv_head) * SPLITS
                           + splits[:, None]) * BM + qm) * D + rd[None, :])
    attn = tl.sum(part * weight[:, None], axis=0) / tl.sum(weight, axis=0)
    # Production FA3 rounds the attention result to BF16 before output norm.
    attn = attn.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(attn * attn, axis=0) / D + 1.0e-5)
    gate = tl.load(QKV + token * QKV_STRIDE + (HQ + 2 * HKV + head) * D + rd).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(OUT + row * D + rd, attn * inv * sigmoid)



def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    # Fused producer from our passing standalone attention candidate. Cache
    # rows include slot zero, which is live in these standalone problems.
    gate = qkv[:, (hq + 2 * hkv) * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    window = cfg["window"]
    # General shape dispatch: grouped split-KV has clear profile headroom for
    # single-token decode and small-batch verify. Old FA3 + combine + norm is
    # at least 13.6/17.9/18.5 us for B=1/16/32 decode and 14.4 us for B=1
    # verify; the corresponding V3 attention + fused combine/norm is
    # 5.2/11.3/14.8/8.5 us in separate runs. The old norm is omitted from the
    # truncated profile, so these are hypotheses, not end-to-end measurements.
    # Large-batch verify has uncertain margin and keeps the complete FA3 path.
    use_window = bool(window) and (length == 1 or batch <= 2)
    q_dtype = torch.bfloat16 if use_window else FP8
    q = torch.empty((tokens, hq, d), dtype=q_dtype, device=qkv.device)
    rows = tokens * (hq + 2 * hkv)
    block_m = 4 if rows < 1024 else 8
    _prepare_qkv[(triton.cdiv(rows, block_m),)](
        qkv, q, data["positions"], state["cos_sin"], data["cache_loc"],
        k_cache, v_cache, data["k_scale"], data["v_scale"],
        qkv.stride(0), rows, hq, hkv, d, bool(cfg["rope_theta"]), scaled, block_m,
        num_warps=4, enable_fp_fusion=False,
    )
    if use_window:
        group = hq // hkv
        bm = max(16, triton.next_power_of_2(length * group))
        splits = 16 if batch <= 2 else 4
        bn = 64
        chunk = triton.cdiv(triton.cdiv(window + length - 1, splits), bn) * bn
        # Per-call scratch and output: no input-derived state survives a call.
        # Producer, attention, combine and O projection use the current stream.
        partial = torch.empty((batch, hkv, splits, bm, d), dtype=torch.float32, device=qkv.device)
        lse = torch.empty((batch, hkv, splits, bm), dtype=torch.float32, device=qkv.device)
        o = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
        _window_attention[(batch, hkv, splits)](
            q, k_cache, v_cache, data["page_table"], data["seq_lens"],
            data["k_scale"], data["v_scale"], partial, lse,
            hq, hkv, d, length, group, cfg["page_size"],
            data["page_table"].stride(0), window, splits, chunk, bm, bn,
            cfg["sm_scale"] * cfg["qk_scale_factor"], num_warps=4,
        )
        _window_combine_norm_gate[(tokens * hq,)](
            partial, lse, qkv, o, hq, hkv, d, length, group,
            splits, bm, qkv.stride(0), num_warps=4,
        )
    else:
        descale = {}
        if scaled:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
        out = flash_attn_with_kvcache(
            q=q.view(tokens, hq, d).to(FP8),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            scheduler_metadata=data["scheduler_metadata"],
            **descale,
        )
        if isinstance(out, tuple):
            out = out[0]
        o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
