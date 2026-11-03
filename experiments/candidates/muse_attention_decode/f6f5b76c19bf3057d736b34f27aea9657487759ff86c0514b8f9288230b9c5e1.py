# okc: problem=muse_attention_decode
"""Fuse query/key normalization, RoPE, FP8 conversion and cache scatter.

Sliding-window attention uses grouped split-KV Triton kernels, with the
partial reduction fused into output RMSNorm and gating. Full attention retains
the production FA3 path. BF16 activation and FP8 cache boundaries are preserved.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: q (Hq*64), keys (Hkv*64),
                               # values (Hkv*64), the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


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
                    mask=k_live[None, :], other=0).to(tl.bfloat16)
        v = tl.load(VC + (cache_row[:, None] * HKV + kv_head) * D + rd[None, :],
                    mask=k_live[:, None], other=0).to(tl.bfloat16)
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


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    window = cfg["window"]
    q_dtype = torch.bfloat16 if window else FP8
    q = torch.empty((tokens, hq, d), dtype=q_dtype, device=device)
    has_rope = bool(cfg["rope_theta"])
    cs = _cos_sin(cfg["rope_theta"], d, device) if has_rope else qkv
    rows = tokens * (hq + 2 * hkv)
    block_m = 4 if rows < 1024 else 8
    _prepare_qkv[(triton.cdiv(rows, block_m),)](
        qkv, q, data["positions"], cs, data["cache_loc"],
        k_cache, v_cache, data["k_scale"], data["v_scale"],
        qkv.stride(0), rows, hq, hkv, d, has_rope, scaled, block_m,
        num_warps=4, enable_fp_fusion=False,
    )

    if window:
        group = hq // hkv
        bm = max(16, triton.next_power_of_2(length * group))
        splits = 16 if batch <= 2 else 4
        bn = 64
        # Include the union of all causal windows in this verify block.
        # For L=8 this is WINDOW+7 keys, not only WINDOW.
        chunk = triton.cdiv(triton.cdiv(window + length - 1, splits), bn) * bn
        partial = torch.empty((batch, hkv, splits, bm, d), dtype=torch.float32, device=device)
        lse = torch.empty((batch, hkv, splits, bm), dtype=torch.float32, device=device)
        output = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
        _window_attention[(batch, hkv, splits)](
            q, k_cache, v_cache, data["page_table"], data["seq_lens"],
            data["k_scale"], data["v_scale"], partial, lse,
            hq, hkv, d, length, group, cfg["page_size"],
            data["page_table"].stride(0), window, splits, chunk, bm, bn,
            cfg["sm_scale"] * cfg["qk_scale_factor"], num_warps=4,
        )
        _window_combine_norm_gate[(tokens * hq,)](
            partial, lse, qkv, output, hq, hkv, d, length, group,
            splits, bm, qkv.stride(0), num_warps=4,
        )
        return output

    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)
