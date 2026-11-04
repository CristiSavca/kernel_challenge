from typing import Optional

import torch
import triton
import triton.language as tl

_MIN_TOKENS_FOR_FUSED = 32


@triton.jit
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            acc += y * s

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _small_post_expert_inverse_kernel(
    x_ptr,
    w_ptr,
    ids_ptr,
    inv_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok, slot = tl.program_id(0), tl.program_id(1)
    route = tok * TOPK + slot
    inverse = 1.0
    if APPLY_NORM:
        offs = tl.arange(0, BLOCK_H)
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + route) >= 0
        # Match the Torch fallback's where(ids >= 0, x, 0). Keep the
        # zero-slot arithmetic, including its nonfinite/zero-epsilon behavior.
        f = tl.load(x_ptr + route * H + offs, mask=(offs < H) & live, other=0.0).to(tl.float32)
        if HAS_WEIGHT and WEIGHT_FIRST:
            w = tl.load(w_ptr + offs, mask=offs < H, other=0.0).to(tl.float32)
            f = f * w
        variance = tl.sum(f * f, axis=0) / H
        inverse = tl.rsqrt(variance + eps)
    tl.store(inv_ptr + route, inverse)


@triton.jit
def _small_post_expert_combine_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    ids_ptr,
    inv_ptr,
    out_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    if APPLY_NORM and HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for slot in tl.static_range(TOPK):
        route = tok * TOPK + slot
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + route) >= 0
        f = tl.load(x_ptr + route * H + offs, mask=mask & live, other=0.0).to(tl.float32)
        if APPLY_NORM:
            inverse = tl.load(inv_ptr + route)
            if HAS_WEIGHT and WEIGHT_FIRST:
                f = f * w
                f = f * inverse
            elif HAS_WEIGHT:
                f = f * inverse
                f = f * w
            else:
                f = f * inverse
        score = tl.load(s_ptr + route).to(tl.float32)
        f = f * score
        acc = acc + f
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_post_expert_norm_weighted_sum(
    expert_output: torch.Tensor,
    topk_scores: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    weight_first: bool = True,
    apply_norm: bool = True,
    slot_expert_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = expert_output.contiguous()
    tokens, topk, hidden = x.shape
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    # Parallelize the small batch over routes for normalization, then over
    # output columns for combination. Only a scalar per route crosses kernels.
    small_cuda = (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and tokens < _MIN_TOKENS_FOR_FUSED
        and 0 < topk <= 32
        and 0 < hidden <= 16384
        and topk_scores.shape == (tokens, topk)
        and topk_scores.device == x.device
        and topk_scores.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and (
            norm_weight is None
            or (
                norm_weight.shape == (hidden,)
                and norm_weight.device == x.device
                and norm_weight.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            )
        )
        and (
            slot_expert_ids is None
            or (
                slot_expert_ids.numel() == tokens * topk
                and slot_expert_ids.device == x.device
                and slot_expert_ids.dtype in (torch.int32, torch.int64)
            )
        )
    )
    if small_cuda:
        weight = norm_weight.contiguous() if norm_weight is not None else x
        scores = topk_scores.contiguous()
        ids = slot_expert_ids.contiguous() if slot_expert_ids is not None else x
        inverse = torch.empty((tokens, topk), dtype=torch.float32, device=x.device)
        block_h = triton.next_power_of_2(hidden)
        _small_post_expert_inverse_kernel[(tokens, topk)](
            x, weight, ids, inverse,
            TOPK=topk, H=hidden, eps=eps,
            APPLY_NORM=apply_norm, HAS_WEIGHT=norm_weight is not None,
            WEIGHT_FIRST=weight_first, HAS_IDS=slot_expert_ids is not None,
            BLOCK_H=block_h,
            num_warps=8 if block_h >= 4096 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _small_post_expert_combine_kernel[(tokens, triton.cdiv(hidden, 256))](
            x, weight, scores, ids, inverse, out,
            TOPK=topk, H=hidden,
            APPLY_NORM=apply_norm, HAS_WEIGHT=norm_weight is not None,
            WEIGHT_FIRST=weight_first, HAS_IDS=slot_expert_ids is not None,
            BLOCK_H=256,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return out
    if tokens < _MIN_TOKENS_FOR_FUSED:
        y = x
        if slot_expert_ids is not None:
            y = torch.where(
                (slot_expert_ids.reshape(tokens, topk) >= 0).unsqueeze(-1),
                y,
                torch.zeros((), dtype=y.dtype, device=y.device),
            )
        if apply_norm:
            f = y.float()
            if norm_weight is not None and weight_first:
                f = f * norm_weight.float()
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            elif norm_weight is not None:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
                f = f * norm_weight.float()
            else:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            y = f
        else:
            y = y.float()
        y = y * topk_scores.float().unsqueeze(-1)
        return y.sum(dim=1).to(x.dtype)

    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x,
        norm_weight if norm_weight is not None else x,
        topk_scores.contiguous(),
        out,
        slot_expert_ids if slot_expert_ids is not None else x,
        TOPK=topk,
        H=hidden,
        eps=eps,
        APPLY_NORM=apply_norm,
        HAS_WEIGHT=norm_weight is not None,
        WEIGHT_FIRST=weight_first,
        HAS_IDS=slot_expert_ids is not None,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


__all__ = ["fused_post_expert_norm_weighted_sum"]
