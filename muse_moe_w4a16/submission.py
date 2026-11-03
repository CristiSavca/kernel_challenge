# okc: problem=muse_moe_w4a16
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


@triton.jit
def _norm_weighted_sum(Y, W, Scores, Ids, Out, H: tl.constexpr, TOPK: tl.constexpr,
                       BLOCK_H: tl.constexpr):
    """Fuse the small-batch eager fallback as well as larger token batches."""
    token = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    weight = tl.load(W + h, h < H, other=0).to(tl.float32)
    result = tl.full((BLOCK_H,), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        route = token * TOPK + slot
        expert = tl.load(Ids + route)
        if expert >= 0:
            y = tl.load(Y + route * H + h, h < H, other=0).to(tl.float32)
            y = y * weight
            rstd = tl.rsqrt(tl.sum(y * y, 0) / H + 1.0e-8)
            score = tl.load(Scores + route).to(tl.float32)
            result += (y * rstd) * score
    tl.store(Out + token * H + h, result, h < H)


def _finalize(y, scores, norm_weight, local_ids):
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    _norm_weighted_sum[(tokens,)](
        y, norm_weight, scores, local_ids, out,
        H=hidden, TOPK=topk, BLOCK_H=triton.next_power_of_2(hidden),
        num_warps=8, num_stages=1,
    )
    return out


@triton.jit
def _unpack_fp4(code):
    """Decode the exact E2M1 grid, including signed zero, in FP32."""
    magnitude = code & 7
    bits = (((magnitude >> 1) + 126) << 23) | ((magnitude & 1) << 22)
    value = bits.to(tl.float32, bitcast=True)
    value = tl.where(magnitude < 2, magnitude.to(tl.float32) * 0.5, value)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _packed_gate_up(X, Ids, W, Scales, Global, Inter,
                    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    """Read packed weights only for a live route; fuse gate/up GEMVs and SwiGLU."""
    route = tl.program_id(1)
    expert = tl.load(Ids + route)
    if expert >= 0:
        token = route // TOPK
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        scale_global = tl.load(Global + expert).to(tl.float32)
        gate_acc = tl.zeros((BN, BK), tl.float32)
        up_acc = tl.zeros((BN, BK), tl.float32)
        for base in range(tl.cdiv(H, BK)):
            cols = base * BK + k
            x = tl.load(X + token * H + cols, cols < H, other=0).to(tl.float32)
            mask = (rows[:, None] < I) & (cols[None, :] < H)
            gate_bytes = tl.load(W + expert * (2 * I * (H // 2))
                                 + rows[:, None] * (H // 2) + cols[None, :] // 2,
                                 mask, other=0).to(tl.int32)
            up_bytes = tl.load(W + expert * (2 * I * (H // 2))
                               + (rows[:, None] + I) * (H // 2) + cols[None, :] // 2,
                               mask, other=0).to(tl.int32)
            shift = (cols[None, :] & 1) * 4
            gate_value = _unpack_fp4((gate_bytes >> shift) & 15)
            up_value = _unpack_fp4((up_bytes >> shift) & 15)
            gate_scale = tl.load(Scales + expert * (2 * I * (H // 16))
                                 + rows[:, None] * (H // 16) + cols[None, :] // 16,
                                 mask, other=0.0).to(tl.float32) * scale_global
            up_scale = tl.load(Scales + expert * (2 * I * (H // 16))
                               + (rows[:, None] + I) * (H // 16) + cols[None, :] // 16,
                               mask, other=0.0).to(tl.float32) * scale_global
            # Match the BF16 weight/activation GEMMs at the production boundaries.
            gate_w = (gate_value * gate_scale).to(tl.bfloat16).to(tl.float32)
            up_w = (up_value * up_scale).to(tl.bfloat16).to(tl.float32)
            gate_acc += gate_w * x[None, :]
            up_acc += up_w * x[None, :]
        gate = tl.sum(gate_acc, 1).to(tl.bfloat16).to(tl.float32)
        up = tl.sum(up_acc, 1).to(tl.bfloat16).to(tl.float32)
        value = (gate * tl.sigmoid(gate)) * up
        tl.store(Inter + route * I + rows, value, rows < I)


@triton.jit
def _packed_down(Inter, Ids, W, Scales, Global, Y,
                 H: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    route = tl.program_id(1)
    expert = tl.load(Ids + route)
    if expert >= 0:
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        scale_global = tl.load(Global + expert).to(tl.float32)
        acc = tl.zeros((BN, BK), tl.float32)
        for base in range(tl.cdiv(I, BK)):
            cols = base * BK + k
            x = tl.load(Inter + route * I + cols, cols < I, other=0).to(tl.float32)
            mask = (rows[:, None] < H) & (cols[None, :] < I)
            packed = tl.load(W + expert * (H * (I // 2))
                             + rows[:, None] * (I // 2) + cols[None, :] // 2,
                             mask, other=0).to(tl.int32)
            code = (packed >> ((cols[None, :] & 1) * 4)) & 15
            scale = tl.load(Scales + expert * (H * (I // 16))
                            + rows[:, None] * (I // 16) + cols[None, :] // 16,
                            mask, other=0.0).to(tl.float32) * scale_global
            weight = (_unpack_fp4(code) * scale).to(tl.bfloat16).to(tl.float32)
            acc += weight * x[None, :]
        value = tl.sum(acc, 1).to(tl.bfloat16)
        tl.store(Y + route * H + rows, value, rows < H)


def _direct_small(x, local_ids, scores, state, size_n):
    tokens, size_k = x.shape
    routes = tokens * TOP_K
    inter = torch.empty((routes, size_n), device=x.device, dtype=x.dtype)
    y = torch.empty((tokens, TOP_K, size_k), device=x.device, dtype=x.dtype)
    raw = state["raw"]
    _packed_gate_up[(triton.cdiv(size_n, 8), routes)](
        x, local_ids, raw["w13"], raw["w13_sf"], raw["w13_gs"], inter,
        H=size_k, I=size_n, TOPK=TOP_K, BN=8, BK=256,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    _packed_down[(triton.cdiv(size_k, 8), routes)](
        inter, local_ids, raw["w2"], raw["w2_sf"], raw["w2_gs"], y,
        H=size_k, I=size_n, BN=8, BK=256,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return _finalize(y, scores, state["norm_w"], local_ids)


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        # Lossless local packed copies avoid retaining the full 256-expert storage.
        # Dequantization happens inside the timed GEMV, never in prepare().
        "raw": {name: weights[name][:32].clone() for name in
                ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")},
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    if num_tokens <= 8:
        return _direct_small(x, local_ids, topk_weights, state, size_n)

    # Only one eighth of all routes are local under EP8. Size a tile
    # for the expected routes per local expert, rather than counting
    # non-local routes that this GPU will never execute. Skewed experts
    # still receive as many complete tiles as their actual counts need.
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / N_EXPERTS / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            # local_ids already contains only -1 and [0, count).
            # The wrapper adds its own bucket for the -1 sentinel.
            local_ids, block_size_m, bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _finalize(y, topk_weights, state["norm_w"], local_ids)
