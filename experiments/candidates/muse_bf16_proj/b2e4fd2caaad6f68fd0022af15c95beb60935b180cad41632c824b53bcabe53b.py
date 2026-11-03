# okc: problem=muse_bf16_proj
"""Profile-guided H200 projections with stock fallbacks.

Retain bandwidth GEMVs and small-pooler GEMMs where they measured faster.
The FP32 router keeps both permitted BF16 residual components, accumulated
in FP32, with enough K partitions to occupy the GPU for larger batches.
"""
import torch
import triton as tr
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


@tr.jit
def _gemv(X, W, P, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    w = tl.load(W + n[:, None] * K + k[None, :],
                (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    v = tl.sum(w * x[None, :], 1)
    tl.store(P + tl.program_id(1) * N + n, v, n < N)


@tr.jit
def _split_mm(X, W, LO, P, M: tl.constexpr, N: tl.constexpr,
              K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, SPLIT: tl.constexpr, ROUTER: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.program_id(2) * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(tr.cdiv(K, BK * SPLIT)):
        kk = rk + i * BK * SPLIT
        a = tl.load(X + rm[:, None] * K + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(W + kk[:, None] * N + rn[None, :],
                    (kk[:, None] < K) & (rn[None, :] < N), 0)
        acc += tl.dot(a, b)
        if ROUTER:
            lo = tl.load(LO + kk[:, None] * N + rn[None, :],
                         (kk[:, None] < K) & (rn[None, :] < N), 0)
            acc += tl.dot(a, lo)
    tl.store(P + tl.program_id(2) * M * N + rm[:, None] * N + rn[None, :],
             acc, (rm[:, None] < M) & (rn[None, :] < N))


@tr.jit
def _reduce(P, Y, MN: tl.constexpr, S: tl.constexpr, BS: tl.constexpr):
    idx = tl.program_id(0) * BS + tl.arange(0, BS)
    val = tl.full((BS,), 0, tl.float32)
    for s in range(S):
        val += tl.load(P + s * MN + idx, idx < MN, 0)
    tl.store(Y + idx, val, idx < MN)


def prepare(weights):
    state = dict(weights)
    state["pooler_t"] = weights["pooler"].t().contiguous()
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    router = layer == "router"
    gemv = m == 1 and layer != "o"
    split_mm = router or (layer == "pooler" and 1 < m <= 16)
    if not gemv and not split_mm:
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    y = torch.empty((m, n), device=x.device,
                    dtype=torch.float32 if router else torch.bfloat16)
    if gemv:
        split = tr.cdiv(k, 1024)
        p = torch.empty((split, n), device=x.device, dtype=torch.float32)
        _gemv[(tr.cdiv(n, 4), split)](x, state[layer], p, n, k, 4, 1024, num_warps=4)
        _reduce[(tr.cdiv(n, 256),)](p, y, n, split, 256)
        return y
    split = 8
    p = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    if router:
        wt, lo = state["router_hi_t"], state["router_lo_t"]
    else:
        wt = lo = state["pooler_t"]
    bm = 16 if m <= 16 else 32
    _split_mm[(tr.cdiv(m, bm), tr.cdiv(n, 64), split)](
        x, wt, lo, p, m, n, k, bm, 64, 64, split, router,
        num_warps=4, num_stages=3)
    _reduce[(tr.cdiv(m * n, 256),)](p, y, m * n, split, 256)
    return y
