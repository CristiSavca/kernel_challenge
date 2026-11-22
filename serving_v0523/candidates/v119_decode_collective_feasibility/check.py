"""Pinned decode transport arithmetic and Muse Hopper norm order, CPU only."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
TP, HIDDEN, EXPERT_WIDTH, EXPERTS, K, LAYERS = 8, 8192, 4096, 256, 8, 62
MIB = 2**20


def pinned(path):
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "sglang"), "show", f"{PIN}:python/sglang/{path}"]
    ).decode()


def locate_receipt_field(node, name):
    if isinstance(node, dict):
        if name in node:
            return node[name]
        for child in node.values():
            found = locate_receipt_field(child, name)
            if found is not None:
                return found
    return None


def check_sources():
    sources = {
        "kernels/jit/utils/arch.py": ["get_jit_cuda_arch().major >= 9"],
        "kernels/ops/communication/all_reduce.py": ["world_size, is_arch_support_pdl()"],
        "kernels/ops/layernorm/muse_spark_norm_residual_jit.py": [
            "is_arch_support_pdl()", "get_device_capability(branch.device)[0] >= 10",
        ],
        "kernels/jit/csrc/muse_spark/norm_residual.cuh": [
            "normed_b = round_bf16(normed_b)", "scaled_branch_sum, 2.0f * cross_sum",
            "normed = round_bf16(normed)", "updated.x * pre_rstd",
            "PDLWaitPrimary<kUsePDL>()", "PDLTriggerSecondary<kUsePDL>()",
        ],
        "kernels/jit/csrc/distributed/custom_all_reduce.cuh": [
            "all_reduce_1shot_push_kernel", "PDLTriggerSecondary<kUsePDL>()",
        ],
        "srt/distributed/device_communicators/configs/custom_all_reduce_v2.py": [
            "num_push_blocks=num_sm", "graph=(128.0 * KB, 128.0 * KB, 32.00 * MB, Range(512 * KB",
        ],
        "srt/models/muse_spark.py": ["ffn_input, stream, router_input = self._fused_attn_boundary("],
    }
    digests = {}
    for path, terms in sources.items():
        source = pinned(path)
        assert all(term in source for term in terms), path
        digests[path] = sha256(source.encode()).hexdigest()
    v64 = next((ROOT / "candidates").glob("v64*/patch.diff")).read_bytes()
    assert sha256(v64).hexdigest() == "f6aaf93ce06ccfce9461e06bc3f71dea05dd901c01497c8be31bd2c515da689a"
    assert b'os.environ.get("NCCL_NVLS_ENABLE", "1") != "0"' in v64
    assert b"full = group.all_reduce(full)" in v64
    assert b"return full.narrow(0, group.rank_in_group * rows, rows)" in v64
    v83 = (ROOT / "candidates/v83_sharded_router_pooler/patch.diff").read_bytes()
    for token in [b"gather_pooler_and_router", b"self.weight[experts]", b"hi[:, experts]", b"gemm_local("]:
        assert token in v83
    digests["V83"] = sha256(v83).hexdigest()
    return digests


def fp32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return struct.unpack("f", struct.pack("I", rounded))[0]


def cluster_counterexample():
    # Every thread sees four (1,2) pairs per vector. CTA and warp tree reductions
    # sum identical lane partials; multiplication by a power of two is exact.
    # Eight cluster ranks each publish an equal partial. Preserve sequential
    # cluster addition as distinct rounded operations.
    def row_sum(pairs):
        lane = 0
        for _ in range(4):
            for left, right in pairs:
                lane = fp32(lane + left * right)
        cta = fp32(lane * 128)
        out = 0
        for _ in range(8):
            out = fp32(out + cta)
        return out

    branch = [1., 2.]
    post = fp32(1 / math.sqrt(fp32(row_sum(zip_items(branch, branch)) / HIDDEN + 1e-8)))
    alpha, beta = fp32(math.sqrt(.75)), .5  # checkpoint-independent legal gate w=0
    normalized = [bf16(fp32(v * post)) for v in branch]
    stream = [fp32((.001 - beta * v) / alpha) for v in normalized]
    residual = [fp32(alpha * v) for v in stream]
    updated = [fp32(r + beta * x) for r, x in zip(residual, normalized)]
    # Materialized Hopper one-CTA reduction. Every channel of this row has
    # exactly the same updated value, so changes in summation order are minor.
    assert updated[0] == updated[1]
    exact_mean = fp32(updated[0] ** 2)
    scaled = [fp32(beta * v) for v in branch]
    rr = row_sum(zip_items(residual, residual))
    rc = row_sum(zip_items(residual, scaled))
    cc = row_sum(zip_items(scaled, scaled))
    expansion = fp32(post * fp32(post * cc + 2 * rc) + rr)
    expanded_mean = max(expansion, 0.) / HIDDEN
    def normalized_output(mean):
        return [bf16(z * fp32(1 / math.sqrt(fp32(mean + 1.e-5)))) for z in updated]
    exact, expanded = normalized_output(exact_mean), normalized_output(expanded_mean)
    assert exact != expanded
    return {"repeated_branch_pair": branch, "repeated_fp32_stream_pair": stream,
            "rounded_branch_pair": normalized, "fp32_updated_stream": updated,
            "materialized_mean_square": exact_mean, "expanded_mean_square": expanded_mean,
            "materialized_bf16_output": exact, "cluster_identity_bf16_output": expanded}


def zip_items(a, b):
    return list(zip(a, b))


@dataclass(frozen=True)
class Rung:
    users: int
    acceptance: float
    rate: float

    def estimate(self):
        m = 8 * self.users
        send = (TP - 1) / TP
        hidden = m * HIDDEN * 2 * send
        pool = m * EXPERT_WIDTH * 2 * send
        logits = m * EXPERTS * 4 * send
        routes = m * K * (4 + 4) * send
        # Per-rank sends for an ideal logical ring; physical NVLink multicast
        # bytes and per-collective latencies cannot be recovered from this.
        shard_weights = 2 * hidden + 2 * pool + pool + logits + hidden
        shard_rows = hidden + pool + routes + pool + hidden
        inherited_rows = 2 * hidden + pool + routes + 2 * pool + hidden
        dense_bytes = (2 * (HIDDEN * EXPERT_WIDTH * 2) + HIDDEN * EXPERTS * 4) * send
        delta = shard_weights - shard_rows
        period_ms = 1000 * self.acceptance / self.rate
        assert math.isclose(shard_weights - inherited_rows, logits - routes)
        return {"verify_rows": m, "v83_model_ring_send_MiB": shard_weights/MIB,
                "token_owner_ideal_ring_send_MiB": shard_rows/MIB,
                "token_owner_current_v64_ring_send_MiB": inherited_rows/MIB,
                "ideal_ring_send_saving_MiB": delta/MIB,
                "extra_distinct_dense_weight_MiB_if_cold": dense_bytes/MIB,
                "break_even_distinct_weight_bytes_per_ideal_send_byte_saved": dense_bytes/delta,
                "eliminable_O_intermediate_write_plus_read_MiB_62_layers": 2*m*HIDDEN*2*LAYERS/MIB,
                "illustrative_step_ms": period_ms,
                "saved_us_per_layer_required_for_2x_current_decode": period_ms*1000*.5/LAYERS,
                "saved_us_per_layer_required_for_3x_current_decode": period_ms*1000*(2/3)/LAYERS}


if __name__ == "__main__":
    receipt = json.loads((ROOT / "runs/abd217baf0ed49fd.json").read_text())
    rows = locate_receipt_field(receipt, "rungs")
    assert [row["concurrency"] for row in rows] == [1, 4, 16, 32]
    print(json.dumps({"pin": PIN, "sources": check_sources(),
                      "cluster_counterexample": cluster_counterexample(),
                      "rungs": {x["concurrency"]: Rung(x["concurrency"], x["acceptance"],
                          x["tok_s_per_user_p50"]).estimate() for x in rows}}, indent=2))
