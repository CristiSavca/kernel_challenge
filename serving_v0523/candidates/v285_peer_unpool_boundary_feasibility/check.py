"""Price a direct-read unpool boundary and enumerate producer/reader lifetimes."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess


HERE = Path(__file__).resolve().parent
SERVING = HERE.parents[1]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PREFIX = "python/sglang/"
PATHS = {
    "comm": "srt/distributed/device_communicators/custom_all_reduce_v2.py",
    "policy": "srt/distributed/device_communicators/configs/custom_all_reduce_v2.py",
    "barrier": "kernels/jit/include/sgl_kernel/distributed/communicator.cuh",
    "pull": "kernels/jit/csrc/distributed/custom_all_reduce.cuh",
    "gather": "kernels/jit/csrc/muse_spark/all_gather.cuh",
    "fused": "kernels/jit/csrc/muse_spark/all_gather_norm_residual.cuh",
    "fused_policy": "kernels/ops/communication/muse_spark_all_gather_norm_residual.py",
    "norm": "kernels/jit/csrc/muse_spark/norm_residual.cuh",
    "projection": "srt/layers/quantization/unquant.py",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_receipts():
    files = {
        role: subprocess.check_output(
            ["git", "-C", str(SERVING / "sglang"), "show", f"{PIN}:{PREFIX}{path}"]
        ).decode()
        for role, path in PATHS.items()
    }
    assert "torch.cuda.is_current_stream_capturing()" in files["comm"]
    assert "self._graph_inputs.append((input.data_ptr(), nbytes))" in files["comm"]
    assert "self._register_graph_inputs()" in files["comm"]
    assert "self.graph_params[self._graph_counter : self._graph_counter + count].copy_(rows)" in files["comm"]
    assert "params.graph_params[i]" in files["pull"]
    assert "/*num_arrives=*/2" in files["pull"] and "semaphores[rank][blockIdx.x]" in files["barrier"]
    sm90 = files["policy"].split("def _sm90_configs", 1)[1].split("return {", 1)[0]
    assert "num_pull_blocks=64" in sm90
    assert "get_device_capability()[0] >= 10" in files["fused_policy"]
    assert "muse_spark_all_gather_norm_residual_row_kernel" in files["fused"]
    assert "if ((w[j] & 0xffffu) == 0) w[j] |= 0x8000u" in files["fused"]
    assert "if ((w[j] & 0xffffu) == 0) w[j] |= 0x8000u" in files["gather"]
    assert "branch_v.load(branch, idx);" in files["norm"]
    signature = files["projection"].split("def bf16_gemm_dispatch(", 1)[1].split(") ->", 1)[0]
    assert "out:" not in signature
    return {PREFIX + PATHS[role]: digest(text.encode()) for role, text in files.items()}


def check_lifetime(require_publication, require_retirement):
    """Two ranks, two uses of one allocation. Each rank's reader consumes both shards."""
    schedules = 0
    unsafe = 0
    witness = None

    def walk(positions, history, failed):
        nonlocal schedules, unsafe, witness
        if positions == [4, 4]:
            schedules += 1
            if failed:
                unsafe += 1
                if witness is None:
                    witness = history
            return
        for rank in (0, 1):
            operation = positions[rank]
            if operation == 4:
                continue
            epoch, is_read = divmod(operation, 2)
            peer = positions[1 - rank]
            if not is_read and require_retirement and peer < 2 * epoch:
                continue
            if is_read and require_publication and peer < 2 * epoch + 1:
                continue
            # Every completed producer has installed its epoch until the next produce.
            peer_contents = (peer - 1) // 2 if peer else None
            invalid = is_read and peer_contents != epoch
            next_positions = positions.copy()
            next_positions[rank] += 1
            name = ("read" if is_read else "publish") + f"({rank}, {epoch})"
            walk(next_positions, history + [name], failed or invalid)

    walk([0, 0], [], False)
    return {"complete_interleavings": schedules, "unsafe": unsafe, "first_unsafe": witness}


def main():
    source_hashes = source_receipts()
    v283 = SERVING / "candidates/v283_rank_major_unpool_boundary/work.json"
    v64 = SERVING / "runs/abd217baf0ed49fd.json"
    previous = json.loads(v283.read_bytes())
    rungs = {r["concurrency"]: r for r in json.loads(v64.read_bytes())["result"]["rungs"]}
    layers = previous["bound_layers"]
    hidden = previous["hidden_width"]
    results = []
    for shape in previous["cases"]:
        if shape["world"] != 8:
            continue
        rows, users, ranks = (shape[key] for key in ("rows", "users", "world"))
        full = rows * hidden * 2 * layers
        peer = full * (ranks - 1) // ranks
        rung = rungs[users]
        period = rung["acceptance"] / rung["tok_s_per_user_p50"] * 1000
        results.append({
            "users": users,
            "rows": rows,
            "v283_eligible": bool(shape["modeled_eligible_boundaries"]),
            "branch_write_bytes_and_following_read_bytes_if_eligible": 2 * full,
            "incoming_raw_peer_bytes_if_eligible": peer,
            "logical_transfer_us_at_450GBs_unidirectional": peer / 450e3,
            "local_read_and_write_us_at_4_8TBs": 2 * full / 4.8e6,
            "illustrative_V64_verify_interval_ms": period,
            "saved_us_per_boundary_for_4x_this_decode_interval": period * 1000 * 0.75 / layers,
        })

    # Positive coefficients, a negative-zero residual and a normalized zero branch
    # are enough to reach distinct FP32 residual and BF16 output signs.
    zero_bits = lambda x: struct.pack(">f", x).hex()
    signed = {
        "ordinary_residual_fp32_hex": zero_bits(-0.0 + (1.0 * +0.0)),
        "lamport_residual_fp32_hex": zero_bits(-0.0 + (1.0 * -0.0)),
    }
    assert signed["ordinary_residual_fp32_hex"] == "00000000"
    assert signed["lamport_residual_fp32_hex"] == "80000000"
    lifetimes = {
        "both": check_lifetime(True, True),
        "no_publication": check_lifetime(False, True),
        "no_retirement": check_lifetime(True, False),
    }
    assert lifetimes["both"]["unsafe"] == 0
    assert all(lifetimes[key]["unsafe"] > 0 for key in ("no_publication", "no_retirement"))
    print(json.dumps({
        "pin": PIN, "source_sha256": source_hashes,
        "v64_sha256": digest(v64.read_bytes()), "v283_work_sha256": digest(v283.read_bytes()),
        "signed_zero": signed, "interleavings": lifetimes, "shapes": results,
        "scope": "Analytic incidence, signed-zero control and abstract lifetime enumeration. No H200 timing.",
    }, indent=2))


if __name__ == "__main__":
    main()
