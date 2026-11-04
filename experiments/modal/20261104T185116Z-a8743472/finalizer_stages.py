"""Fixed cancellation-stage diagnostic. Stdlib host import; run() alone uses CUDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import struct
import sys
import time
import traceback

CANDIDATE_SHA = "91e6173baa984fb873994cc3cd2883da16e013091f10218262785e70e613ed98"
BASELINE_SHA = "9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591"
REFERENCE_SHA = "c94c31345575540c74a3867512aefff1480c0dbc9471409d5c1ceb39d48f5ede"
IMAGE = "lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af"
FILES = frozenset({"candidate.py", "baseline.py", "validation_reference.py", "finalizer_stages.py"})
RESULT = Path("/tmp/okc-result.json")
FOLDER = Path("/tmp/okc-finalizer-stages")
MAX_RESULT_BYTES = 1024 * 1024
BUDGET_SECONDS = 240
CASE = dict(name="finite_cancellation", t=3, k=3, h=511, apply=True, weight=True,
            first=True, ids=True, mask="local", poison=False, eps=1e-8,
            xdtype="bfloat16", sdtype="float32", wdtype="bfloat16", idtype="int32",
            strided=False, broadcast_weight=False, pattern="cancellation", special=None,
            seed=20268578)


class Inconclusive(RuntimeError):
    pass


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def manifest_sha():
    return sha(json.dumps(CASE, sort_keys=True, separators=(",", ":")))


def validate_payload(payload):
    expected = dict(serving_finalizer_stages=True, gpu="H200", image=IMAGE,
                    timeout_seconds=300, validation_budget_seconds=BUDGET_SECONDS,
                    seed=CASE["seed"], source_sha256=CANDIDATE_SHA,
                    baseline_sha256=BASELINE_SHA, reference_sha256=REFERENCE_SHA,
                    case_manifest_sha256=manifest_sha())
    if any(payload.get(k) != v or type(payload.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Unexpected fixed stage diagnostic policy")
    if set(payload.get("files", {})) != FILES or set(payload.get("file_sha256", {})) != FILES:
        raise ValueError("Expected the exact four-file stage package")
    pins = {"candidate.py": CANDIDATE_SHA, "baseline.py": BASELINE_SHA,
            "validation_reference.py": REFERENCE_SHA,
            "finalizer_stages.py": sha(Path(__file__).read_bytes())}
    for name, pin in pins.items():
        source = payload["files"][name]
        if (type(source) is not str or len(source.encode()) > MAX_RESULT_BYTES or
                sha(source) != pin or payload["file_sha256"][name] != pin):
            raise ValueError(f"Stage package identity mismatch: {name}")


def build_payload(root, image=IMAGE):
    root = Path(root)
    relative = "python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py"
    original = subprocess.check_output(
        ["git", "show", "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f:" + relative],
        cwd=root / "serving/sglang", text=True, timeout=15)
    files = {"candidate.py": (root / "serving/sglang" / relative).read_text(),
             "baseline.py": original,
             "validation_reference.py": (root / "tools/serving_finalizer_validation_worker.py").read_text(),
             "finalizer_stages.py": Path(__file__).read_text()}
    payload = dict(serving_finalizer_stages=True, gpu="H200", image=image,
                   timeout_seconds=300, validation_budget_seconds=BUDGET_SECONDS,
                   seed=CASE["seed"], source_sha256=CANDIDATE_SHA,
                   baseline_sha256=BASELINE_SHA, reference_sha256=REFERENCE_SHA,
                   case_manifest_sha256=manifest_sha(), files=files,
                   file_sha256={name: sha(source) for name, source in files.items()})
    validate_payload(payload)
    return payload


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _save(report):
    raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode()
    if len(raw) > MAX_RESULT_BYTES:
        raise RuntimeError("Stage report exceeded 1MiB")
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_bytes(raw)
    temporary.replace(RESULT)


def _raw(torch, tensor):
    return bytes(tensor.detach().contiguous().view(torch.uint8).cpu().reshape(-1).tolist())


def _metadata(tensor):
    result = {"shape": list(tensor.shape), "stride": list(tensor.stride()),
              "dtype": str(tensor.dtype), "device": str(tensor.device),
              "pointer": tensor.data_ptr(), "storage_offset": tensor.storage_offset()}
    if tensor.ndim >= 2:
        rows = tensor.reshape(-1, tensor.shape[-1]) if tensor.is_contiguous() else None
        if rows is not None:
            addresses = [rows[i].data_ptr() for i in range(min(9, rows.shape[0]))]
        elif tensor.ndim == 3:
            addresses = [tensor[t, k].data_ptr() for t in range(tensor.shape[0]) for k in range(tensor.shape[1])][:9]
        else:
            addresses = [tensor[i].data_ptr() for i in range(min(9, tensor.shape[0]))]
        result["row_address_mod256"] = [p % 256 for p in addresses]
        result["row_address_mod16"] = [p % 16 for p in addresses]
    return result


def _tensor_record(torch, tensor, scalar=False):
    raw = _raw(torch, tensor)
    result = {**_metadata(tensor), "sha256": sha(raw), "bytes": len(raw),
              "raw_prefix_hex": raw[:32].hex()}
    if scalar:
        if tensor.numel() != 9 or not bool(torch.isfinite(tensor).all()):
            raise Inconclusive("Expected nine finite route scalars")
        result["values"] = tensor.detach().cpu().reshape(-1).tolist()
        width = tensor.element_size()
        result["bits_little_endian_hex"] = [raw[i:i + width].hex() for i in range(0, len(raw), width)]
    return result


def _summary(torch, tensor):
    value = tensor.to(torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise Inconclusive("Unexpected nonfinite value in the fixed finite case")
    return {"l2": float(value.norm()), "row_l2": value.reshape(-1, value.shape[-1]).norm(dim=-1).cpu().tolist(),
            "max_abs": float(value.abs().max()), "nonzero_values": int((value != 0).sum()),
            "sha256": sha(_raw(torch, tensor)), "raw_prefix_hex": _raw(torch, tensor)[:32].hex()}


def _difference(torch, left, right):
    delta = left.to(torch.float64) - right.to(torch.float64)
    positions = torch.nonzero(delta.reshape(-1) != 0).reshape(-1)[:16].cpu().tolist()
    a, b = left.reshape(-1), right.reshape(-1)
    return {"delta": _summary(torch, delta), "coordinates": [
        {"flat": i, "token": i // left.shape[-1], "column": i % left.shape[-1],
         "left": float(a[i]), "right": float(b[i])} for i in positions]}


def _bit_equal(torch, left, right):
    return left.shape == right.shape and left.dtype == right.dtype and _raw(torch, left) == _raw(torch, right)


def _original_stages(torch, args):
    x = args["expert_output"].contiguous()
    t, k, h = x.shape
    masked = torch.where((args["slot_expert_ids"].reshape(t, k) >= 0).unsqueeze(-1),
                         x, torch.zeros((), dtype=x.dtype, device=x.device))
    converted = masked.float()
    weighted = converted * args["norm_weight"].float()
    squares = weighted.pow(2)
    variance = squares.mean(-1, keepdim=True)
    epsilon_added = variance + args["eps"]
    inverse = torch.rsqrt(epsilon_added)
    normalized = weighted * inverse
    scored = normalized * args["topk_scores"].float().unsqueeze(-1)
    route_sum = scored.sum(dim=1)
    stages = dict(masked=masked, converted=converted, weighted=weighted, squares=squares,
                  variance=variance, epsilon_added=epsilon_added, inverse=inverse,
                  normalized=normalized, scored=scored, route_sum=route_sum)
    return route_sum.to(x.dtype), stages


def _oracle_stages(torch, args):
    x, scores = args["expert_output"], args["topk_scores"]
    t, k, h = x.shape
    acc = torch.zeros((t, h), device=x.device, dtype=torch.float32)
    ids, weight = args["slot_expert_ids"].reshape(t, k), args["norm_weight"].float()
    records, variances, inverses, normalized = [], [], [], []
    for slot in range(k):
        row = x[:, slot].float()
        row = torch.where((ids[:, slot] >= 0)[:, None], row, torch.zeros_like(row))
        row = row * weight
        squares = row * row
        variance = squares.sum(dim=-1, keepdim=True) / h
        inverse = torch.rsqrt(variance + args["eps"])
        row = row * inverse
        acc = acc + row * scores[:, slot].float()[:, None]
        records.append(_tensor_record(torch, squares))
        variances.append(variance); inverses.append(inverse); normalized.append(row)
    return acc.to(x.dtype), {"variance": torch.stack(variances, dim=1),
                            "inverse": torch.stack(inverses, dim=1),
                            "normalized": torch.stack(normalized, dim=1),
                            "squared_operand_metadata": records}


def _candidate_stages(torch, triton, module, args):
    x = args["expert_output"].contiguous()
    t, k, h = x.shape
    w, s, ids = args["norm_weight"].contiguous(), args["topk_scores"].contiguous(), args["slot_expert_ids"].contiguous()
    inverse = torch.empty((t, k), dtype=torch.float32, device=x.device)
    output = torch.empty((t, h), dtype=x.dtype, device=x.device)
    bh = triton.next_power_of_2(h)
    module._small_post_expert_inverse_kernel[(t, k)](
        x, w, ids, inverse, TOPK=k, H=h, eps=args["eps"], APPLY_NORM=True,
        HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=bh,
        num_warps=8 if bh >= 4096 else 4, num_stages=1, enable_fp_fusion=False)
    module._small_post_expert_combine_kernel[(t, triton.cdiv(h, 256))](
        x, w, s, ids, inverse, output, TOPK=k, H=h, APPLY_NORM=True,
        HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=256,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return output, inverse


def _comparisons(torch, reference, new, old, expected):
    return {"candidate_vs_original": reference.compare(torch, new, old),
            "candidate_vs_fp32_oracle": reference.compare(torch, new, expected),
            "original_vs_fp32_oracle": reference.compare(torch, old, expected)}


def reproduction_matches(comparisons):
    co, cp, op = (comparisons.get(k, {}) for k in ("candidate_vs_original", "candidate_vs_fp32_oracle", "original_vs_fp32_oracle"))
    return (cp.get("pass") is True and cp.get("equal_values") is True and cp.get("max_abs") == 0.0
            and co.get("pass") is False and op.get("pass") is False
            and co.get("max_abs") == op.get("max_abs") == 2.384185791015625e-7
            and co.get("whole_rel_l2") == co.get("worst_row_rel_l2") == 1.0
            and abs(op.get("whole_rel_l2", 0.) - 1582823.078587789) <= 1e-6
            and abs(op.get("worst_row_rel_l2", 0.) - 1582823.078587789) <= 1e-6)


def _validate_scalars(record, dtype, shape):
    width, fmt = (4, "<9f") if dtype == "torch.float32" else (8, "<9d")
    bits, values = record.get("bits_little_endian_hex", []), record.get("values", [])
    if (record.get("dtype") != dtype or record.get("shape") != shape or
            record.get("bytes") != 9 * width or len(bits) != 9 or len(values) != 9 or
            any(type(b) is not str or len(b) != 2 * width for b in bits) or
            any(type(v) not in (float, int) or not math.isfinite(v) for v in values)):
        raise ValueError("Incomplete route scalar record")
    try: raw = bytes.fromhex("".join(bits))
    except ValueError as exc: raise ValueError("Invalid scalar bit encoding") from exc
    if (sha(raw) != record.get("sha256") or raw[:32].hex() != record.get("raw_prefix_hex") or
            list(struct.unpack(fmt, raw)) != values):
        raise ValueError("Scalar bits/values/hash disagree")


def _validate_summary(record):
    nums = [record.get("l2"), record.get("max_abs"), *record.get("row_l2", [])]
    if (len(record.get("row_l2", [])) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in nums) or
            type(record.get("nonzero_values")) is not int or record["nonzero_values"] < 0 or
            type(record.get("sha256")) is not str or len(record["sha256"]) != 64 or
            type(record.get("raw_prefix_hex")) is not str or len(record["raw_prefix_hex"]) != 64):
        raise ValueError("Missing finite output summary")


def _validate_output_comparisons(section):
    comparisons = section.get("comparisons", {})
    if set(comparisons) != {"candidate_vs_original", "candidate_vs_fp32_oracle", "original_vs_fp32_oracle"}:
        raise ValueError("Missing three-way output comparisons")
    for record in comparisons.values():
        if (type(record.get("pass")) is not bool or type(record.get("equal_values")) is not bool or
                any(type(record.get(k)) not in (int, float) or not math.isfinite(record[k]) or record[k] < 0
                    for k in ("max_abs", "whole_rel_l2", "worst_row_rel_l2"))):
            raise ValueError("Incomplete finite output comparison")
    if set(section.get("outputs", {})) != {"candidate", "original", "oracle"}:
        raise ValueError("Missing actual output summaries")
    for output in section["outputs"].values(): _validate_summary(output)


def _layouts(torch, squares, eps):
    t, k, h = squares.shape
    padded = torch.empty((t, k, 512), dtype=squares.dtype, device=squares.device)
    padded[..., :h].copy_(squares)
    view = padded[..., :h]
    rows = [squares[tok, slot].clone() for tok in range(t) for slot in range(k)]
    values = []
    for name, operand in (("contiguous_h511", squares), ("padded_pitch512_h511", view)):
        variance = operand.mean(-1, keepdim=True)
        values.append({"name": name, "logical_values_equal": _bit_equal(torch, operand, squares),
                       "operand": _tensor_record(torch, operand),
                       "variance": _tensor_record(torch, variance, True),
                       "inverse": _tensor_record(torch, torch.rsqrt(variance + eps), True)})
    variance = torch.stack([row.mean() for row in rows]).reshape(t, k, 1)
    values.append({"name": "aligned_row_copies_h511",
                   "logical_values_equal": all(_bit_equal(torch, row, squares.reshape(-1, h)[i]) for i, row in enumerate(rows)),
                   "operands": [_tensor_record(torch, row) for row in rows],
                   "variance": _tensor_record(torch, variance, True),
                   "inverse": _tensor_record(torch, torch.rsqrt(variance + eps), True)})
    return values


def _fp64(torch, args, weighted):
    raw_product = args["expert_output"].double() * args["norm_weight"].double()
    results = []
    for name, product in (("bf16_inputs_fp64_products", raw_product), ("rounded_fp32_products_fp64", weighted.double())):
        variance = (product * product).mean(-1, keepdim=True)
        inverse = torch.rsqrt(variance + args["eps"])
        output = (product * inverse * args["topk_scores"].double().unsqueeze(-1)).sum(1)
        results.append({"name": name, "variance": _tensor_record(torch, variance, True),
                        "inverse": _tensor_record(torch, inverse, True), "output": _summary(torch, output)})
    return results


def validate_result(result, payload):
    """Stdlib reader gate; partial failure records remain valid, never complete."""
    pins = {"candidate_sha256": payload["source_sha256"], "baseline_sha256": payload["baseline_sha256"],
            "reference_sha256": payload["reference_sha256"], "worker_sha256": payload["file_sha256"]["finalizer_stages.py"],
            "case_manifest_sha256": payload["case_manifest_sha256"], "image": payload["image"],
            "file_sha256": payload["file_sha256"], "diagnostic_only": True, "official": False,
            "generic_validation_status": "failed_at_finite_cancellation"}
    if any(result.get(k) != v for k, v in pins.items()):
        raise ValueError("Stage result provenance/purpose mismatch")
    if result.get("status") not in ("running", "failed", "incomplete", "diagnostic_complete"):
        raise ValueError("Unexpected stage result status")
    if result.get("diagnostic_complete") is not (result["status"] == "diagnostic_complete"):
        raise ValueError("Inconsistent diagnostic completion flag")
    if result["status"] == "diagnostic_complete":
        if (result.get("diagnostic_complete") is not True or result.get("descriptor") != CASE or
                result.get("reproduction", {}).get("matches_retained_failure") is not True or
                not reproduction_matches(result.get("reproduction", {}).get("comparisons", {})) or
                result.get("recomposition") != {"original": True, "oracle": True, "candidate": True} or
                result.get("inputs_unchanged") is not True):
            raise ValueError("Missing diagnostic reproduction/recomposition coverage")
        layouts = result.get("layouts", [])
        if ([r.get("name") for r in layouts] != ["contiguous_h511", "padded_pitch512_h511", "aligned_row_copies_h511"] or
                any(r.get("logical_values_equal") is not True for r in layouts)):
            raise ValueError("Missing same-value layout coverage")
        if set(result.get("stages", {})) != {"original", "oracle", "candidate"}:
            raise ValueError("Missing stage evidence")
        for family in (result["stages"]["original"], result["stages"]["oracle"], *layouts):
            for key in ("variance", "inverse"):
                _validate_scalars(family.get(key, {}), "torch.float32", [3, 3, 1])
        _validate_scalars(result["stages"]["candidate"].get("inverse", {}), "torch.float32", [3, 3])
        if ([r.get("name") for r in result.get("fp64", [])] != ["bf16_inputs_fp64_products", "rounded_fp32_products_fp64"] or
                result.get("h512_control", {}).get("changed_denominator") != [511, 512]):
            raise ValueError("Missing candidate/FP64/H512 evidence")
        for family in result["fp64"]:
            for key in ("variance", "inverse"): _validate_scalars(family.get(key, {}), "torch.float64", [3, 3, 1])
            _validate_summary(family.get("output", {}))
        for name in ("reproduction", "h512_control"): _validate_output_comparisons(result[name])
        opposing = result.get("opposing_routes", {})
        flags = ("input_value_negation", "weighted_value_negation", "squares_bit_equal", "original_inverse_bit_equal",
                 "oracle_inverse_bit_equal", "candidate_inverse_bit_equal")
        if any(type(opposing.get(k)) is not bool for k in flags):
            raise ValueError("Missing opposing-route evidence")
        _validate_summary(opposing.get("original_normalized_pair_sum", {}))


def run(payload):
    start = time.monotonic()
    report = dict(status="running", diagnostic_complete=False, diagnostic_only=True, official=False,
                  generic_validation_status="failed_at_finite_cancellation", descriptor=CASE,
                  candidate_sha256=CANDIDATE_SHA, baseline_sha256=BASELINE_SHA,
                  reference_sha256=REFERENCE_SHA, worker_sha256=sha(Path(__file__).read_bytes()),
                  case_manifest_sha256=manifest_sha(), file_sha256=payload.get("file_sha256"), image=payload.get("image"),
                  limits=["Stage instrumentation can change allocations; exact recomposition is required.",
                          "Layout effects do not prove an instruction order or a universally safe shape.",
                          "No generic validation PASS, performance measurement or promotion."])
    def checkpoint(stage):
        report["active_stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - start
        if report["elapsed_seconds"] >= BUDGET_SECONDS:
            raise TimeoutError("Fixed stage budget exhausted")
        _save(report)
    try:
        validate_payload(payload); checkpoint("imports")
        import torch
        import triton
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Expected exactly one H200")
        report["environment"] = dict(python=sys.version.split()[0], torch=torch.__version__,
                                     cuda=torch.version.cuda, triton=triton.__version__, device=torch.cuda.get_device_name(0))
        FOLDER.mkdir(exist_ok=True)
        for name, source in payload["files"].items(): (FOLDER / name).write_text(source)
        candidate = _load(FOLDER / "candidate.py", "okc_stage_candidate")
        baseline = _load(FOLDER / "baseline.py", "okc_stage_baseline")
        reference = _load(FOLDER / "validation_reference.py", "okc_stage_reference")
        if next(c for c in reference.cases() if c["name"] == CASE["name"]) != CASE:
            raise ValueError("Pinned cancellation descriptor disagrees")
        args = reference.make_inputs(torch, CASE)
        report["inputs"] = {k: _tensor_record(torch, v) for k, v in args.items() if isinstance(v, torch.Tensor)}
        new = candidate.fused_post_expert_norm_weighted_sum(**args)
        old = baseline.fused_post_expert_norm_weighted_sum(**args)
        expected = reference.oracle(torch, args)
        comparisons = _comparisons(torch, reference, new, old, expected)
        report["reproduction"] = {"comparisons": comparisons, "matches_retained_failure": reproduction_matches(comparisons),
                                  "outputs": {k: _summary(torch, v) for k, v in (("candidate", new), ("original", old), ("oracle", expected))},
                                  "candidate_minus_original": _difference(torch, new, old)}
        checkpoint("recomposition")
        if not report["reproduction"]["matches_retained_failure"]:
            raise Inconclusive("Retained failure did not reproduce exactly; no stage attribution")
        staged_old, old_stages = _original_stages(torch, args)
        staged_oracle, oracle_stages = _oracle_stages(torch, args)
        staged_new, candidate_inverse = _candidate_stages(torch, triton, candidate, args)
        report["recomposition"] = {"original": _bit_equal(torch, staged_old, old),
                                   "oracle": _bit_equal(torch, staged_oracle, expected),
                                   "candidate": _bit_equal(torch, staged_new, new)}
        report["stages"] = {
            "original": {k: _tensor_record(torch, v, k in ("variance", "epsilon_added", "inverse")) for k, v in old_stages.items()},
            "oracle": {k: _tensor_record(torch, v, k in ("variance", "inverse")) if isinstance(v, torch.Tensor) else v for k, v in oracle_stages.items()},
            "candidate": {"variance_observed": False, "inverse": _tensor_record(torch, candidate_inverse, True)}}
        report["opposing_routes"] = {
            "input_value_negation": bool((args["expert_output"][:, 0] == -args["expert_output"][:, 1]).all()),
            "weighted_value_negation": bool((old_stages["weighted"][:, 0] == -old_stages["weighted"][:, 1]).all()),
            "squares_bit_equal": _bit_equal(torch, old_stages["squares"][:, 0], old_stages["squares"][:, 1]),
            "original_inverse_bit_equal": _bit_equal(torch, old_stages["inverse"][:, 0], old_stages["inverse"][:, 1]),
            "oracle_inverse_bit_equal": _bit_equal(torch, oracle_stages["inverse"][:, 0], oracle_stages["inverse"][:, 1]),
            "candidate_inverse_bit_equal": _bit_equal(torch, candidate_inverse[:, 0], candidate_inverse[:, 1]),
            "original_normalized_pair_sum": _summary(torch, old_stages["normalized"][:, 0] + old_stages["normalized"][:, 1])}
        checkpoint("layouts")
        if not all(report["recomposition"].values()):
            raise Inconclusive("Stage recomposition changed helper output; no stage attribution")
        report["layouts"] = _layouts(torch, old_stages["squares"], args["eps"])
        if not all(r["logical_values_equal"] for r in report["layouts"]):
            raise Inconclusive("Layout control changed logical input values")
        checkpoint("fp64")
        report["fp64"] = _fp64(torch, args, old_stages["weighted"])
        checkpoint("h512_control")
        padded = dict(args)
        padded["expert_output"] = torch.zeros((3, 3, 512), dtype=args["expert_output"].dtype, device="cuda")
        padded["expert_output"][..., :511].copy_(args["expert_output"])
        padded["norm_weight"] = torch.ones(512, dtype=args["norm_weight"].dtype, device="cuda")
        padded["norm_weight"][:511].copy_(args["norm_weight"])
        n512 = candidate.fused_post_expert_norm_weighted_sum(**padded)
        o512 = baseline.fused_post_expert_norm_weighted_sum(**padded)
        e512 = reference.oracle(torch, padded)
        report["h512_control"] = {"changed_denominator": [511, 512], "new_random_draws": False,
                                   "comparisons": _comparisons(torch, reference, n512, o512, e512),
                                   "outputs": {k: _summary(torch, v) for k, v in (("candidate", n512), ("original", o512), ("oracle", e512))}}
        report["inputs_unchanged"] = all(sha(_raw(torch, args[k])) == v["sha256"] for k, v in report["inputs"].items())
        checkpoint("final_coverage")
        report["status"] = "diagnostic_complete"; report["diagnostic_complete"] = True
        validate_result(report, payload)
        checkpoint("complete")
        print(json.dumps({"status": report["status"], "recomposition": report["recomposition"]}), flush=True)
        return report
    except Exception as exc:
        report["status"] = "incomplete" if isinstance(exc, (Inconclusive, TimeoutError)) else "failed"
        report["diagnostic_complete"] = False
        report["elapsed_seconds"] = time.monotonic() - start
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000], "traceback": traceback.format_exc()[-8000:]}
        _save(report)
        print(json.dumps({"status": report["status"], "stage": report.get("active_stage"), "error": report["error"]["message"]}), flush=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: finalizer_stages.py payload.json")
    run(json.loads(Path(sys.argv[1]).read_text()))
