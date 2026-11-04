"""Finish only the missing frozen coverage. Host imports stdlib; no generic PASS."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

CANDIDATE_SHA = "91e6173baa984fb873994cc3cd2883da16e013091f10218262785e70e613ed98"
BASELINE_SHA = "9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591"
REFERENCE_SHA = "c94c31345575540c74a3867512aefff1480c0dbc9471409d5c1ceb39d48f5ede"
IMAGE = "lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af"
CHECKPOINT_SHA = "e5ba29486796ba4d01c92388b3437c680c73358bb0e3bbdeee98d42e9052be95"
STAGE_SHA = "3ce10b107e0108b284be0ec7d185b6d7b3b8edfdba8d02046156e2cbcf6900cc"
ORIGINAL_MANIFEST_SHA = "a57d656ba2635c8e3c6d95c93351de0aa4508e2c3b292510108c73abc535ccb2"
FILES = frozenset({"candidate.py", "baseline.py", "validation_reference.py", "finalizer_coverage.py"})
INDICES = list(range(75, 90))
SEED = 20261104
BUDGET_SECONDS = 240
MAX_RESULT_BYTES = 1024 * 1024
RESULT = Path("/tmp/okc-result.json")
FOLDER = Path("/tmp/okc-finalizer-coverage")
POLICY = dict(whole_relative_l2=0.015, worst_row_relative_l2=0.075,
              row_norm_floor=0.1, exceptional_masks_exact=True,
              diagnostic_metric_accumulation="float64",
              nonfinite_cases_are_not_official_evaluator_acceptance=True)


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def historical_failure():
    def metric(passed, equal, absolute, relative):
        return {"pass": passed, "equal_values": equal, "max_abs": absolute,
                "whole_rel_l2": relative, "worst_row_rel_l2": relative}
    return {"historical_not_rerun": True, "checkpoint_sha256": CHECKPOINT_SHA,
            "stage_result_sha256": STAGE_SHA, "original_case_index": 74,
            "name": "finite_cancellation", "seed": 20268578, "shape": [3, 3, 511],
            "pass": False, "comparisons": {
                "candidate_vs_fp32_oracle": metric(True, True, 0., 0.),
                "candidate_vs_original": metric(False, False, 2.384185791015625e-7, 1.),
                "original_vs_fp32_oracle": metric(False, False, 2.384185791015625e-7, 1582823.078587789)}}


def _reference(source):
    if sha(source) != REFERENCE_SHA: raise ValueError("Reference identity mismatch")
    module = types.ModuleType("okc_coverage_reference_contract")
    module.__file__ = "validation_reference.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module  # pinned source has stdlib-only import; no run()/CUDA invocation


def _selection(reference):
    full = reference.cases()
    if (len(full) != 90 or reference.manifest_sha() != ORIGINAL_MANIFEST_SHA or
            (reference.TOL, reference.ROW_FACTOR, reference.ROW_NORM_FLOOR) != (.015, 5., .1)):
        raise ValueError("Reference cases or numerical contract changed")
    return [full[i] for i in INDICES]


def _manifest(descriptors):
    return sha(json.dumps({"indices": INDICES, "descriptors": descriptors}, sort_keys=True, separators=(",", ":")))


def validate_payload(payload):
    policy = dict(serving_finalizer_coverage=True, gpu="H200", image=IMAGE,
                  timeout_seconds=300, validation_budget_seconds=BUDGET_SECONDS, seed=SEED,
                  source_sha256=CANDIDATE_SHA, baseline_sha256=BASELINE_SHA,
                  reference_sha256=REFERENCE_SHA, original_case_manifest_sha256=ORIGINAL_MANIFEST_SHA)
    if any(payload.get(k) != v or type(payload.get(k)) is not type(v) for k, v in policy.items()):
        raise ValueError("Unexpected fixed coverage policy")
    if set(payload.get("files", {})) != FILES or set(payload.get("file_sha256", {})) != FILES:
        raise ValueError("Expected exactly four coverage source files")
    pins = {"candidate.py": CANDIDATE_SHA, "baseline.py": BASELINE_SHA,
            "validation_reference.py": REFERENCE_SHA, "finalizer_coverage.py": sha(Path(__file__).read_bytes())}
    for name, pin in pins.items():
        source = payload["files"][name]
        if (type(source) is not str or len(source.encode()) > MAX_RESULT_BYTES or sha(source) != pin or
                payload["file_sha256"][name] != pin):
            raise ValueError(f"Frozen coverage file mismatch: {name}")
    selected = _selection(_reference(payload["files"]["validation_reference.py"]))
    if (payload.get("case_descriptors") != selected or payload.get("remaining_indices") != INDICES or
            payload.get("case_manifest_sha256") != _manifest(selected) or
            payload.get("historical_failure") != historical_failure() or payload.get("numerical_policy") != POLICY):
        raise ValueError("Coverage manifest/history/threshold mismatch")


def build_payload(root, image=IMAGE):
    root = Path(root)
    checkpoint = root / "experiments/modal/20261104T183437Z-9826c968/result-checkpoint.json"
    stage = root / "experiments/modal/20261104T185116Z-a8743472/result.json"
    if sha(checkpoint.read_bytes()) != CHECKPOINT_SHA or sha(stage.read_bytes()) != STAGE_SHA:
        raise ValueError("Historical evidence changed or missing")
    old = json.loads(checkpoint.read_text()); prior = old["cases"][74]; history = historical_failure()
    if (old["candidate_sha256"] != CANDIDATE_SHA or old["baseline_sha256"] != BASELINE_SHA or
            old["worker_sha256"] != REFERENCE_SHA or old["status"] != "failed" or
            any(prior[k] != history[k] for k in ("name", "seed", "shape", "pass")) or
            any({k: prior["comparisons"][name][k] for k in expected} != expected
                for name, expected in history["comparisons"].items())):
        raise ValueError("Historical failure attribution changed")
    relative = "python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py"
    baseline = subprocess.check_output(["git", "show", "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f:" + relative],
                                       cwd=root / "serving/sglang", text=True, timeout=15)
    files = {"candidate.py": (root / "serving/sglang" / relative).read_text(), "baseline.py": baseline,
             "validation_reference.py": (root / "tools/serving_finalizer_validation_worker.py").read_text(),
             "finalizer_coverage.py": Path(__file__).read_text()}
    selected = _selection(_reference(files["validation_reference.py"]))
    payload = dict(serving_finalizer_coverage=True, gpu="H200", image=image,
                   timeout_seconds=300, validation_budget_seconds=BUDGET_SECONDS, seed=SEED,
                   source_sha256=CANDIDATE_SHA, baseline_sha256=BASELINE_SHA, reference_sha256=REFERENCE_SHA,
                   original_case_manifest_sha256=ORIGINAL_MANIFEST_SHA, case_manifest_sha256=_manifest(selected),
                   case_descriptors=selected, remaining_indices=INDICES, historical_failure=history,
                   numerical_policy=POLICY, files=files, file_sha256={k: sha(v) for k, v in files.items()})
    validate_payload(payload)
    return payload


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _save(report):
    raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode()
    if len(raw) > MAX_RESULT_BYTES: raise RuntimeError("Coverage checkpoint exceeded1MiB")
    tmp = RESULT.with_suffix(".tmp"); tmp.write_bytes(raw); tmp.replace(RESULT)


def _metric_pass(metric):
    """Recompute the frozen comparator decision; malformed evidence is rejected."""
    if type(metric.get("pass")) is not bool: raise ValueError("Missing comparison decision")
    if metric.get("reason") == "shape/dtype/device mismatch":
        if metric["pass"] is not False: raise ValueError("Invalid shape failure")
        return False
    classes = metric.get("classification", {})
    if set(classes) != {"nan", "positive_inf", "negative_inf"}: raise ValueError("Missing exceptional classifications")
    for row in classes.values():
        if set(row) != {"output", "reference", "mismatch"} or any(type(v) is not int or v < 0 for v in row.values()):
            raise ValueError("Malformed exceptional counts")
    mismatched = any(row["mismatch"] for row in classes.values())
    if mismatched:
        if metric["pass"] is not False or metric.get("reason") != "nonfinite classification mismatch":
            raise ValueError("Nonfinite mismatch incorrectly accepted")
        return False
    if any(row["output"] != row["reference"] for row in classes.values()):
        raise ValueError("Equal classifications have unequal counts")
    if type(metric.get("equal_values")) is not bool or type(metric.get("exceptional")) is not bool:
        raise ValueError("Missing finite comparison fields")
    if metric["exceptional"] != any(row["reference"] for row in classes.values()):
        raise ValueError("Exceptional flag disagrees with classification")
    if not {"whole_rel_l2", "worst_row_rel_l2", "max_abs"} <= metric.keys():
        raise ValueError("Missing numerical comparison measurements")
    values = [metric.get(k) for k in ("whole_rel_l2", "worst_row_rel_l2", "max_abs")]
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0) for v in values):
        raise ValueError("Malformed numerical metric")
    decision = all(v is not None for v in values) and values[0] <= .015 and values[1] <= .075
    if metric["pass"] != decision: raise ValueError("Comparison pass disagrees with frozen thresholds")
    return decision


def _case_pass(row, descriptor, index, reference):
    if (row.get("original_index") != index or row.get("descriptor") != descriptor or
            row.get("name") != descriptor["name"] or row.get("seed") != descriptor["seed"] or
            row.get("shape") != [descriptor[k] for k in ("t", "k", "h")] or
            row.get("fallback_exact_required") is not (not reference.fast_case(descriptor))):
        raise ValueError("Remaining-case identity mismatch")
    metrics = row.get("comparisons", {})
    if set(metrics) != {"candidate_vs_original", "candidate_vs_fp32_oracle", "original_vs_fp32_oracle"}:
        raise ValueError("Missing three independent comparisons")
    decisions = [_metric_pass(m) for m in metrics.values()]
    decision = all(decisions) and (not row["fallback_exact_required"] or metrics["candidate_vs_original"].get("equal_values") is True)
    if type(row.get("pass")) is not bool or row["pass"] != decision: raise ValueError("Case pass disagrees with comparisons")
    return decision


def _execution_checks(section, kind):
    checks = section.get("checks", [])
    if section.get("pass") is not True or len(checks) != 8: raise ValueError("Incomplete stream/graph checks")
    for i, row in enumerate(checks):
        name = f"concurrent_{i}" if kind == "streams" else f"graph_{i % 2}_replay_{i // 2}"
        seed = SEED + (70000 if kind == "streams" else 90000) + i
        metrics = row.get("comparisons", [])
        if row.get("name") != name or row.get("seed") != seed or row.get("pass") is not True or len(metrics) != 3:
            raise ValueError("Wrong stream/graph identity")
        if not all([_metric_pass(m) for m in metrics]): raise ValueError("Stream/graph comparison failure")
    if section.get("actual_execution_overlap_measured") is not False: raise ValueError("Unmeasured overlap claim")
    if kind == "streams":
        if section.get("calls") != 8 or section.get("streams") != 2 or section.get("distinct_output_storage") is not True:
            raise ValueError("Incomplete stream ownership evidence")
    else:
        if (section.get("graphs") != 2 or section.get("fresh_replays") != 8 or
                any(section.get(k) is not True for k in ("cross_graph_ranges_disjoint", "graphB_preserves_graphA_output", "fresh_values_changed_outputs")) or
                section.get("scratch_tensor_references_retained") is not False):
            raise ValueError("Incomplete graph ownership evidence")
        groups = section.get("capture_allocations", [])
        if len(groups) != 2: raise ValueError("Missing graph allocations")
        expected = [([7, 3], "torch.float32", 84), ([7, 511], "torch.bfloat16", 7154)]
        for group in groups:
            if sorted((r.get("shape"), r.get("dtype"), r.get("bytes")) for r in group) != expected:
                raise ValueError("Unexpected graph allocation geometry")
            if any(type(r.get("pointer")) is not int or r["pointer"] <= 0 for r in group):
                raise ValueError("Invalid graph allocation pointers")
        if any(a["pointer"] < b["pointer"] + b["bytes"] and b["pointer"] < a["pointer"] + a["bytes"]
               for a in groups[0] for b in groups[1]):
            raise ValueError("Graph allocation ranges overlap")


def validate_result(result, payload):
    reference = _reference(payload["files"]["validation_reference.py"])
    selected = _selection(reference)
    pins = dict(candidate_sha256=CANDIDATE_SHA, baseline_sha256=BASELINE_SHA, reference_sha256=REFERENCE_SHA,
                worker_sha256=payload["file_sha256"]["finalizer_coverage.py"], file_sha256=payload["file_sha256"],
                case_manifest_sha256=payload["case_manifest_sha256"], image=IMAGE,
                historical_failure=historical_failure(), numerical_policy=POLICY,
                original_case_manifest_sha256=ORIGINAL_MANIFEST_SHA, planned_indices=INDICES,
                diagnostic_only=True, official=False, generic_validation_status="failed_at_finite_cancellation")
    if any(result.get(k) != v for k, v in pins.items()): raise ValueError("Coverage result identity/history mismatch")
    status = result.get("status")
    if status not in ("running", "failed", "incomplete", "coverage_complete"): raise ValueError("Invalid coverage status")
    if result.get("coverage_complete") is not (status == "coverage_complete"): raise ValueError("Inconsistent completion flag")
    rows = result.get("cases", [])
    if len(rows) > 15 or result.get("completed_cases") != len(rows): raise ValueError("Invalid remaining-case count")
    decisions = [_case_pass(row, selected[i], INDICES[i], reference) for i, row in enumerate(rows)]
    failures = [r["name"] for r, passed in zip(rows, decisions) if not passed]
    expected_pass = all(decisions) if len(rows) == 15 else None
    if result.get("remaining_case_failures") != failures or result.get("remaining_numerics_passed") is not expected_pass:
        raise ValueError("Remaining numerical outcome is concealed or inconsistent")
    if status == "coverage_complete":
        if len(rows) != 15 or result.get("input_mutation_checks") != 15 or result.get("output_alias_checks") != 15:
            raise ValueError("Missing numerical/input-ownership coverage")
        _execution_checks(result.get("two_streams", {}), "streams")
        _execution_checks(result.get("two_graphs", {}), "graphs")


def _snapshots(torch, args):
    return {k: value.detach().clone() for k, value in args.items() if isinstance(value, torch.Tensor)}


def _assert_input_ownership(torch, args, saved, output):
    for name, original in saved.items():
        value = args[name]
        if not torch.equal(value.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)):
            raise RuntimeError(f"Input mutation: {name}")
        if output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
            raise RuntimeError(f"Output aliases input storage: {name}")


def _run_cases(torch, reference, new, old, selected, report, checkpoint):
    for index, descriptor in zip(INDICES, selected):
        checkpoint(f"case_{index}:{descriptor['name']}")
        args = reference.make_inputs(torch, descriptor)
        saved = _snapshots(torch, args)
        output = new(**args)
        item = reference._evaluate(torch, new, old, args, descriptor, actual=output)
        _assert_input_ownership(torch, args, saved, output)
        item["original_index"] = index
        report["cases"].append(item)
        report["completed_cases"] = len(report["cases"])
        report["input_mutation_checks"] += 1; report["output_alias_checks"] += 1
        report["remaining_case_failures"] = [r["name"] for r in report["cases"] if not r["pass"]]
        report["remaining_numerics_passed"] = not report["remaining_case_failures"] if len(report["cases"]) == 15 else None
        checkpoint(f"case_{index}:recorded")
        # Numerical disagreement is evidence, not an unsafe CUDA/runtime state.
        # Continue these independent cases without changing any comparator.


def run(payload):
    start = time.monotonic()
    report = dict(status="running", coverage_complete=False, diagnostic_only=True, official=False,
                  generic_validation_status="failed_at_finite_cancellation",
                  candidate_sha256=CANDIDATE_SHA, baseline_sha256=BASELINE_SHA, reference_sha256=REFERENCE_SHA,
                  worker_sha256=sha(Path(__file__).read_bytes()), file_sha256=payload.get("file_sha256"), image=payload.get("image"),
                  case_manifest_sha256=payload.get("case_manifest_sha256"), original_case_manifest_sha256=ORIGINAL_MANIFEST_SHA,
                  historical_failure=historical_failure(), numerical_policy=POLICY, planned_indices=INDICES,
                  cases=[], completed_cases=0, remaining_case_failures=[], remaining_numerics_passed=None,
                  input_mutation_checks=0, output_alias_checks=0,
                  limits=["Coverage only; the prior generic numerical failure remains unresolved.",
                          "Exceptional classification is not NaN-payload equivalence or official nonfinite acceptance.",
                          "No performance/full-model/rank proof or measured stream-execution overlap.",
                          "Unchanged stream/graph helpers abort on their existing assertions; partial coverage is retained."])
    def check_budget():
        if time.monotonic() - start >= BUDGET_SECONDS: raise TimeoutError("Fixed coverage budget exhausted")
    def checkpoint(stage):
        report["active_stage"] = stage; report["elapsed_seconds"] = time.monotonic() - start
        check_budget(); _save(report)
    try:
        validate_payload(payload); checkpoint("imports")
        import torch
        import triton
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Expected exactly one H200")
        report["environment"] = dict(python=sys.version.split()[0], torch=torch.__version__, triton=triton.__version__,
                                     cuda=torch.version.cuda, device=torch.cuda.get_device_name(0))
        FOLDER.mkdir(exist_ok=True)
        for name, source in payload["files"].items(): (FOLDER / name).write_text(source)
        reference = _load(FOLDER / "validation_reference.py", "okc_coverage_reference")
        new = _load(FOLDER / "candidate.py", "okc_coverage_candidate").fused_post_expert_norm_weighted_sum
        old = _load(FOLDER / "baseline.py", "okc_coverage_original").fused_post_expert_norm_weighted_sum
        _run_cases(torch, reference, new, old, _selection(reference), report, checkpoint)
        checkpoint("two_streams")
        report["two_streams"] = reference._concurrency(torch, new, old, check_budget)
        checkpoint("two_graphs")
        report["two_graphs"] = reference._graphs(torch, new, old, check_budget)
        checkpoint("final_coverage")
        report["status"] = "coverage_complete"; report["coverage_complete"] = True
        validate_result(report, payload); checkpoint("complete")
        print(json.dumps({"status": report["status"], "remaining_case_failures": report["remaining_case_failures"],
                          "generic_validation_status": report["generic_validation_status"]}), flush=True)
        return report
    except Exception as exc:
        report["status"] = "incomplete" if isinstance(exc, TimeoutError) else "failed"
        report["coverage_complete"] = False; report["elapsed_seconds"] = time.monotonic() - start
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:12000], "traceback": traceback.format_exc()[-16000:]}
        _save(report)
        print(json.dumps({"status": report["status"], "stage": report.get("active_stage"),
                          "error": report["error"]["message"][:1500]}), flush=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: finalizer_coverage.py payload.json")
    run(json.loads(Path(sys.argv[1]).read_text()))
