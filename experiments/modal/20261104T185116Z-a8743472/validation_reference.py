"""Fixed synthetic CUDA validation of the serving finalizer; no model or allocation.

The root-owned runner supplies the three-file payload and owns process timeout.
Importing this module uses stdlib only. run() is the sole GPU entry point.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

CANDIDATE_SHA = "91e6173baa984fb873994cc3cd2883da16e013091f10218262785e70e613ed98"
BASELINE_SHA = "9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591"
IMAGE = "lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af"
FILES = frozenset({"candidate.py", "baseline.py", "finalizer_validation.py"})
RESULT = Path("/tmp/okc-result.json")
FOLDER = Path("/tmp/okc-serving-finalizer")
MAX_RESULT_BYTES = 1024 * 1024
BUDGET_SECONDS = 540
SMALL_VARIANT_CAP = 80
TOL = 0.015
ROW_FACTOR = 5.0
ROW_NORM_FLOOR = 0.1
SEED = 20261104


class IncompleteCoverage(RuntimeError):
    pass


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def cases():
    """Fixed directed coverage. T varies independently of compile-time H/K."""
    result = []
    def add(name, t=3, k=3, h=511, **kw):
        value = dict(name=name, t=t, k=k, h=h, apply=True, weight=True, first=True,
                     ids=True, mask="mixed", poison=True, eps=1e-8, xdtype="bfloat16",
                     sdtype="float32", wdtype="bfloat16", idtype="int32",
                     strided=False, broadcast_weight=False, pattern="normal", special=None)
        value.update(kw)
        value["seed"] = SEED + len(result) * 101
        result.append(value)
    # High-risk semantics and dispatch boundary precede broad coverage.
    add("remote_poison", mask="remote")
    add("one_live_poison", mask="one")
    add("zero_epsilon_remote", t=1, k=1, h=1, mask="remote", eps=0.)
    add("remote_infinite_weight", t=1, k=1, h=1, mask="remote", special="weight_inf")
    add("remote_nan_score", t=1, k=1, h=1, mask="remote", special="score_nan")
    add("ragged_last_column", t=31, k=5, h=257, strided=True)
    add("boundary31", t=31)
    add("boundary32", t=32, poison=False)
    add("boundary33", t=33, poison=False)
    add("zero_tokens", t=0)
    add("fallback_fp32", xdtype="float32")
    add("fallback_topk33", t=1, k=33)
    add("fallback_hidden16385", t=1, h=16385)
    add("fallback_broadcast_weight", broadcast_weight=True)
    for t in range(1, 32):
        add(f"token_{t}", t=t)
    for t, k, h in [(1, 1, 1), (3, 5, 7), (7, 2, 255), (16, 8, 256),
                    (2, 32, 512), (8, 8, 4096), (1, 1, 16384), (1, 32, 16384)]:
        add(f"geometry_{t}_{k}_{h}", t=t, k=k, h=h)
    for apply in (False, True):
        for weight in (False, True):
            for first in (False, True):
                for ids in (False, True):
                    add(f"flags_{int(apply)}{int(weight)}{int(first)}{int(ids)}",
                        apply=apply, weight=weight, first=first, ids=ids)
    for dtype in ("float16", "bfloat16", "float32", "float64"):
        add(f"aux_{dtype}", sdtype=dtype, wdtype=dtype, idtype="int64", strided=True)
    add("flat_reshaped_ids", idtype="int64", strided=True, reshape_ids=True)
    for pattern in ("cancellation", "tiny", "subnormal", "large", "zero"):
        add(f"finite_{pattern}", pattern=pattern, mask="local", poison=False)
    for special in ("score_inf", "weight_nan", "live_nan", "live_inf"):
        add(f"exception_{special}", t=1, k=1, h=1, mask="local", special=special)
    add("zero_epsilon_live_zero", t=1, k=1, h=1, mask="local", pattern="zero", eps=0.)
    add("no_norm_remote_inf_score", t=1, k=1, h=1, mask="remote", apply=False, special="score_inf")
    add("no_norm_live_positive_inf", t=1, k=1, h=1, mask="local", apply=False, special="positive_inf")
    add("no_norm_live_negative_inf", t=1, k=1, h=1, mask="local", apply=False, special="negative_inf")
    add("nan_epsilon", t=1, k=1, h=1, mask="local", eps="nan")
    add("infinite_epsilon", t=1, k=1, h=1, mask="local", eps="inf")
    add("negative_epsilon_remote", t=1, k=1, h=1, mask="remote", eps=-1.)
    return result


def manifest_sha():
    return sha(json.dumps(cases(), sort_keys=True, separators=(",", ":")))


def fast_case(c):
    return (0 < c["t"] < 32 and c["xdtype"] == "bfloat16" and
            0 < c["k"] <= 32 and 0 < c["h"] <= 16384 and not c["broadcast_weight"])


def signature_bound():
    # Conservative: score dtype is counted for inverse even though unused there.
    keys = {(c["h"], c["k"], c["apply"], c["weight"], c["first"], c["ids"],
             c["xdtype"], c["sdtype"], c["wdtype"] if c["weight"] else c["xdtype"],
             c["idtype"] if c["ids"] else c["xdtype"]) for c in cases() if fast_case(c)}
    return {"small_specialization_upper_bound": 2 * len(keys), "cap": SMALL_VARIANT_CAP,
            "includes_pointer_dtypes": True, "not_actual_compiler_cache_count": True}


def validate_payload(payload):
    if (payload.get("serving_finalizer_validation") is not True or
            payload.get("gpu") != "H200" or payload.get("timeout_seconds") != 600 or
            payload.get("validation_budget_seconds") != BUDGET_SECONDS or
            payload.get("image") != IMAGE or payload.get("case_manifest_sha256") != manifest_sha() or
            payload.get("seed") != SEED or payload.get("source_sha256") != CANDIDATE_SHA or
            payload.get("baseline_sha256") != BASELINE_SHA):
        raise ValueError("Unexpected fixed serving-finalizer validation policy")
    if set(payload.get("files", {})) != FILES or set(payload.get("file_sha256", {})) != FILES:
        raise ValueError("Expected exactly candidate.py, baseline.py, finalizer_validation.py")
    for name, source in payload["files"].items():
        if type(source) is not str or len(source.encode()) > MAX_RESULT_BYTES or sha(source) != payload["file_sha256"][name]:
            raise ValueError(f"Invalid source payload: {name}")
    if (payload["file_sha256"]["candidate.py"] != CANDIDATE_SHA or
            payload["file_sha256"]["baseline.py"] != BASELINE_SHA or
            payload["file_sha256"]["finalizer_validation.py"] != sha(Path(__file__).read_bytes())):
        raise ValueError("Frozen helper/worker identity mismatch")
    if signature_bound()["small_specialization_upper_bound"] > SMALL_VARIANT_CAP:
        raise ValueError("Fixed manifest exceeds compile-variant budget")


def build_payload(root, image=IMAGE):
    """Host-side bounded source reads only; does not import Torch or allocate."""
    import subprocess
    root = Path(root)
    relative = "python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py"
    baseline = subprocess.check_output(
        ["git", "show", "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f:" + relative],
        cwd=root / "serving/sglang", text=True, timeout=15)
    files = {"candidate.py": (root / "serving/sglang" / relative).read_text(),
             "baseline.py": baseline, "finalizer_validation.py": Path(__file__).read_text()}
    payload = {"serving_finalizer_validation": True, "gpu": "H200", "image": image,
               "timeout_seconds": 600, "validation_budget_seconds": BUDGET_SECONDS,
               "seed": SEED, "source_sha256": CANDIDATE_SHA, "baseline_sha256": BASELINE_SHA,
               "case_manifest_sha256": manifest_sha(), "files": files,
               "file_sha256": {name: sha(source) for name, source in files.items()}}
    validate_payload(payload)
    return payload


def _save(report):
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode()
    if len(data) > MAX_RESULT_BYTES:
        raise RuntimeError("Validation report exceeded 1MiB")
    path = RESULT.with_suffix(".tmp")
    path.write_bytes(data)
    path.replace(RESULT)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.fused_post_expert_norm_weighted_sum


def _strided(torch, tensor):
    shape = list(tensor.shape)
    shape[-1] *= 2
    backing = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    view = backing[..., ::2]
    view.copy_(tensor)
    return view


def make_inputs(torch, c):
    generator = torch.Generator(device="cpu").manual_seed(c["seed"])
    t, k, h = c["t"], c["k"], c["h"]
    x = torch.randn((t, k, h), generator=generator, dtype=torch.float32)
    scores = torch.randn((t, k), generator=generator, dtype=torch.float32) / max(1, k)
    weight = 1. + 0.25 * torch.randn((h,), generator=generator, dtype=torch.float32)
    ids = torch.randint(0, 9, (t, k), generator=generator, dtype=torch.int64)
    if c["mask"] == "mixed":
        ids[(torch.arange(t * k).reshape(t, k) + c["seed"]) % 3 == 0] = -1
    elif c["mask"] == "remote":
        ids.fill_(-1)
    elif c["mask"] == "one":
        ids.fill_(-1)
        if t and k:
            ids[:, -1] = 0
    if t and k and c["mask"] == "local":
        ids[0, 0] = 0
    if c["pattern"] == "cancellation":
        x[:, 1] = -x[:, 0]
        x[:, 2:] = 0
        scores.fill_(1.)
    elif c["pattern"] == "tiny":
        x *= 1e-30
    elif c["pattern"] == "subnormal":
        x *= 1e-39
    elif c["pattern"] == "large":
        x *= 1e20
    elif c["pattern"] == "zero":
        x.zero_()
    if c["ids"] and c["poison"]:
        x[ids < 0] = float("nan")
    special = c["special"]
    if special == "weight_inf": weight.fill_(float("inf"))
    if special == "weight_nan": weight.fill_(float("nan"))
    if special == "score_inf": scores.fill_(float("inf"))
    if special == "score_nan": scores.fill_(float("nan"))
    if special == "live_nan": x.fill_(float("nan"))
    if special == "live_inf": x.fill_(float("inf"))
    if special == "positive_inf": x.fill_(float("inf")); scores.fill_(1.)
    if special == "negative_inf": x.fill_(float("-inf")); scores.fill_(1.)
    x = x.to(device="cuda", dtype=getattr(torch, c["xdtype"]))
    scores = scores.to(device="cuda", dtype=getattr(torch, c["sdtype"]))
    weight = weight.to(device="cuda", dtype=getattr(torch, c["wdtype"])) if c["weight"] else None
    ids = ids.to(device="cuda", dtype=getattr(torch, c["idtype"])) if c["ids"] else None
    if c["strided"]:
        x, scores = _strided(torch, x), _strided(torch, scores)
        if weight is not None: weight = _strided(torch, weight)
        if ids is not None: ids = _strided(torch, ids)
    if c.get("reshape_ids") and ids is not None:
        # Same flat order, unusual metadata shape and still noncontiguous.
        ids = ids.reshape(-1)[::1].reshape(k, t)
    if c["broadcast_weight"] and weight is not None:
        weight = weight.unsqueeze(0)
    return dict(expert_output=x, topk_scores=scores, norm_weight=weight, eps=float(c["eps"]),
                weight_first=c["first"], apply_norm=c["apply"], slot_expert_ids=ids)


def oracle(torch, args):
    """Independent explicit slot loop, FP32 sum-of-squares and route accumulation."""
    x, scores = args["expert_output"], args["topk_scores"]
    t, k, h = x.shape
    acc = torch.zeros((t, h), device=x.device, dtype=torch.float32)
    ids = args["slot_expert_ids"]
    if ids is not None: ids = ids.reshape(t, k)
    weight = args["norm_weight"]
    if weight is not None: weight = weight.float()
    for slot in range(k):
        row = x[:, slot].float()
        if ids is not None:
            row = torch.where((ids[:, slot] >= 0)[:, None], row, torch.zeros_like(row))
        if args["apply_norm"]:
            if weight is not None and args["weight_first"]: row = row * weight
            variance = (row * row).sum(dim=-1, keepdim=True) / h
            row = row * torch.rsqrt(variance + args["eps"])
            if weight is not None and not args["weight_first"]: row = row * weight
        acc = acc + row * scores[:, slot].float()[:, None]
    return acc.to(dtype=x.dtype)


def compare(torch, output, reference):
    if output.shape != reference.shape or output.dtype != reference.dtype or output.device != reference.device:
        return {"pass": False, "reason": "shape/dtype/device mismatch"}
    # The threshold/floor is the official metric's mathematical definition.
    # FP64 diagnostic accumulation prevents finite large values overflowing a
    # FP32 norm/difference and accidentally passing through NaN comparisons.
    a, b = output.to(torch.float64), reference.to(torch.float64)
    classes = {}
    for name, predicate in (("nan", torch.isnan), ("positive_inf", torch.isposinf), ("negative_inf", torch.isneginf)):
        left, right = predicate(a), predicate(b)
        classes[name] = {"output": int(left.sum()), "reference": int(right.sum()),
                         "mismatch": int((left != right).sum())}
    if any(v["mismatch"] for v in classes.values()):
        return {"pass": False, "reason": "nonfinite classification mismatch", "classification": classes}
    finite = torch.isfinite(b)
    a = torch.where(finite, a, torch.zeros_like(a))
    b = torch.where(finite, b, torch.zeros_like(b))
    diff = a - b
    if a.numel() == 0:
        return {"pass": True, "whole_rel_l2": 0., "worst_row_rel_l2": 0., "max_abs": 0.,
                "equal_values": True, "exceptional": False, "classification": classes}
    whole = float(diff.norm() / b.norm().clamp(min=1e-12))
    row_ref = b.reshape(-1, b.shape[-1]).norm(dim=-1)
    row_diff = diff.reshape(-1, b.shape[-1]).norm(dim=-1)
    floor = (ROW_NORM_FLOOR * row_ref.mean()).clamp(min=1e-12)
    row_err = row_diff / torch.maximum(row_ref, floor)
    worst = float(row_err.max())
    maximum = float(diff.abs().max())
    finite_metrics = all(math.isfinite(x) for x in (whole, worst, maximum))
    return {"pass": finite_metrics and whole <= TOL and worst <= ROW_FACTOR * TOL,
            "whole_rel_l2": whole if math.isfinite(whole) else None,
            "worst_row_rel_l2": worst if math.isfinite(worst) else None,
            "max_abs": maximum if math.isfinite(maximum) else None,
            "equal_values": bool((a == b).all()), "exceptional": not bool(finite.all()),
            "classification": classes}


def _evaluate(torch, new, old, args, descriptor, actual=None):
    candidate = new(**args) if actual is None else actual
    baseline = old(**args)
    expected = oracle(torch, args)
    results = {"candidate_vs_original": compare(torch, candidate, baseline),
               "candidate_vs_fp32_oracle": compare(torch, candidate, expected),
               "original_vs_fp32_oracle": compare(torch, baseline, expected)}
    fallback = not fast_case(descriptor)
    passed = all(r["pass"] for r in results.values())
    if fallback and not results["candidate_vs_original"].get("equal_values", False):
        passed = False
    ids = args["slot_expert_ids"]
    return {"name": descriptor["name"], "seed": descriptor["seed"], "shape": list(args["expert_output"].shape),
            "descriptor": descriptor, "pass": passed, "fallback_exact_required": fallback,
            "negative_ids": int((ids < 0).sum()) if ids is not None else None,
            "tensor_metadata": {name: {"shape": list(value.shape), "stride": list(value.stride()),
                                       "dtype": str(value.dtype), "device": str(value.device)}
                                for name, value in args.items() if isinstance(value, torch.Tensor)},
            "comparisons": results}


def _profile(torch, fn, args, kind):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        output = fn(**args)
    torch.cuda.synchronize()
    counts = Counter(event.name for event in prof.events()
                     if str(event.device_type).endswith("CUDA") and event.time_range.elapsed_us() > 0)
    if not counts:
        raise IncompleteCoverage("Profiler returned no positive-duration CUDA events")
    inverse = sum(n for key, n in counts.items() if "_small_post_expert_inverse_kernel" in key)
    combine = sum(n for key, n in counts.items() if "_small_post_expert_combine_kernel" in key)
    original = sum(n for key, n in counts.items() if "_post_expert_norm_weighted_sum_kernel" in key and "_small_" not in key)
    if kind == "fast": ok = inverse == combine == 1 and original == 0
    elif kind == "large": ok = original == 1 and inverse == combine == 0
    elif kind == "eager": ok = inverse == combine == original == 0 and sum(counts.values()) > 0
    else: raise ValueError("Invalid activation profile kind")
    rows = sorted(counts.items())
    return {"kind": kind, "pass": ok, "cuda_positive_duration_events_only": True,
            "inverse_calls": inverse, "combine_calls": combine,
            "original_calls": original, "gpu_event_count": sum(counts.values()),
            "gpu_names": rows[:40], "name_list_truncated": len(rows) > 40,
            "all_names_sha256": sha(json.dumps(rows))}, output


def _copy(torch, dst, src):
    for key in dst:
        if isinstance(dst[key], torch.Tensor): dst[key].copy_(src[key])
        elif dst[key] != src[key]: raise ValueError("Graph scalar metadata changed")


@contextmanager
def _observe_empty(torch, records):
    original = torch.empty
    def observed(*args, **kwargs):
        tensor = original(*args, **kwargs)
        if tensor.is_cuda:
            records.append({"pointer": tensor.data_ptr(), "bytes": tensor.numel() * tensor.element_size(),
                            "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        return tensor  # records never retain a scratch Tensor reference
    torch.empty = observed
    try: yield
    finally: torch.empty = original


def _disjoint(left, right):
    return all(a["pointer"] + a["bytes"] <= b["pointer"] or b["pointer"] + b["bytes"] <= a["pointer"]
               for a in left for b in right if a["bytes"] and b["bytes"])


def _concurrency(torch, new, old, check_budget):
    descriptor = next(c for c in cases() if c["name"] == "token_7")
    default = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    fresh = []
    expected = []
    for index in range(8):
        d = {**descriptor, "name": f"concurrent_{index}", "seed": SEED + 70000 + index}
        args = make_inputs(torch, d)
        fresh.append((d, args))
        expected.append((old(**args), oracle(torch, args)))
    # Both queues depend on input/reference production; neither queue waits on the other.
    for stream in streams: stream.wait_stream(default)
    outputs = []
    for index, (_, args) in enumerate(fresh):
        check_budget()
        with torch.cuda.stream(streams[index % 2]): outputs.append(new(**args))
    for stream in streams: default.wait_stream(stream)
    torch.cuda.synchronize()
    checks = []
    for index, output in enumerate(outputs):
        b, o = expected[index]
        metrics = [compare(torch, output, b), compare(torch, output, o), compare(torch, b, o)]
        checks.append({"name": fresh[index][0]["name"], "seed": fresh[index][0]["seed"],
                       "pass": all(m["pass"] for m in metrics), "comparisons": metrics})
        if not checks[-1]["pass"]: raise AssertionError(json.dumps(checks[-1]))
    ptrs = [x.data_ptr() for x in outputs]
    if len(set(ptrs)) != len(ptrs): raise AssertionError("Outstanding stream outputs alias")
    return {"pass": True, "calls": len(checks), "streams": 2, "checks": checks,
            "distinct_output_storage": True, "actual_execution_overlap_measured": False}


def _graphs(torch, new, old, check_budget):
    descriptor = next(c for c in cases() if c["name"] == "token_7")
    default = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    static, graphs, outputs, allocation_records = [], [], [], []
    for which, stream in enumerate(streams):
        check_budget()
        args = make_inputs(torch, {**descriptor, "seed": SEED + 80000 + which})
        stream.wait_stream(default)
        with torch.cuda.stream(stream):
            warm = new(**args)
        default.wait_stream(stream)
        torch.cuda.synchronize()
        del warm
        graph, records = torch.cuda.CUDAGraph(), []
        with _observe_empty(torch, records):
            with torch.cuda.graph(graph, stream=stream):
                output = new(**args)
        static.append(args); graphs.append(graph); outputs.append(output); allocation_records.append(records)
        expected = sorted([([7, 511], "torch.bfloat16"), ([7, 3], "torch.float32")])
        found = sorted((r["shape"], r["dtype"]) for r in records)
        if found != expected: raise AssertionError(f"Unexpected capture allocations: {records}")
    if outputs[0].data_ptr() == outputs[1].data_ptr() or not _disjoint(*allocation_records):
        raise AssertionError("Independent graph output/inverse allocations overlap")
    pending, references = [], []
    for replay in range(4):
        for which in range(2):
            d = {**descriptor, "name": f"graph_{which}_replay_{replay}", "seed": SEED + 90000 + replay * 2 + which}
            args = make_inputs(torch, d)
            references.append((d, args, old(**args), oracle(torch, args)))
    for stream in streams: stream.wait_stream(default)
    for index, (_, args, _, _) in enumerate(references):
        check_budget()
        which = index % 2
        with torch.cuda.stream(streams[which]):
            _copy(torch, static[which], args)
            graphs[which].replay()
            pending.append(outputs[which].clone())
    for stream in streams: default.wait_stream(stream)
    torch.cuda.synchronize()
    checks = []
    for output, (d, _, baseline, expected) in zip(pending, references):
        metrics = [compare(torch, output, baseline), compare(torch, output, expected), compare(torch, baseline, expected)]
        record = {"name": d["name"], "seed": d["seed"], "pass": all(m["pass"] for m in metrics), "comparisons": metrics}
        checks.append(record)
        if not record["pass"]: raise AssertionError(json.dumps(record))
    if torch.equal(pending[0], pending[-2]) or torch.equal(pending[1], pending[-1]):
        raise AssertionError("Fresh graph values did not change both outputs")
    held_a = outputs[0].clone()
    torch.cuda.synchronize()
    with torch.cuda.stream(streams[1]): graphs[1].replay()
    streams[1].synchronize()
    if not torch.equal(held_a, outputs[0]): raise AssertionError("GraphB modified retained graphA output")
    return {"pass": True, "graphs": 2, "fresh_replays": 8, "checks": checks,
            "capture_allocations": allocation_records, "cross_graph_ranges_disjoint": True,
            "graphB_preserves_graphA_output": True, "fresh_values_changed_outputs": True,
            "scratch_tensor_references_retained": False, "actual_execution_overlap_measured": False}


def run(payload):
    start = time.monotonic()
    report = {"status": "running", "official": False, "diagnostic_only": True,
              "purpose": "serving finalizer numerical/activation/concurrency/graph validation",
              "candidate_sha256": CANDIDATE_SHA, "baseline_sha256": BASELINE_SHA,
              "worker_sha256": sha(Path(__file__).read_bytes()), "case_manifest_sha256": manifest_sha(),
              "numerical_policy": {"whole_relative_l2": TOL, "worst_row_relative_l2": TOL * ROW_FACTOR,
                                   "row_norm_floor": ROW_NORM_FLOOR, "exceptional_masks_exact": True,
                                   "diagnostic_metric_accumulation": "float64",
                                   "nonfinite_cases_are_not_official_evaluator_acceptance": True},
              "planned_cases": len(cases()), "signature_bound": signature_bound(), "cases": [], "profiles": [],
              "limits": ["Synthetic component test, not full serving/model/rank validation.",
                         "No throughput claim or actual concurrent-execution-overlap measurement.",
                         "Exceptional classifications do not prove NaN payload/bit equivalence."]}
    def checkpoint():
        report["elapsed_seconds"] = time.monotonic() - start
        _save(report)
    def check_budget():
        if time.monotonic() - start >= BUDGET_SECONDS:
            raise TimeoutError("Fixed worker budget exhausted; coverage incomplete")
    try:
        validate_payload(payload)
        report["file_sha256"] = payload["file_sha256"]
        report["image"] = payload["image"]
        checkpoint()
        import torch
        import triton
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA GPU")
        if "H200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This fixed validation requires H200")
        report["environment"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                                 "triton": triton.__version__, "cuda": torch.version.cuda,
                                 "device": torch.cuda.get_device_name(0)}
        FOLDER.mkdir(exist_ok=True)
        for name in ("candidate.py", "baseline.py"):
            (FOLDER / name).write_text(payload["files"][name])
        new = _load(FOLDER / "candidate.py", "okc_serving_finalizer_candidate")
        old = _load(FOLDER / "baseline.py", "okc_serving_finalizer_original")
        profile_cases = {"boundary31": "fast", "boundary32": "large", "boundary33": "large",
                         "fallback_fp32": "eager", "fallback_topk33": "eager", "fallback_hidden16385": "eager"}
        for c in cases():
            check_budget()
            report["active_case"] = c
            checkpoint()
            args = make_inputs(torch, c)
            item = _evaluate(torch, new, old, args, c)
            report["cases"].append(item)
            if not item["pass"]: raise AssertionError(f"Numerical failure: {c['name']}")
            if c["name"] in profile_cases:
                check_budget()
                prof, output = _profile(torch, new, args, profile_cases[c["name"]])
                prof["name"] = c["name"]
                report["profiles"].append(prof)
                if not prof["pass"]: raise AssertionError(f"Activation failure: {c['name']}")
                del output
            checkpoint()
        report.pop("active_case", None)
        check_budget(); report["active_stage"] = "two_streams"; checkpoint()
        report["two_streams"] = _concurrency(torch, new, old, check_budget)
        check_budget(); report["active_stage"] = "two_graphs"; checkpoint()
        report["two_graphs"] = _graphs(torch, new, old, check_budget)
        check_budget()
        report.pop("active_stage", None)
        metrics = [m for c in report["cases"] for m in c["comparisons"].values()]
        metrics.extend(m for section in ("two_streams", "two_graphs")
                       for c in report[section]["checks"] for m in c["comparisons"])
        report["max_errors"] = {key: max((m.get(key) or 0.) for m in metrics)
                                for key in ("whole_rel_l2", "worst_row_rel_l2", "max_abs")}
        report["completed_cases"] = len(report["cases"])
        report["status"] = "pass"
        checkpoint()
        return report
    except Exception as exc:
        report["status"] = "incomplete" if isinstance(exc, (TimeoutError, IncompleteCoverage)) else "failed"
        report["completed_cases"] = len(report["cases"])
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:12000],
                           "traceback": traceback.format_exc()[-16000:]}
        checkpoint()
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validation_worker.py /path/to/payload.json")
    run(json.loads(Path(sys.argv[1]).read_text()))
