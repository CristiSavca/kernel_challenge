"""Fixed, standalone-only NCU diagnostic; never an official-score candidate."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

SOURCE_SHA = "ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a"
NATIVE_SHA = "af4f0dba97e347f8afa8f27508e40d5c03f60b402825cadba9a9537700b20d6e"
DECODER_SHA = "80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c"
NCU = "/usr/local/cuda/bin/ncu"
FOLDER = Path("/tmp/okc-input")
PAYLOAD = Path("/tmp/okc-payload.json")
CHILD_RESULT = Path("/tmp/okc-hardware-child.json")
RESULT = Path("/tmp/okc-result.json")
FILES = frozenset({"baseline.py", "reference.py", "checks.py", "nvfp4.py", "hardware_metrics.py"})
CASES = tuple({"bs": m, "sharding": "ep8", "dist": "normal", "rank": "busiest"} for m in (32, 256))
PROFILE_SEEDS = (7302, 9302)
REQUIRED = (
    "dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)
ISSUE = (
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
)
POLICY = {
    "replay_mode": "kernel", "cache_control": "all", "clock_control": "none",
    "graph_profiling": "node", "profile_from_start": "off", "launch_count": 2,
    "requested_pass_count": None,
    "pass_count_policy": "NCU chooses passes for the fixed metric list; actual counts required; no collection retry",
    "scope": "standalone MoE only; not decoder routing, decoder PDL, or benchmark speedup",
    "ordinary_timing_limitation": "Collected with profiling disabled inside an NCU-injected process; not independent benchmark evidence",
    "cache_limitation": "NCU cache-control all flushes caches on replay; distinct from the ordinary read sweep and official benchmark conditions",
    "interpretation_limit": "Low bandwidth and aggregate stall percentages do not establish a removable critical path or justify a shared-dequant rewrite; no source-PC attribution is collected",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    # A reader sees either the previous checkpoint or the complete next one.
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def validate_payload(payload: dict) -> None:
    if payload.get("problem") != "muse_moe_w4a16" or payload.get("hardware_profile") is not True:
        raise ValueError("Hardware mode is restricted to standalone MoE")
    if set(payload.get("files", {})) != FILES or set(payload.get("file_sha256", {})) != FILES:
        raise ValueError("Unexpected hardware payload file set")
    if payload.get("cases") != list(CASES):
        raise ValueError("Hardware cases must be exactly M32 and M256")
    if payload.get("source_sha256") != SOURCE_SHA or digest(payload["files"]["baseline.py"]) != SOURCE_SHA:
        raise ValueError("Hardware mode requires the exact frozen ef7 incumbent")
    for name, source in payload["files"].items():
        if not isinstance(source, str) or digest(source) != payload["file_sha256"][name]:
            raise ValueError(f"Hardware payload hash mismatch: {name}")
    if type(payload.get("repeats")) is not int or not 2 <= payload["repeats"] <= 6:
        raise ValueError("Hardware repeats must be 2..6")
    if type(payload.get("hardware_budget_seconds")) is not int or not 20 <= payload["hardware_budget_seconds"] <= 860:
        raise ValueError("Invalid hardware subprocess budget")
    if type(payload.get("hardware_issue")) is not bool:
        raise ValueError("hardware_issue must be a boolean")
    if payload.get("seed") != 7202 or payload.get("weight_seed") != 7311:
        raise ValueError("Hardware seeds are fixed: ordinary base7202, profile7302/9302, weights7311")
    if payload.get("hardware_input_seeds") != list(PROFILE_SEEDS):
        raise ValueError("Hardware profile seeds must be exactly M32=7302 and M256=9302")


def run_bounded(argv: list[str], timeout: float, cap: int = 262144) -> dict:
    """No shell; bound both pipes and kill the entire owned process group on failure."""
    if timeout <= 0 or cap <= 0:
        raise ValueError("Invalid subprocess bounds")
    started = time.monotonic()
    result = {"command": argv, "timeout_seconds": timeout, "output_limit_bytes": cap,
              "timed_out": False, "output_exceeded": False}
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
    try:
        while sel.get_map():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                result["timed_out"] = True
                break
            for key, _ in sel.select(min(remaining, 0.25)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                room = cap - sum(map(len, buffers.values()))
                buffers[key.data].extend(chunk[:max(0, room)])
                if len(chunk) > room:
                    result["output_exceeded"] = True
                    break
            if result["output_exceeded"]:
                break
        # Pipe EOF need not mean process exit: a child can close both pipes and
        # still perform cleanup. Give it the remaining budget before killing.
        if not result["timed_out"] and not result["output_exceeded"]:
            try:
                proc.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
    finally:
        if result["timed_out"] or result["output_exceeded"] or proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=5)
        sel.close()
        proc.stdout.close()
        proc.stderr.close()
    result.update(returncode=proc.returncode, elapsed_seconds=time.monotonic() - started,
                  **{key: value.decode("utf-8", errors="replace") for key, value in buffers.items()})
    return result


def command_ok(result: dict) -> bool:
    return result.get("returncode") == 0 and not result.get("timed_out") and not result.get("output_exceeded")


def available_metrics(output: str) -> set[str]:
    return set(re.findall(r"\b[a-z][a-z0-9]*__[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+\b", output))


def query_command(metrics: tuple[str, ...]) -> list[str]:
    if metrics not in (REQUIRED, ISSUE):
        raise ValueError("Metric query is not a predeclared group")
    bases = list(dict.fromkeys(name.split(".", 1)[0] for name in metrics))
    return [NCU, "--query-metrics-mode", "suffix", "--metrics", ",".join(bases)]


def collection_command(metrics: list[str]) -> list[str]:
    if metrics not in (list(REQUIRED), list(REQUIRED + ISSUE)):
        raise ValueError("Collection metrics are not a predeclared group")
    return [NCU, "--profile-from-start", "off", "--target-processes", "application-only",
            "--replay-mode", "kernel", "--cache-control", "all", "--clock-control", "none",
            "--graph-profiling", "node", "--kernel-name-base", "demangled",
            "--kernel-name", "regex:.*okc::moe_fused.*", "--launch-count", "2", "--kill", "no",
            "--rename-kernels", "off", "--disable-extra-suffixes",
            "--metrics", ",".join(metrics), "--csv", "--page", "raw", "--print-units", "base",
            sys.executable, str(FOLDER / "hardware_metrics.py"), "--child", str(PAYLOAD)]


def parse_counters(output: str, metrics: list[str]) -> dict:
    """Accept NCU long or raw-wide CSV; require two complete, unambiguous launches."""
    if metrics not in (list(REQUIRED), list(REQUIRED + ISSUE)):
        raise ValueError("Unexpected requested metric set")
    header, layout, units = None, None, None
    launches = {}

    def number(text):
        value = float(text.replace(",", ""))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Non-finite/negative hardware metric")
        return value

    def identity(entry):
        if not re.search(r"\bokc::moe_fused\s*<", entry.get("Kernel Name", "")) or not entry.get("ID", "").isdigit():
            raise ValueError("Unexpected profiled kernel identity")
        # Canonicalize IDs so 0 and 00 cannot be accepted as different launches.
        return str(int(entry["ID"])), entry["Kernel Name"]

    for row in csv.reader(io.StringIO(output)):
        if "ID" in row and "Kernel Name" in row:
            if header is not None or len(row) != len(set(row)):
                raise ValueError("Duplicate/ambiguous hardware CSV header")
            header = row
            layout = "long" if "Metric Name" in row and "Metric Value" in row else "wide"
            if layout == "long" and "Metric Unit" not in row:
                raise ValueError("Missing metric-unit column")
            if layout == "wide" and not set(metrics + ["profiler__replayer_passes"]) <= set(row):
                raise ValueError("Missing required wide metric/pass columns")
            continue
        if header is None:
            continue
        if len(row) != len(header):
            if row and row[0].isdigit():
                raise ValueError("Partial hardware CSV data row")
            continue  # NCU warnings/progress are not CSV data.
        entry = dict(zip(header, row))
        if layout == "wide" and units is None:
            if entry["ID"] or entry["Kernel Name"]:
                raise ValueError("Missing wide CSV units row")
            units = entry
            continue
        if layout == "wide":
            ident, kernel = identity(entry)
            if ident in launches:
                raise ValueError("Duplicate hardware launch row")
            if units["profiler__replayer_passes"] != "pass":
                raise ValueError("Unexpected replay pass unit")
            passes = number(entry["profiler__replayer_passes"])
            if passes <= 0 or not passes.is_integer():
                raise ValueError("Invalid replay pass count")
            launches[ident] = {"id": ident, "kernel": kernel,
                               "metrics": {name: {"value": number(entry[name]), "unit": units[name]} for name in metrics},
                               "actual_pass_count": int(passes),
                               "actual_pass_count_source": "wide profiler__replayer_passes column"}
            continue
        name = entry.get("Metric Name")
        if name not in metrics:
            continue
        ident, kernel = identity(entry)
        launch = launches.setdefault(ident, {"id": ident, "kernel": kernel, "metrics": {}})
        if launch["kernel"] != kernel or name in launch["metrics"]:
            raise ValueError("Duplicate or inconsistent hardware metric")
        launch["metrics"][name] = {"value": number(entry["Metric Value"]), "unit": entry["Metric Unit"]}
    if len(launches) != 2 or any(set(x["metrics"]) != set(metrics) for x in launches.values()):
        raise ValueError("Missing/partial hardware launches or metrics")
    ordered = sorted(launches.values(), key=lambda x: int(x["id"]))
    for size, launch in zip((32, 256), ordered):
        launch["M"] = size  # exact two launch windows, independently checked in the child report
        values = launch["metrics"]
        for name, metric in values.items():
            allowed = ({"byte"} if name in REQUIRED[:2] else {"ns", "nsecond"} if name == REQUIRED[2]
                       else {"warp"} if name == ISSUE[1] else {"%"})
            if metric["unit"] not in allowed:
                raise ValueError(f"Unexpected unit for {name}: {metric['unit']}")
        ns = values[REQUIRED[2]]["value"]
        if ns <= 0:
            raise ValueError("Non-positive profiler duration")
        launch["read_plus_write_bytes_per_second"] = sum(values[name]["value"] for name in REQUIRED[:2]) * 1e9 / ns
        launch["duration_is_benchmark_evidence"] = False
    if layout == "long":
        passes = [int(value) for value in re.findall(r"\b(\d+)\s+passes?\b", output)]
        if len(passes) != 2 or any(value <= 0 for value in passes):
            raise ValueError("Missing/ambiguous actual replay pass counts")
        for launch, count in zip(ordered, passes):
            launch["actual_pass_count"] = count
            launch["actual_pass_count_source"] = "long-output progress text"
    return {"csv_layout": layout, "launches": ordered,
            "actual_pass_counts": [launch["actual_pass_count"] for launch in ordered]}


def route_summary(ids: list[list[int]], lo: int, count: int) -> dict:
    if count != 32 or lo < 0 or lo + count > 256 or any(len(row) != 8 for row in ids):
        raise ValueError("Unexpected standalone route dimensions")
    if any(type(e) is not int or not 0 <= e < 256 for row in ids for e in row):
        raise ValueError("Invalid expert ID")
    counts = [sum(e == lo + i for row in ids for e in row) for i in range(count)]
    rows = [min(64, n - start) for n in counts for start in range(0, n, 64)]
    return {"actual_topk_ids": ids, "local_start": lo, "local_count": count, "local_counts": counts,
            "U": sum(n > 0 for n in counts), "J": len(rows), "job_rows": rows,
            "padded_job_rows": sum((n + 7) // 8 * 8 for n in rows),
            "unique_weight_bytes": 27 * 2**20 * sum(n > 0 for n in counts),
            "nominal_request_bytes": 27 * 2**20 * len(rows),
            "logical_byte_comparison_limitation": "27MiB*U and 27MiB*J are logical packed-weight comparisons, not expected DRAM-byte identities"}


def run_profile(payload: dict) -> None:
    """Remote parent: discovery then one bounded NCU process, never retries."""
    validate_payload(payload)
    started = time.monotonic()
    deadline = started + payload["hardware_budget_seconds"]
    report = {"status": "hardware_running", "official": False, "diagnostic_only": True,
              "source_sha256": SOURCE_SHA, "native_source_sha256": NATIVE_SHA,
              "decoder_source_for_lineage_only": DECODER_SHA, "file_sha256": payload["file_sha256"],
              "policy": POLICY, "preflight": {}, "selected_metrics": [], "child": None}
    def command(argv, limit=20, cap=262144):
        return run_bounded(argv, min(limit, max(0.01, deadline - time.monotonic() - 5)), cap)
    try:
        if not Path(NCU).is_file():
            raise RuntimeError("Fixed NCU binary is unavailable")
        report["preflight"]["uid"] = os.getuid()
        params = Path("/proc/driver/nvidia/params")
        report["preflight"]["profiling_permission_setting"] = [
            line for line in params.read_text()[:65536].splitlines() if line.startswith("RmProfilingAdminOnly:")
        ] if params.exists() else []
        version = command([NCU, "--version"], cap=16384)
        report["preflight"]["version"] = version
        if not command_ok(version):
            raise RuntimeError("NCU version discovery failed")
        required_query = command(query_command(REQUIRED))
        report["preflight"]["required_metric_query"] = required_query
        missing = sorted(set(REQUIRED) - available_metrics(required_query["stdout"] + required_query["stderr"]))
        if not command_ok(required_query) or missing:
            report["missing_required_metrics"] = missing
            raise RuntimeError("Required NCU metrics unavailable")
        metrics = list(REQUIRED)
        if payload["hardware_issue"]:
            optional_query = command(query_command(ISSUE))
            report["preflight"]["optional_metric_query"] = optional_query
            missing_optional = sorted(set(ISSUE) - available_metrics(optional_query["stdout"] + optional_query["stderr"]))
            if command_ok(optional_query) and not missing_optional:
                metrics.extend(ISSUE)
            else:
                report["optional_group_omitted"] = {"missing": missing_optional, "query_ok": command_ok(optional_query)}
        report["selected_metrics"] = metrics
        write_json(RESULT, report)
        if CHILD_RESULT.exists():
            raise RuntimeError("Refusing an existing child result in fresh sandbox")
        collected = command(collection_command(metrics), limit=max(0.01, deadline - time.monotonic() - 5), cap=524288)
        report["collection"] = collected
        if CHILD_RESULT.exists():
            if CHILD_RESULT.stat().st_size > 1048576:
                raise RuntimeError("Child result exceeds 1 MiB")
            report["child"] = json.loads(CHILD_RESULT.read_text())
        if not command_ok(collected):
            combined = collected["stdout"] + collected["stderr"]
            report["permission_denied"] = "ERR_NVGPUCTRPERM" in combined
            raise RuntimeError("NCU collection failed, timed out, or exceeded its output bound; no retry")
        child = report["child"]
        if not child or child.get("status") != "hardware_child_complete" or child.get("profiled_sizes") != [32, 256]:
            raise RuntimeError("Child did not complete both checked launch windows")
        report["counters"] = parse_counters(collected["stdout"] + "\n" + collected["stderr"], metrics)
        report["status"] = "hardware_complete"
    except Exception as exc:
        report["status"] = "hardware_unavailable"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(RESULT, report)
        print(json.dumps({"status": report["status"], "metrics": report["selected_metrics"],
                          "error": report.get("error")}), flush=True)


def run_child(payload: dict) -> None:
    """One exact prepare; all ordinary checks finish before counter collection starts."""
    validate_payload(payload)
    import statistics
    import traceback
    import torch
    import triton
    import triton.language as tl

    report = {"status": "hardware_child_running", "source_sha256": SOURCE_SHA, "policy": POLICY,
              "prepare_calls": 0, "cases": [], "profiled_sizes": [],
              "weight_seed": payload["weight_seed"], "input_seed_base": payload["seed"]}
    started = time.monotonic()
    def checkpoint():
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(CHILD_RESULT, report)
    def load(name):
        spec = importlib.util.spec_from_file_location(name, FOLDER / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    def clone(value):
        if isinstance(value, torch.Tensor): return value.clone()
        if isinstance(value, dict): return {k: clone(v) for k, v in value.items()}
        return value
    def copy_into(dst, src):
        if isinstance(dst, torch.Tensor):
            if dst.shape != src.shape or dst.dtype != src.dtype: raise ValueError("Captured input mismatch")
            dst.copy_(src)
        elif isinstance(dst, dict):
            if dst.keys() != src.keys(): raise ValueError("Captured dictionary mismatch")
            for key in dst: copy_into(dst[key], src[key])
        elif dst != src: raise ValueError("Captured scalar mismatch")
    def routes(data):
        ids = data["topk_ids"].cpu().tolist()
        lo, count = int(data["config"]["local_start"]), int(data["config"]["local_count"])
        return route_summary(ids, lo, count)
    try:
        sys.path.insert(0, str(FOLDER))
        if torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Hardware diagnostic requires exactly one H200")
        report.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                      triton=triton.__version__)
        reference, incumbent = load("reference"), load("baseline")
        if incumbent.PDL is not False:
            raise RuntimeError("This diagnostic requires standalone PDL=False")
        report["PDL"] = incumbent.PDL
        native = next(ast.literal_eval(n.value) for n in ast.parse(payload["files"]["baseline.py"]).body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))
        if digest(native) != NATIVE_SHA: raise ValueError("Native source identity mismatch")
        weights = reference.generate_weights(payload["weight_seed"])
        report["weight_tensor_bytes"] = sum(x.numel() * x.element_size() for x in weights.values())
        before = torch.cuda.memory_allocated()
        prepare_started = time.monotonic()
        report["prepare_calls"] += 1
        state = incumbent.prepare(clone(weights))
        torch.cuda.synchronize()
        report["prepare_wall_seconds_including_weight_clone"] = time.monotonic() - prepare_started
        report["memory_allocated_before_after_prepare"] = [before, torch.cuda.memory_allocated()]
        larger = state["larger"]
        actual_main = list(incumbent._main_defines(larger.get("sfx", ())))
        if "-DOKC_WGS=1" not in actual_main:
            raise RuntimeError("Actual scale fallback disabled WGS")
        report["actual_main_defines"] = actual_main + list(larger.get("sfx", ()))
        report["use_mul"] = bool(larger["use_mul"])
        report["grid"] = int(larger["grid"])
        report["owned_extensions"] = [{"defines": list(key), "file": Path(ext.__file__).name}
                                      for key, ext in incumbent._EXTS.items()]
        checkpoint()
        flush = torch.ones(128 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)
        sums = torch.empty(flush.numel() // 4096, device="cuda", dtype=torch.int32)
        @triton.jit
        def read_sweep(source, output):
            block = tl.program_id(0)
            values = tl.load(source + block * 4096 + tl.arange(0, 4096), cache_modifier=".ca")
            tl.store(output + block, tl.sum(values, 0))
        def sweep(): read_sweep[(sums.numel(),)](flush, sums)
        sweep()
        torch.cuda.synchronize()
        if not bool(torch.all(sums == 4096).item()): raise RuntimeError("Read sweep check failed")
        report["ordinary_cache_preconditioning"] = "128 MiB read sweep before timing; observable sums verified"
        captures = []
        for index, case in enumerate(payload["cases"]):
            row = {"case": case, "checks": [], "timings_us": [], "routing_replays": []}
            report["cases"].append(row)
            def check(data, output, phase, seed):
                ok, detail = reference.check_implementation(data, output, weights)
                row["checks"].append({"phase": phase, "seed": seed, "passed": bool(ok), "detail": detail})
                if not ok: raise RuntimeError(f"{phase} correctness failure at M={case['bs']}: {detail}")
            seed0 = payload["hardware_input_seeds"][index] - 100
            for step in range(2):
                fresh = reference.generate_input(**case, seed=seed0 + step)
                output = incumbent.kernel_fn(fresh, state)
                torch.cuda.synchronize()
                check(fresh, output, "eager", seed0 + step)
            data = reference.generate_input(**case, seed=seed0 + 2)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): output = incumbent.kernel_fn(data, state)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): output = incumbent.kernel_fn(data, state)
            for repeat in range(payload["repeats"]):
                seed = seed0 + 100 + repeat
                fresh = reference.generate_input(**case, seed=seed)
                row["routing_replays"].append({"input_seed": seed, **routes(fresh)})
                copy_into(data, fresh)
                sweep()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); graph.replay(); end.record(); end.synchronize()
                row["timings_us"].append(begin.elapsed_time(end) * 1000)
                check(data, output, "unprofiled_graph_in_ncu_process", seed)
            row["median_us"] = statistics.median(row["timings_us"])
            row["timings_are_independent_benchmark_evidence"] = False
            captures.append((case, seed0 + 100, data, graph, output))
            checkpoint()
        # All ordinary timings finish before the first profiler start.
        for row, (case, seed, data, graph, output) in zip(report["cases"], captures):
            fresh = reference.generate_input(**case, seed=seed)
            row["hardware_input_seed"] = seed
            row["hardware_routing"] = routes(fresh)
            copy_into(data, fresh)
            graph.replay()
            torch.cuda.synchronize()
            ok, detail = reference.check_implementation(data, output, weights)
            row["checks"].append({"phase": "before_ncu", "seed": seed, "passed": bool(ok), "detail": detail})
            if not ok: raise RuntimeError(f"Pre-NCU correctness failure: {detail}")
            sweep()
            torch.cuda.synchronize()
            checkpoint()
            torch.cuda.profiler.start()
            try:
                graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            report["profiled_sizes"].append(case["bs"])
            ok, detail = reference.check_implementation(data, output, weights)
            row["checks"].append({"phase": "after_ncu", "seed": seed, "passed": bool(ok), "detail": detail})
            if not ok: raise RuntimeError(f"Post-NCU correctness failure: {detail}")
            row["hardware_output_sha256"] = hashlib.sha256(output.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            checkpoint()
        report["status"] = "hardware_child_complete"
    except Exception as exc:
        report["status"] = "hardware_child_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        checkpoint()
    if report["status"] != "hardware_child_complete": raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] != ["--child", str(PAYLOAD)]:
        raise SystemExit("Only the fixed hardware child entrypoint is supported")
    run_child(json.loads(PAYLOAD.read_text()))
