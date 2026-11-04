#!/usr/bin/env python3
"""Diagnostic only; NEVER submit this driver or its instrumented source to OKC.

Cold-L2 MoE graphs using the exact decoder incumbent's CUDA source, the
standalone MoE weight packer, and unmodified public MoE reference inputs.
The route distribution is standalone MoE, not the whole decoder's router.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
MOE_HASH = "ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a"
DECODER_HASH = "80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c"
MAX_GRID = 160
TRACE_WORDS = MAX_GRID * 16
EXPERT_BYTES = 27 * 1024**2
ALLOWED_FILES = {"native.py", "reference.py", "checks.py", "nvfp4.py", "control.cu", "probe.cu"}


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing source constant {name}")


def instrument(source: str) -> str:
    """Only add A/B completion stamps; retain every original instruction/phase."""
    for phase, kind in enumerate(("false", "true")):
        anchor = f"    wgs_phase<{kind}>(p, S, ring, S.wgr[{phase * 2}], S.wgr[{phase * 2 + 1}], seq);"
        if source.count(anchor) != 1:
            raise ValueError(f"Expected exactly one phase-{phase} instrumentation anchor")
        stamp = ("\n    // DIAGNOSTIC ONLY: globaltimer is comparable across SMs.\n"
                 "    if ((threadIdx.x & 127) == 0 && p.dbg)\n"
                 f"      p.dbg[160 * 8 + blockIdx.x * 8 + (threadIdx.x / 128) * 2 + {phase}] = gtimer();")
        source = source.replace(anchor, anchor + stamp)
    return source


def route_summary(ids: list[list[int]], local_start: int, local_count: int) -> dict:
    if local_count != 32 or not 0 <= local_start <= 224:
        raise ValueError("Diagnostic requires an actual contiguous 32-expert local bank")
    counts = [0] * local_count
    for row in ids:
        if len(row) != 8 or len(set(row)) != 8 or any(type(e) is not int or not 0 <= e < 256 for e in row):
            raise ValueError("Malformed actual routing IDs")
        for expert in row:
            if local_start <= expert < local_start + local_count:
                counts[expert - local_start] += 1
    job_rows = [min(64, count - offset) for count in counts for offset in range(0, count, 64)]
    unique, jobs, routes = sum(c > 0 for c in counts), len(job_rows), sum(counts)
    if not routes:
        raise ValueError("No local routes; a readiness test would be meaningless")
    weight_bytes = unique * EXPERT_BYTES
    routed_tokens = sum(any(local_start <= e < local_start + local_count for e in row) for row in ids)
    return {"local_start": local_start, "local_count": local_count,
            "local_counts": counts, "local_routes": routes, "routed_tokens": routed_tokens, "U": unique, "J": jobs,
            "job_rows": job_rows, "padded_job_rows": sum((n + 7) // 8 * 8 for n in job_rows),
            "unique_weight_bytes": weight_bytes, "nominal_weight_request_bytes": jobs * EXPERT_BYTES,
            "cold_hbm_floor_us": weight_bytes / 4.8e12 * 1e6,
            "hbm_floor_basis": "unique packed weights only, assuming a fully cold L2",
            "minimum_input_output_bytes": (routed_tokens + len(ids)) * 4096 * 2,
            "nominal_request_time_us": jobs * EXPERT_BYTES / 4.8e12 * 1e6,
            "math_floor_us": 2 * routes * 3 * 4096**2 / 989e12 * 1e6,
            "request_time_is_lower_bound": False}


def timeline_summary(words: list[int], grid: int) -> dict:
    if not 0 < grid <= MAX_GRID or len(words) != TRACE_WORDS:
        raise ValueError("Missing, oversized or partial timeline buffer")
    timelines, warpgroups = [], []
    for cta in range(grid):
        t = words[cta * 8:(cta + 1) * 8]
        wg = words[MAX_GRID * 8 + cta * 8:MAX_GRID * 8 + (cta + 1) * 8]
        if any(type(v) is not int or v <= 0 for v in t + wg):
            raise ValueError(f"Incomplete timeline for CTA {cta}")
        if not t[0] <= t[1] <= t[7] <= t[6] <= t[2] <= t[3] <= t[4] <= t[5]:
            raise ValueError(f"Nonmonotonic global timestamps for CTA {cta}")
        for index in range(4):
            a, b = wg[index * 2:index * 2 + 2]
            if not t[6] <= a <= b <= t[4]:
                raise ValueError(f"Partial/malformed warpgroup timestamps for CTA {cta}")
            warpgroups.append({"cta": cta, "wg": index, "a_ns": a, "b_ns": b})
        timelines.append(t)
    last_a, last_b = max(w["a_ns"] for w in warpgroups), max(w["b_ns"] for w in warpgroups)
    if any(t[4] < last_b for t in timelines):
        raise ValueError("Recorded global barrier precedes a warpgroup's completion")
    start, finish = min(t[0] for t in timelines), max(t[5] for t in timelines)
    duration = finish - start
    if duration <= 0:
        raise ValueError("Empty timeline duration")
    for w in warpgroups:
        allowance = min(w["b_ns"] - w["a_ns"], max(0, last_a - w["a_ns"]))
        w["phase_window_allowance_ns"] = allowance
        w["ideal_b_ns"] = w["b_ns"] - allowance
    removable = last_b - max(w["ideal_b_ns"] for w in warpgroups)
    return {"clock": "globaltimer_ns", "kernel_span_us": duration / 1000,
            "a_completion_spread_us": (last_a - min(w["a_ns"] for w in warpgroups)) / 1000,
            "b_completion_spread_us": (last_b - min(w["b_ns"] for w in warpgroups)) / 1000,
            "phase_window_heuristic_us": removable / 1000,
            "phase_window_heuristic_fraction": removable / duration,
            "certifying_bound": False,
            "scheduler_decision": "requires correlated doneAs/BW_F/fixup intervals; low heuristic cannot reject scheduling",
            "heuristic_limitation": "independent per-WG allowances omit propagation through cross-CTA B fixups",
            "timeline_origin_ns": start,
            "cta_timestamps_ns": [[v - start for v in t] for t in timelines],
            "warpgroup_completion_ns": [{**w, "a_ns": w["a_ns"] - start,
                                           "b_ns": w["b_ns"] - start, "ideal_b_ns": w["ideal_b_ns"] - start}
                                          for w in warpgroups]}


def build_payload(seed: int, weight_seed: int, repeats: int) -> dict:
    from modal_h200 import IMAGE
    native = (ROOT / "experiments/candidates/muse_moe_w4a16" / f"{MOE_HASH}.py").read_text()
    decoder = (ROOT / "experiments/candidates/muse_decode_layer" / f"{DECODER_HASH}.py").read_text()
    if sha(native) != MOE_HASH or sha(decoder) != DECODER_HASH:
        raise ValueError("Frozen incumbent source changed")
    control = literal(decoder, "MOE_CUDA_SRC")
    # The only difference is an explanatory comment. Use exact decoder text anyway.
    standalone = literal(native, "CUDA_SRC").replace(
        "#define OKC_BIGM 32  // activates interleaved fat stream at M=32 (measured -0.45 us on Box A)",
        "#define OKC_BIGM 32")
    if standalone != control or literal(native, "CPP_SRC") != literal(decoder, "MOE_CPP_SRC"):
        raise ValueError("Standalone native core is not the decoder incumbent's core")
    flags = literal(decoder, "MOE_DEFINES")
    if flags != literal(native, "MAIN_DEFINES") or "-DOKC_WGS=1" not in flags:
        raise ValueError("Expected the reviewed incumbent WGS main flags")
    if any("DEBUG" in f or "FIT" in f or "PROF" in f for f in flags):
        raise ValueError("Control must not enable diagnostics")
    files = {"native.py": native, "control.cu": control, "probe.cu": instrument(control)}
    for name in ("reference.py", "checks.py", "nvfp4.py"):
        files[name] = (ROOT / "muse_moe_w4a16" / name).read_text()
    return {"diagnostic_only": True, "official_eligible": False, "image": IMAGE,
            "native_source_sha256": MOE_HASH, "decoder_source_sha256": DECODER_HASH,
            "files": files, "file_sha256": {k: sha(v) for k, v in files.items()},
            "control_main_flags": flags, "probe_extra_flags": ["-DOKC_DEBUG=1", "-DOKC_DIAG_WG_TIMELINE=1"],
            "sizes": [32, 128, 256], "input_seeds": [seed, seed + 1, seed + 2],
            "weight_seed": weight_seed, "repeats": repeats,
            "route_source": "unmodified standalone MoE reference, dist=normal, rank=busiest; NOT decoder routes",
            "score_limitation": "diagnostic only; no official score or promotion; graph replay overhead retained"}


def run_worker(payload_path: str) -> None:
    import importlib.util
    import time
    import traceback
    import torch
    payload = json.loads(Path(payload_path).read_text())
    report = {k: v for k, v in payload.items() if k != "files"}
    report.update(status="running", cases=[])
    started = time.monotonic()
    try:
        if set(payload["files"]) != ALLOWED_FILES or not payload["diagnostic_only"] or payload["official_eligible"]:
            raise ValueError("Unexpected diagnostic payload")
        folder = Path("/tmp/okc-moe-diagnostic")
        folder.mkdir(exist_ok=True)
        for name, source in payload["files"].items():
            if sha(source) != payload["file_sha256"][name]:
                raise ValueError(f"Source hash mismatch: {name}")
            (folder / name).write_text(source)
        sys.path.insert(0, str(folder))
        def load(name, filename):
            spec = importlib.util.spec_from_file_location(name, folder / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        if torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name():
            raise RuntimeError("Expected exactly one H200")
        report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
        reference = load("reference", "reference.py")
        control = load("diagnostic_control", "native.py")
        probe = load("diagnostic_probe", "native.py")
        for module, source_name in ((control, "control.cu"), (probe, "probe.cu")):
            module.CUDA_SRC = payload["files"][source_name]
            module.MAIN_DEFINES = tuple(payload["control_main_flags"])
            module.TINY_DEFINES = module.TINY2_DEFINES = module.MID_DEFINES = ()
        probe.MAIN_DEFINES += tuple(payload["probe_extra_flags"])
        weights = reference.generate_weights(payload["weight_seed"])
        state = control._prepare_layout(weights, True)  # one layout and one main build only
        if not state["sfx"] or "-DOKC_WGS=1" not in control._main_defines(state["sfx"]):
            raise RuntimeError("Actual weight scales selected a non-WGS fallback; diagnostic is inapplicable")
        grid = state["grid"]
        if not 0 < grid <= MAX_GRID:
            raise RuntimeError("Grid exceeds audited trace capacity")
        modules = (control, probe)
        extensions = [m._ext(tuple(m.MAIN_DEFINES) + state["sfx"], use_mul=True) for m in modules]
        scratch = [state["scratch"], probe._make_scratch(weights["w13"].device)]
        # Guard each independent scratch allocation. Timed kernels see only the original view.
        guards = []
        for buffers in scratch:
            for index, tensor in enumerate(buffers):
                raw = tensor.contiguous().view(torch.uint8)
                storage = torch.full((raw.numel() + 4096,), 0xA5, dtype=torch.uint8, device=raw.device)
                storage[:raw.numel()].copy_(raw)
                buffers[index] = storage[:raw.numel()].view(tensor.dtype).reshape(tensor.shape)
                guards.append(storage[raw.numel():])
        trace_storage = torch.full((TRACE_WORDS * 8 + 4096,), 0xA5, dtype=torch.uint8, device="cuda")
        trace = trace_storage[:TRACE_WORDS * 8].view(torch.int64)
        trace.zero_()
        guards.append(trace_storage[TRACE_WORDS * 8:])
        scratch[1].append(trace)
        report.update(grid=grid, actual_scale_flags=state["sfx"], use_mul=True,
                      actual_control_compile_flags=list(control.MAIN_DEFINES + state["sfx"]) + ["-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3"],
                      actual_probe_compile_flags=list(probe.MAIN_DEFINES + state["sfx"]) + ["-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3"])
        flush = torch.empty(128 * 1024**2, dtype=torch.uint8, device="cuda")
        print(json.dumps({"event": "compiled", "grid": grid, "elapsed_s": time.monotonic() - started}), flush=True)
        for size in payload["sizes"]:
            fixed = reference.generate_input(bs=size, sharding="ep8", seed=payload["input_seeds"][0])
            lo, count = fixed["config"]["local_start"], fixed["config"]["local_count"]
            if (lo, count) != (state["lo"], state["cnt"]):
                raise RuntimeError("Actual input bank differs from the prepared weight bank")
            captures = []
            for side, ext in enumerate(extensions):
                ext.okc_set_pdl(int(size <= 32))  # same decoder setting, but no upstream overlap here
                out = torch.empty_like(fixed["hidden_states"])
                def call(ext=ext, buffers=scratch[side], out=out):
                    ext.okc_run(state["w"], buffers, fixed["hidden_states"], fixed["topk_ids"],
                                fixed["topk_weights"], out, lo, count, grid)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                captures.append((graph, out))
            for input_seed in payload["input_seeds"]:
                fresh = reference.generate_input(bs=size, sharding="ep8", seed=input_seed)
                if fresh["config"] != fixed["config"]:
                    raise ValueError("Fresh input changed captured scalar metadata")
                for key in ("hidden_states", "topk_ids", "topk_weights"):
                    fixed[key].copy_(fresh[key])
                ids = fresh["topk_ids"].cpu().tolist()  # actual GPU-produced routes, outside timing
                row = {"M": size, "input_seed": input_seed, "weight_seed": payload["weight_seed"],
                       "actual_topk_ids": ids, "routing": route_summary(ids, lo, count),
                       "timings_us": {"control": [], "probe": []}, "checks": []}
                expected = reference.ref_kernel(fresh, weights)
                for repeat in range(payload["repeats"]):
                    for side in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        label = ("control", "probe")[side]
                        graph, out = captures[side]
                        if side:
                            trace.zero_()  # stale/missing timestamp writes cannot pass validation
                        flush.zero_()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(); graph.replay(); end.record(); end.synchronize()
                        row["timings_us"][label].append(begin.elapsed_time(end) * 1000)
                        ok, detail = reference.compare(out, expected, reference.REL_L2_TOL)
                        row["checks"].append({"implementation": label, "replay": repeat, "passed": bool(ok), "detail": detail})
                        if not ok:
                            raise RuntimeError(f"{label} reference check failed: {detail}")
                        if side:
                            row.setdefault("timeline_replays", []).append(timeline_summary(trace.cpu().tolist(), grid))
                if any(not bool((guard == 0xA5).all()) for guard in guards):
                    raise RuntimeError("Native kernel overwrote a scratch guard")
                row["median_us"] = {k: statistics.median(v) for k, v in row["timings_us"].items()}
                row["probe_overhead_ratio"] = row["median_us"]["probe"] / row["median_us"]["control"]
                row["trace_perturbation_above_5pct"] = abs(row["probe_overhead_ratio"] - 1) > 0.05
                row["optimistic_hbm_headroom_fraction"] = max(0., 1 - row["routing"]["cold_hbm_floor_us"] / row["median_us"]["control"])
                row["output_sha256"] = {name: sha(out.view(torch.uint8).cpu().numpy().tobytes())
                                         for name, (_, out) in zip(("control", "probe"), captures)}
                report["cases"].append(row)
                print(json.dumps({"M": size, "input_seed": input_seed, "U": row["routing"]["U"],
                                  "J": row["routing"]["J"], "median_us": row["median_us"],
                                  "phase_window_heuristic": max(t["phase_window_heuristic_fraction"] for t in row["timeline_replays"]),
                                  "certifying_bound": False}), flush=True)
            del captures, graph, out, fixed, fresh, expected
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(Path("/tmp/okc-result.json"), report)
    if report["status"] == "failed":
        print(report["traceback"], file=sys.stderr)
        raise SystemExit(1)


def launch(payload: dict, timeout: int) -> None:
    """Same allocation guard and resource/lifetime bounds as modal_h200.py."""
    import modal
    from modal_credit_budget import allocation_guard
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = ROOT / "experiments/modal" / stamp
    destination.mkdir(parents=True)
    worker = Path(__file__).read_text()
    manifest = {k: v for k, v in payload.items() if k != "files"}
    manifest.update(mode="moe_wgs_diagnostic", status="prepared", official=False, gpu="H200:1",
                    timeout_seconds=timeout, worker_sha256=sha(worker), modal_client=modal.__version__)
    save(destination / "manifest.json", manifest)
    with allocation_guard(ROOT, stamp, timeout) as credit:
        manifest["credit_guard"] = credit
        guard_source = Path(__file__).with_name("modal_credit_budget.py").read_text()
        manifest["credit_guard_sha256"] = sha(guard_source)
        (destination / "credit_guard.py").write_text(guard_source)
        (destination / "credit_grant.json").write_bytes((ROOT / "tools/modal_credit_grant.json").read_bytes())
        for name, source in payload["files"].items():
            (destination / name).write_text(source)
        (destination / "worker.py").write_text(worker)
        save(destination / "request.json", {k: v for k, v in payload.items() if k != "files"})
        save(destination / "manifest.json", manifest)
        sandbox = None
        try:
            app = modal.App("okc-h200-bounded")
            with modal.enable_output(), app.run(environment_name=credit["environment"]):
                sandbox = modal.Sandbox.create("sleep", str(timeout), app=app,
                    image=modal.Image.from_registry(payload["image"]), gpu="H200", cpu=(4, 4), memory=(32768, 32768),
                    block_network=True, timeout=timeout, idle_timeout=timeout)
                manifest.update(status="running", sandbox_id=sandbox.object_id, dashboard=sandbox.get_dashboard_url())
                save(destination / "manifest.json", manifest)
                print(f"Sandbox: {sandbox.object_id}\nEvidence: {destination}", flush=True)
                sandbox.filesystem.write_text(worker, "/tmp/okc-moe-worker.py")
                sandbox.filesystem.write_text(json.dumps(payload), "/tmp/okc-payload.json")
                command = sandbox.exec("python3", "/tmp/okc-moe-worker.py", "worker", "--payload", "/tmp/okc-payload.json",
                                       timeout=max(30, timeout - 20))
                with (destination / "stdout.log").open("w") as log:
                    for line in command.stdout:
                        log.write(line); log.flush(); print(line, end="", flush=True)
                stderr = command.stderr.read()
                (destination / "stderr.log").write_text(stderr)
                command.wait()
                manifest["exit_code"] = command.returncode
                result = json.loads(sandbox.filesystem.read_text("/tmp/okc-result.json"))
                save(destination / "result.json", result)
                manifest["status"] = result["status"]
                if command.returncode:
                    raise RuntimeError(f"Diagnostic worker failed; see {destination}")
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc))
            if sandbox is None:
                manifest["sandbox_created"] = False
            raise
        finally:
            if sandbox is not None:
                try:
                    sandbox.terminate()
                    manifest["sandbox_terminated"] = True
                except Exception as exc:
                    manifest.update(status="failed", termination_error=str(exc))
            save(destination / "manifest.json", manifest)
            print(f"Saved: {destination}", flush=True)
            if "termination_error" in manifest:
                raise RuntimeError("Sandbox termination was not confirmed")


def self_test() -> None:
    ids = [[0, 1 if i == 0 else 33, 40, 41, 42, 43, 44, 45] for i in range(65)]
    r = route_summary(ids, 0, 32)
    assert (r["U"], r["J"], r["padded_job_rows"], r["local_routes"]) == (2, 3, 80, 66)
    assert r["unique_weight_bytes"] == 54 * 1024**2 and r["nominal_weight_request_bytes"] == 81 * 1024**2
    shifted = [[e + 32 if e < 32 else e + 64 for e in row] for row in ids]
    assert route_summary(shifted, 32, 32)["local_counts"] == r["local_counts"]
    words = [0] * TRACE_WORDS
    for cta in range(2):
        a = [5000 + cta * 5000 + w * 1000 for w in range(4)]
        b = [15000 + cta * 3000 + w * 1000 for w in range(4)]
        words[cta * 8:cta * 8 + 8] = [1000, 2000, a[0], b[0], 22000, 24000, 4000, 3000]
        words[MAX_GRID * 8 + cta * 8:MAX_GRID * 8 + cta * 8 + 8] = [v for pair in zip(a, b) for v in pair]
    assert timeline_summary(words, 2)["phase_window_heuristic_fraction"] == 0
    assert timeline_summary(words, 2)["certifying_bound"] is False
    words[3] = words[MAX_GRID * 8 + 1] = 30000
    for cta in range(2):
        words[cta * 8 + 4:cta * 8 + 6] = [31000, 33000]
    assert timeline_summary(words, 2)["phase_window_heuristic_fraction"] == 0.25
    missing = words.copy()
    missing[MAX_GRID * 8 + 7] = 0
    for bad in (words[:-1], [0] * TRACE_WORDS, missing):
        try:
            timeline_summary(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Partial/stale trace was accepted")
    print("PASS: histogram/job/padding/local-bank formulas; non-certifying phase heuristic; partial/stale trace rejection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "run", "worker", "self-test"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--weight-seed", type=int, default=7311)
    parser.add_argument("--payload")
    args = parser.parse_args()
    if args.mode == "worker":
        run_worker(args.payload)
        return
    if args.mode == "self-test":
        self_test()
        return
    if not 60 <= args.timeout <= 900 or not 2 <= args.repeats <= 10:
        parser.error("timeout must be 60..900 and repeats 2..10")
    payload = build_payload(args.seed, args.weight_seed, args.repeats)
    if args.mode == "dry-run":
        print(json.dumps({**{k: v for k, v in payload.items() if k != "files"},
                          "timeout_seconds": args.timeout, "worker_sha256": sha(Path(__file__).read_text())}, indent=2))
    else:
        launch(payload, args.timeout)


if __name__ == "__main__":
    main()
