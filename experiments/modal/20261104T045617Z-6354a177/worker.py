#!/usr/bin/env python3
"""Bounded, unofficial public-reference checks on one Modal H200 (no deployment)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TAG = "lmsysorg/sglang:dev-muse-spark-open"
# Public benchmark's pinned linux/amd64 image; private judge revisions can differ.
# Provenance: third_party/auto_research/018b7a1.../benchmarks/gpu/runtime-image.json
IMAGE = "lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af"
PROBLEMS = ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def moe_route_summary(ids: list[int], local_start: int, local_count: int) -> dict:
    """Describe actual standalone routes outside timing, including partial 64-row jobs."""
    counts = [0] * local_count
    for expert in ids:
        if local_start <= expert < local_start + local_count:
            counts[expert - local_start] += 1
    rows = [min(64, count - start) for count in counts for start in range(0, count, 64)]
    return {"local_start": local_start, "local_count": local_count, "local_counts": counts,
            "U": sum(count > 0 for count in counts), "J": len(rows), "local_routes": sum(counts),
            "job_rows": rows, "NTT1_jobs": sum(0 < count <= 8 for count in rows)}


def default_cases(problem: str) -> list[dict]:
    """A small debugging subset, explicitly NOT the complete official test suite."""
    if problem == "muse_bf16_proj":
        return [{"m": m, "layer": layer} for m, layer in
                ((1, "qkv_gate"), (7, "pooler"), (33, "o"), (130, "unpooler"),
                 (8, "router"), (128, "router"))]
    if problem == "muse_moe_w4a16":
        return [{"bs": m, "sharding": "ep8"} for m in (1, 8, 32, 128)]
    cases = [
        {"batch": 1, "ctx": 2048, "layer": "sliding", "q_len": 1, "page": 1},
        {"batch": 2, "ctx": 2061, "layer": "sliding", "q_len": 8, "page": 64,
         "kv_scale": 0.015625},
        {"batch": 1, "ctx": 4096, "layer": "full", "q_len": 1, "page": 1},
    ]
    if problem == "muse_attention_decode":
        for case in cases:
            case["heads"] = "tp8"
        cases[1]["dist"] = "edges"
    return cases


def run_worker(payload_path: str) -> None:
    """Runs only in the remote sandbox, using the image's own Python and packages."""
    import gc
    import importlib.util
    import platform
    import statistics
    import subprocess
    import time
    import traceback
    import torch

    payload = json.loads(Path(payload_path).read_text())
    folder = Path("/tmp/okc-input")
    folder.mkdir(exist_ok=True)
    for name, source in payload["files"].items():
        if name not in {"candidate.py", "baseline.py", "reference.py", "checks.py", "nvfp4.py"}:
            raise ValueError(f"Unexpected input file: {name}")
        (folder / name).write_text(source)
    sys.path.insert(0, str(folder))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(clone(v) for v in value)
        return value

    def copy_into(dst, src):
        if isinstance(dst, torch.Tensor):
            if dst.shape != src.shape or dst.dtype != src.dtype:
                raise ValueError("Fresh input changed a captured tensor shape or dtype")
            dst.copy_(src)
        elif isinstance(dst, dict):
            for key in dst:
                copy_into(dst[key], src[key])
        elif isinstance(dst, (tuple, list)):
            for left, right in zip(dst, src, strict=True):
                copy_into(left, right)
        elif dst != src:
            raise ValueError("Fresh input changed captured scalar metadata")

    report = {
        "status": "running", "official": False,
        "protocol": "public-reference subset; paired CUDA graphs with pre-replay L2 read sweep; replay overhead retained",
        "cache_preconditioning": {"method": "128 MiB Triton read sweep, .ca loads, observable block sums",
                                  "outside_timing": True, "dram_traffic_measured": False},
        "image": payload["image"], "problem": payload["problem"],
        "source_sha256": payload["source_sha256"], "file_sha256": payload["file_sha256"],
        "comparison_baseline": payload["comparison_baseline"],
        "incumbent": payload.get("incumbent"),
        "profile_requested": payload.get("profile", False),
        "resources_requested": payload.get("resources", False),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "device_capability": torch.cuda.get_device_capability(0), "cases": [],
    }
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "H200" not in report["gpu"]:
            raise RuntimeError(f"Expected exactly one H200, got {report['gpu']}")
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        import triton
        import triton.language as tl
        report["triton"] = triton.__version__
        reference, baseline, candidate = (load(name) for name in ("reference", "baseline", "candidate"))
        weighted = hasattr(reference, "generate_weights")
        weights = reference.generate_weights(payload["weight_seed"]) if weighted else None
        # Independent copies protect the reference and the other implementation from prepare mutation.
        states = []
        report["prepare_wall_seconds_including_weight_clone"] = {}
        for name, module in zip(("baseline", "candidate"), (baseline, candidate)):
            prepare_started = time.monotonic()
            states.append(module.prepare(clone(weights)) if weighted else None)
            torch.cuda.synchronize()
            report["prepare_wall_seconds_including_weight_clone"][name] = time.monotonic() - prepare_started
        if payload.get("resources", False):
            # Read resource metadata from this module's own compiled extensions.
            # REG/STACK/LOCAL are compiler resources, not measured spill traffic.
            import shutil
            dumper = shutil.which("cuobjdump")
            report["extension_resources"] = {}
            for name, module in zip(("baseline", "candidate"), (baseline, candidate)):
                owned = {}
                for defines, ext in getattr(module, "_EXTS", {}).items():
                    if getattr(ext, "__file__", None):
                        owned.setdefault(str(Path(ext.__file__).resolve()), []).append(list(defines))
                paths = sorted(owned)
                entries = []
                report["extension_resources"][name] = entries
                for path in paths:
                    entry = {"extension": Path(path).name, "defines": owned[path], "status": "unavailable"}
                    entries.append(entry)
                    if dumper is None or not Path(path).is_file() or Path(path).suffix != ".so":
                        entry["error"] = "cuobjdump or owned shared object unavailable"
                        continue
                    try:
                        resource = subprocess.run([dumper, "--dump-resource-usage", path],
                                                  capture_output=True, text=True, timeout=30)
                        entry.update(returncode=resource.returncode, output=resource.stdout[:40000],
                                     stderr=resource.stderr[:4000], truncated=len(resource.stdout) > 40000)
                        entry["status"] = "captured" if resource.returncode == 0 else "unavailable"
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        entry["error"] = str(exc)
                if not paths:
                    entries.append({"status": "unavailable", "error": "no module-owned _EXTS shared objects"})
        # Read through L2 instead of relying on writes to allocate cache lines.
        # Every loaded element contributes to an observable per-block sum. This
        # is cache preconditioning, not a measurement of candidate HBM traffic.
        flush = torch.ones(128 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)
        flush_sums = torch.empty(flush.numel() // 4096, device="cuda", dtype=torch.int32)

        @triton.jit
        def read_sweep(source, output):
            block = tl.program_id(0)
            values = tl.load(source + block * 4096 + tl.arange(0, 4096), cache_modifier=".ca")
            tl.store(output + block, tl.sum(values, 0))

        def sweep_l2():
            read_sweep[(flush_sums.numel(),)](flush, flush_sums)

        # Compile and verify before capturing or timing any candidate graph.
        sweep_l2()
        torch.cuda.synchronize()
        if not bool(torch.all(flush_sums == 4096).item()):
            raise RuntimeError("L2 read sweep failed its observable-sum check")
        report["cache_preconditioning"]["sum_check_passed"] = True

        def generate(case, seed):
            return reference.generate_input(**case, seed=seed)

        def call(module, data, state):
            return module.kernel_fn(data, state) if weighted else module.kernel_fn(data)

        def check(data, output):
            raw = reference.collect(data, output) if hasattr(reference, "collect") else output
            return reference.check_implementation(data, raw, weights) if weighted else reference.check_implementation(data, raw)

        for index, case in enumerate(payload["cases"]):
            case = dict(case)
            if hasattr(reference, "case_fields") and "local_start" not in case:
                # A CUDA graph captures scalar metadata. Pick the expert bank once
                # per case, then vary all tensor inputs without changing that bank.
                # This is a debugging protocol, not the judge's router-load calibration.
                case.update(reference.case_fields(
                    {**case, "seed": payload["seed"] + 1000 * index}, weights))
            row = {"case": case, "checks": [], "timings_us": {"baseline": [], "candidate": []}}
            report["cases"].append(row)
            # Test multiple fresh inputs before capturing either implementation.
            for step in range(2):
                fresh = generate(case, payload["seed"] + 1000 * index + step)
                for name, module, state in zip(("baseline", "candidate"), (baseline, candidate), states):
                    data = clone(fresh)
                    output = call(module, data, state)
                    torch.cuda.synchronize()
                    ok, detail = check(data, output)
                    row["checks"].append({"implementation": name, "phase": "eager", "seed_step": step,
                                          "passed": bool(ok), "detail": detail})
                    if not ok:
                        raise RuntimeError(f"{name} failed correctness for {case}: {detail}")
                del fresh, data, output
            fixed = generate(case, payload["seed"] + 1000 * index + 2)
            captures = []
            for module, state in zip((baseline, candidate), states):
                data = clone(fixed)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        output = call(module, data, state)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = call(module, data, state)
                captures.append((data, graph, output))
            for repeat in range(payload["repeats"]):
                fresh = generate(case, payload["seed"] + 1000 * index + 100 + repeat)
                if payload["problem"] == "muse_moe_w4a16":
                    cfg = fresh["config"]
                    routing = moe_route_summary(fresh["topk_ids"].detach().cpu().reshape(-1).tolist(),
                                                int(cfg["local_start"]), int(cfg["local_count"]))
                    routing["input_seed"] = payload["seed"] + 1000 * index + 100 + repeat
                    row.setdefault("routing_replays", []).append(routing)
                order = (0, 1) if repeat % 2 == 0 else (1, 0)
                for side in order:
                    name = ("baseline", "candidate")[side]
                    data, graph, output = captures[side]
                    copy_into(data, fresh)
                    sweep_l2()  # Same stream, before the start event; exceeds H200 L2 capacity.
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    row["timings_us"][name].append(begin.elapsed_time(end) * 1000)
                    ok, detail = check(data, output)
                    row["checks"].append({"implementation": name, "phase": "graph", "seed_step": repeat,
                                          "passed": bool(ok), "detail": detail})
                    if not ok:
                        raise RuntimeError(f"{name} graph failed correctness for {case}: {detail}")
                del fresh
            ratios = [a / b for a, b in zip(row["timings_us"]["baseline"], row["timings_us"]["candidate"])]
            row["median_paired_speedup"] = statistics.median(ratios)
            row["median_us"] = {k: statistics.median(v) for k, v in row["timings_us"].items()}
            if payload.get("profile", False):
                # Diagnostic replays happen after timing; none of these durations enter medians.
                row["profiles"] = {}
                profile_fresh = generate(case, payload["seed"] + 1000 * index + 100 + payload["repeats"])
                for name, (data, graph, output) in zip(("baseline", "candidate"), captures):
                    try:
                        copy_into(data, profile_fresh)
                        sweep_l2()
                        torch.cuda.synchronize()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
                            graph.replay()
                            torch.cuda.synchronize()
                        kernels = {}
                        for event in profile.events():
                            if event.device_type == torch.autograd.DeviceType.CUDA:
                                kernel = kernels.setdefault(event.name, {"name": event.name, "count": 0, "total_us": 0.0})
                                kernel["count"] += 1
                                kernel["total_us"] += float(event.device_time_total)
                        if not kernels:
                            raise RuntimeError("Profiler returned no CUDA events; kernel execution is unverified")
                        row["profiles"][name] = {"status": "captured", "replays": 1,
                                                 "kernels": sorted(kernels.values(), key=lambda event: -event["total_us"])}
                    except Exception as exc:
                        row["profiles"][name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                del profile_fresh
            print(json.dumps({"case": case, "median_us": row["median_us"],
                              "unofficial_speedup": row["median_paired_speedup"]}), flush=True)
            del captures, fixed, data, graph, output
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "passed_subset"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(Path("/tmp/okc-result.json"), report)
    if report["status"] == "failed":
        print(report["traceback"], file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "benchmark", "worker"))
    parser.add_argument("--problem", choices=PROBLEMS)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--incumbent", type=Path, help="Optional paired comparison against this same-problem source")
    parser.add_argument("--profile", action="store_true", help="Record CUDA kernel names from separate diagnostic graph replays")
    parser.add_argument("--resources", action="store_true", help="Record cuobjdump resources from module-owned _EXTS shared objects")
    parser.add_argument("--cases", type=Path, help="JSON list of explicit reference.generate_input kwargs")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--weight-seed", type=int, default=7311)
    parser.add_argument("--timeout", type=int, default=300, help="Benchmark sandbox lifetime, 60..900 seconds")
    parser.add_argument("--payload", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print manifest; no Modal call")
    args = parser.parse_args()
    if args.mode == "worker":
        run_worker(args.payload)
        return
    if not 2 <= args.repeats <= 30 or not 60 <= args.timeout <= 900:
        parser.error("repeats must be 2..30 and timeout 60..900")
    if args.incumbent and args.mode != "benchmark":
        parser.error("--incumbent requires benchmark mode")
    if args.profile and args.mode != "benchmark":
        parser.error("--profile requires benchmark mode")
    if args.resources and args.mode != "benchmark":
        parser.error("--resources requires benchmark mode")
    payload = None
    if args.mode == "benchmark":
        if not args.problem or not args.candidate:
            parser.error("benchmark requires --problem and --candidate")
        source_path = args.candidate.resolve(strict=True)
        if not source_path.is_relative_to(ROOT) or source_path.suffix != ".py":
            parser.error("candidate must be a .py file inside this repository")
        source = source_path.read_text()
        if not source.splitlines() or source.splitlines()[0] != f"# okc: problem={args.problem}":
            parser.error("candidate header does not match --problem")
        compile(source, str(source_path), "exec")
        files = {"candidate.py": source}
        for name in ("baseline.py", "reference.py", "checks.py", "nvfp4.py"):
            path = ROOT / args.problem / name
            if path.exists():
                files[name] = path.read_text()
        incumbent = None
        if args.incumbent:
            incumbent_path = args.incumbent.resolve(strict=True)
            if not incumbent_path.is_relative_to(ROOT) or incumbent_path.suffix != ".py":
                parser.error("incumbent must be a .py file inside this repository")
            incumbent_source = incumbent_path.read_text()
            if not incumbent_source.splitlines() or incumbent_source.splitlines()[0] != f"# okc: problem={args.problem}":
                parser.error("incumbent header does not match --problem")
            compile(incumbent_source, str(incumbent_path), "exec")
            files["baseline.py"] = incumbent_source
            incumbent = {"path": str(incumbent_path.relative_to(ROOT)), "sha256": sha(incumbent_source)}
        cases = json.loads(args.cases.read_text()) if args.cases else default_cases(args.problem)
        if not isinstance(cases, list) or not 1 <= len(cases) <= 64 or not all(isinstance(c, dict) for c in cases):
            parser.error("cases must be a JSON list of 1..64 objects")
        if any("seed" in c or "device" in c for c in cases):
            parser.error("set seed with --seed; device is always CUDA")
        payload = {"problem": args.problem, "files": files, "image": IMAGE,
                   "source_sha256": sha(source), "file_sha256": {k: sha(v) for k, v in files.items()},
                   "cases": cases, "repeats": args.repeats, "seed": args.seed, "weight_seed": args.weight_seed,
                   "comparison_baseline": "incumbent" if incumbent else "official_public_baseline",
                   "incumbent": incumbent, "profile": args.profile, "resources": args.resources}
    worker_source = Path(__file__).read_text()
    manifest = {"mode": args.mode, "gpu": "H200:1", "timeout_seconds": 120 if args.mode == "smoke" else args.timeout,
                "official": False, "status": "prepared", "worker_sha256": sha(worker_source)}
    if payload:
        manifest.update({k: v for k, v in payload.items() if k != "files"})
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    import modal
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = ROOT / "experiments" / "modal" / stamp
    destination.mkdir(parents=True)
    manifest["modal_client"] = modal.__version__
    save(destination / "manifest.json", manifest)
    from modal_credit_budget import allocation_guard
    with allocation_guard(ROOT, stamp, manifest["timeout_seconds"]) as credit:
        manifest["credit_guard"] = credit
        guard_source = Path(__file__).with_name("modal_credit_budget.py").read_text()
        manifest["credit_guard_sha256"] = sha(guard_source)
        (destination / "credit_guard.py").write_text(guard_source)
        (destination / "credit_grant.json").write_bytes((ROOT / "tools" / "modal_credit_grant.json").read_bytes())
        save(destination / "manifest.json", manifest)
        if payload:
            # Freeze every uploaded source locally for this experiment's provenance.
            for name, source in payload["files"].items():
                (destination / name).write_text(source)
            save(destination / "request.json", {k: v for k, v in payload.items() if k != "files"})
            (destination / "worker.py").write_text(worker_source)
        image = modal.Image.debian_slim(python_version="3.12") if args.mode == "smoke" else modal.Image.from_registry(IMAGE)
        app = modal.App("okc-h200-bounded")
        sandbox = None
        try:
            with modal.enable_output(), app.run(environment_name=credit["environment"]):
                sandbox = modal.Sandbox.create("sleep", str(manifest["timeout_seconds"]), app=app, image=image,
                                              gpu="H200", cpu=(4, 4), memory=(32768, 32768), block_network=True,
                                              timeout=manifest["timeout_seconds"], idle_timeout=manifest["timeout_seconds"])
                manifest.update(status="running", sandbox_id=sandbox.object_id, dashboard=sandbox.get_dashboard_url())
                save(destination / "manifest.json", manifest)
                print(f"Sandbox: {sandbox.object_id}\nEvidence: {destination}", flush=True)
                if args.mode == "smoke":
                    command = sandbox.exec("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader", timeout=60)
                else:
                    sandbox.filesystem.write_text(worker_source, "/tmp/okc-worker.py")
                    sandbox.filesystem.write_text(json.dumps(payload), "/tmp/okc-payload.json")
                    command = sandbox.exec("python3", "/tmp/okc-worker.py", "worker", "--payload", "/tmp/okc-payload.json",
                                           timeout=max(30, args.timeout - 20))
                # Persist output continuously; stdout contains only compact case results.
                with (destination / "stdout.log").open("w") as log:
                    for line in command.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                stderr = command.stderr.read()
                (destination / "stderr.log").write_text(stderr)
                command.wait()
                manifest["exit_code"] = command.returncode
                if payload:
                    result = json.loads(sandbox.filesystem.read_text("/tmp/okc-result.json"))
                    save(destination / "result.json", result)
                    manifest["status"] = result["status"]
                else:
                    gpu_lines = (destination / "stdout.log").read_text().strip().splitlines()
                    if command.returncode == 0 and (len(gpu_lines) != 1 or "H200" not in gpu_lines[0]):
                        raise RuntimeError(f"Smoke did not report exactly one H200: {gpu_lines}")
                    manifest["status"] = "smoke_passed" if command.returncode == 0 else "failed"
                if command.returncode:
                    print(stderr[-6000:], file=sys.stderr)
                    raise RuntimeError(f"Remote process exited {command.returncode}; see {destination}")
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
                    manifest["termination_error"] = str(exc)
                    manifest["status"] = "failed"
            save(destination / "manifest.json", manifest)
            print(f"Saved: {destination}", flush=True)
            if "termination_error" in manifest:
                raise RuntimeError(f"Sandbox termination could not be confirmed; see {destination}")


if __name__ == "__main__":
    main()
