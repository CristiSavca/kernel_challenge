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
        "protocol": "public-reference subset; paired cold-L2 CUDA graphs; replay overhead retained",
        "image": payload["image"], "problem": payload["problem"],
        "source_sha256": payload["source_sha256"], "file_sha256": payload["file_sha256"],
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
        report["triton"] = triton.__version__
        reference, baseline, candidate = (load(name) for name in ("reference", "baseline", "candidate"))
        weighted = hasattr(reference, "generate_weights")
        weights = reference.generate_weights(payload["weight_seed"]) if weighted else None
        # Independent copies protect the reference and the other implementation from prepare mutation.
        states = [module.prepare(clone(weights)) if weighted else None for module in (baseline, candidate)]
        flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

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
                order = (0, 1) if repeat % 2 == 0 else (1, 0)
                for side in order:
                    name = ("baseline", "candidate")[side]
                    data, graph, output = captures[side]
                    copy_into(data, fresh)
                    flush.zero_()  # Runs before the start event; exceeds H200 L2 capacity.
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
    payload = None
    if args.mode == "benchmark":
        if not args.problem or not args.candidate:
            parser.error("benchmark requires --problem and --candidate")
        source_path = args.candidate.resolve(strict=True)
        if not source_path.is_relative_to(ROOT) or source_path.suffix != ".py":
            parser.error("candidate must be a .py file inside this repository")
        source = source_path.read_text()
        if source.splitlines()[0] != f"# okc: problem={args.problem}":
            parser.error("candidate header does not match --problem")
        compile(source, str(source_path), "exec")
        files = {"candidate.py": source}
        for name in ("baseline.py", "reference.py", "checks.py", "nvfp4.py"):
            path = ROOT / args.problem / name
            if path.exists():
                files[name] = path.read_text()
        cases = json.loads(args.cases.read_text()) if args.cases else default_cases(args.problem)
        if not isinstance(cases, list) or not 1 <= len(cases) <= 64 or not all(isinstance(c, dict) for c in cases):
            parser.error("cases must be a JSON list of 1..64 objects")
        if any("seed" in c or "device" in c for c in cases):
            parser.error("set seed with --seed; device is always CUDA")
        payload = {"problem": args.problem, "files": files, "image": IMAGE,
                   "source_sha256": sha(source), "file_sha256": {k: sha(v) for k, v in files.items()},
                   "cases": cases, "repeats": args.repeats, "seed": args.seed, "weight_seed": args.weight_seed}
    manifest = {"mode": args.mode, "gpu": "H200:1", "timeout_seconds": 120 if args.mode == "smoke" else args.timeout,
                "official": False, "status": "prepared", "worker_sha256": sha(Path(__file__).read_text())}
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
    if payload:
        # Freeze every uploaded source locally for this experiment's provenance.
        for name, source in payload["files"].items():
            (destination / name).write_text(source)
        save(destination / "request.json", {k: v for k, v in payload.items() if k != "files"})
        (destination / "worker.py").write_text(Path(__file__).read_text())
    image = modal.Image.debian_slim(python_version="3.12") if args.mode == "smoke" else modal.Image.from_registry(IMAGE)
    app = modal.App("okc-h200-bounded")
    sandbox = None
    try:
        with modal.enable_output(), app.run():
            sandbox = modal.Sandbox.create("sleep", str(manifest["timeout_seconds"]), app=app, image=image,
                                          gpu="H200", cpu=(4, 4), memory=(32768, 32768), block_network=True,
                                          timeout=manifest["timeout_seconds"], idle_timeout=manifest["timeout_seconds"])
            manifest.update(status="running", sandbox_id=sandbox.object_id, dashboard=sandbox.get_dashboard_url())
            save(destination / "manifest.json", manifest)
            print(f"Sandbox: {sandbox.object_id}\nEvidence: {destination}", flush=True)
            if args.mode == "smoke":
                command = sandbox.exec("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader", timeout=60)
            else:
                sandbox.filesystem.write_text(Path(__file__).read_text(), "/tmp/okc-worker.py")
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
        save(destination / "manifest.json", manifest)
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
