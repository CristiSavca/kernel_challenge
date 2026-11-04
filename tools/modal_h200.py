#!/usr/bin/env python3
"""Bounded, unofficial public-reference checks on one Modal H200 (no deployment)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
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
V10_SOURCES = {
    "baseline.py": ("muse_attention_decode/candidates/decoder_parent_v10.py",
                    "dc6d0c4884dba218674421dfc6b6a237c2f5eb437069cb088bf1aef7439039f9"),
    "candidate.py": ("muse_attention_decode/candidates/decoder_q8_role_v10.py",
                     "d8be9200bb0357cacf8e48c104a44d9e57aa1e278afe2bdac27a4dd5d7d4790a"),
    "attention_resource_probe.py": ("tools/attention_resource_probe.py",
                                    "cda695f43127acb14827c1cfb235d06167284eb2f9ee213fb2856a26764d7756"),
}
V10_CASES_SHA = "8ed2eccb574fb8eb5255f9c4dc6ab52c3f0e11c5cfa97bf346119ab11122d8ad"
V10_FILES = frozenset(V10_SOURCES) | {"reference.py", "checks.py"}
V11_SOURCES = {**V10_SOURCES, "candidate.py": (
    "muse_attention_decode/candidates/decoder_q8_role_v11.py",
    "6724284b1e8aef03a3af38d1d9d008c1df680b241a37b2d688464b8eeb60633f")}


def attention_screen_sources(variant: str) -> dict:
    if variant not in ("v10", "v11"):
        raise ValueError("Unknown fixed attention screen variant")
    return V10_SOURCES if variant == "v10" else V11_SOURCES


def attention_v10_cases() -> list[dict]:
    common = dict(ctx=80000, heads="tp8", layer="full", page=1, q_len=8, spread="agentic")
    return [{**common, "batch": b, "dist": dist, **override} for b, dist, override in (
        (16, "soft", {}), (32, "coherent", {}), (32, "soft", {"q_len": 1}),
        (16, "coherent", {}), (32, "soft", {}), (16, "soft", {"layer": "sliding"}))] + [
        dict(batch=3, ctx=3003, heads="tp8", layer="full", page=64, q_len=8, dist="edges"),
        dict(batch=4, ctx=4093, heads="tp8", layer="full", page=64, q_len=8,
             dist="soft", kv_scale=0.015625)]


def validate_attention_v10_payload(payload: dict) -> None:
    """Fail closed before imports; each revision admits one fixed source pair."""
    import re
    sources = attention_screen_sources(payload.get("attention_variant", "v10"))
    if (payload.get("attention_v10_screen") is not True or payload.get("problem") != "muse_attention_decode"
            or payload.get("cases") != attention_v10_cases() or payload.get("seed") != 121021
            or payload.get("repeats") != 6 or payload.get("weight_seed") != 7311
            or payload.get("image") != IMAGE or payload.get("profile") is not True
            or payload.get("resources") is not False or set(payload.get("files", {})) != V10_FILES
            or re.fullmatch(r"[a-f0-9]{16}", payload.get("attention_v10_tag", "")) is None):
        raise ValueError("attention-v10-screen requires its fixed policy, cases, seeds and file allowlist")
    for name, source in payload["files"].items():
        if type(source) is not str or sha(source) != payload.get("file_sha256", {}).get(name):
            raise ValueError(f"attention V10 uploaded digest mismatch: {name}")
        if name in sources and sha(source) != sources[name][1]:
            raise ValueError(f"attention V10 frozen source changed: {name}")
    if (payload.get("source_sha256") != sources["candidate.py"][1]
            or payload.get("incumbent", {}).get("sha256") != sources["baseline.py"][1]
            or payload.get("attention_v10_cases_sha256") != V10_CASES_SHA):
        raise ValueError("attention V10 original-source identity mismatch")


def build_attention_v10_payload(root: Path, tag: str, *, variant: str = "v10") -> dict:
    from attention_resource_probe import instrument_source
    sources = attention_screen_sources(variant)
    cases_text = (root / "experiments/attention_q8_role_v10_cases.json").read_text()
    if sha(cases_text) != V10_CASES_SHA or json.loads(cases_text) != attention_v10_cases():
        raise ValueError("attention V10 fixed case file changed")
    files = {name: (root / path).read_text() for name, (path, _) in sources.items()}
    files.update({name: (root / "muse_attention_decode" / name).read_text()
                  for name in ("reference.py", "checks.py")})
    payload = {"problem": "muse_attention_decode", "files": files, "image": IMAGE,
               "attention_v10_screen": True, "attention_v10_tag": tag,
               "attention_v10_cases_sha256": V10_CASES_SHA,
               "source_sha256": sources["candidate.py"][1],
               "file_sha256": {name: sha(source) for name, source in files.items()},
               "cases": attention_v10_cases(), "repeats": 6, "seed": 121021, "weight_seed": 7311,
               "comparison_baseline": "exact_decoder_attention_parent",
               "incumbent": dict(zip(("path", "sha256"), sources["baseline.py"])),
               "profile": True, "resources": False}
    if variant != "v10":
        payload["attention_variant"] = variant
    validate_attention_v10_payload(payload)
    payload["attention_v10_instrumentation"] = {}
    for name in ("baseline", "candidate"):
        source, proof = instrument_source(files[name + ".py"],
            expected_sha256=sources[name + ".py"][1], tag=f"{tag}_{name}")
        compile(source, name, "exec")
        payload["attention_v10_instrumentation"][name] = proof
    for name, source in files.items():
        compile(source, name, "exec")
    return payload


def observe_attention_v10_call(module, invoke, records: list):
    """Observe only actual untimed native scalar arguments; always restore the builder."""
    original = module._module
    fields = {13: "batch", 14: "q_len", 15: "q_heads", 16: "kv_heads", 17: "page_size",
              18: "window", 19: "rope", 20: "n_splits", 21: "flat", 22: "nc",
              25: "cs", 26: "ncl", 29: "tma_flags"}

    def observed_module():
        extension = original()

        class Proxy:
            def run(self, *args):
                if len(args) != 30 or len(records) != 0 or any(type(args[i]) is not int for i in fields):
                    raise RuntimeError("attention V10 native dispatch signature changed")
                records.append({field: args[i] for i, field in fields.items()})
                return extension.run(*args)
        return Proxy()

    module._module = observed_module
    try:
        return invoke()
    finally:
        module._module = original


def attention_v10_dispatch(records: list, resources: dict, *, target: bool) -> dict:
    result = {"native_call_count": len(records), "target": target, "activation_gate": False}
    if len(records) != 1:
        result["reason"] = "native_path_not_observed"
        return result
    record = records[0]
    result.update(record)
    pairs = record["kv_heads"] // 2
    grid = [record["n_splits"], 1 if record["flat"] else record["batch"], pairs]
    q8_flat = record["q_len"] == 8 and record["flat"] == 1
    sms, occupancy = resources.get("sm_count", 0), resources.get("active_blocks_per_sm", 0)
    requested_coop = bool(record["tma_flags"] & 2)
    # Exact frozen host launch<8> rules, using observed run arguments and exported occupancy.
    resident = bool(q8_flat and grid == [2 * sms, 1, 1] and occupancy >= 2)
    coop = bool(q8_flat and requested_coop and grid[0] * grid[1] * grid[2] <= sms * occupancy)
    active = bool(resident and coop and record["window"] == 0 and record["rope"] == 0 and record["cs"] == 0)
    result.update(grid=grid, pairs=pairs, cooperative_requested=requested_coop,
                  cooperative_after_host_guard=coop if q8_flat else None,
                  resident_after_host_guard=resident if q8_flat else None,
                  selected_template="attn_kernel<8,true,0>" if q8_flat else "unchanged_path",
                  role_path_eligible=active,
                  activation_gate=active and grid == [264, 1, 1],
                  evidence="observed untimed native arguments + exact frozen host guard + runtime occupancy")
    return result


def require_attention_v10_resources(resources: dict, compiler: dict) -> None:
    if (compiler.get("complete") is not True or compiler.get("saw_ptxas") is not True
            or compiler.get("actual_ptxas_diagnostics") is not True):
        raise RuntimeError("attention V10 requires complete actual ptxas diagnostics; cache hits are insufficient")
    if any(w["scope"] != "other_kernel" and
           any(marker in w["line"].lower() for marker in ("c7515", "serializ"))
           for w in compiler.get("warnings", [])):
        raise RuntimeError("attention V10 changed-flat/unattributed serialization warning requires review")
    expected = {"status": "ok", "template": "attn_kernel<8,true,0>", "local_bytes": 0,
                "threads_per_block": 384, "smem_struct_bytes": 113664,
                "requested_dynamic_shared_bytes": 114688, "sm_count": 132,
                "compute_major": 9, "compute_minor": 0}
    if (any(resources.get(k) != v for k, v in expected.items())
            or resources.get("active_blocks_per_sm", 0) < 2
            or resources.get("max_dynamic_shared_bytes", 0) < 114688):
        raise RuntimeError("attention V10 exact-specialization resource gate failed (requires localBytes=0, two resident CTAs)")


def attention_v10_compiler_summary(probe, capture: dict) -> dict:
    import re
    result = probe.compiler_summary(capture)
    # A verbose nvcc command containing '-Xptxas=-v' alone is not compiler evidence.
    result["actual_ptxas_diagnostics"] = bool(re.search(r"(?m)^ptxas (?:info|warning|error)\b", capture["text"]))
    result["compile_log_gate"] = bool(result["complete"] and result["saw_ptxas"] and result["actual_ptxas_diagnostics"])
    return result


def save_attention_v10_logs(destination: Path, result: dict) -> None:
    for side in ("baseline", "candidate"):
        capture = result.get("attention_v10", {}).get("compilation", {}).get(side, {}).get("capture", {})
        if "text" in capture:
            text = capture["text"]
            if type(text) is not str or len(text) > 8 * 1024 * 1024:
                raise ValueError("attention V10 compiler log exceeds its capture bound")
            (destination / f"compile_{side}.log").write_text(text)


def recover_attention_v10_checkpoint(sandbox, destination: Path, manifest: dict) -> None:
    """Best effort while the bounded sandbox is alive; never replace the original failure."""
    try:
        result = json.loads(sandbox.filesystem.read_text("/tmp/okc-result.json"))
        save(destination / "result.json", result)
        save_attention_v10_logs(destination, result)
        manifest["attention_v10_checkpoint_recovered"] = True
    except Exception as exc:
        manifest["attention_v10_checkpoint_recovered"] = False
        manifest["attention_v10_retrieval_error_type"] = type(exc).__name__


HARDWARE_READ_CHECKPOINT = """import sys
with open(sys.argv[1], "rb") as stream:
    data = stream.read(int(sys.argv[2]) + 1)
if len(data) > int(sys.argv[2]):
    raise ValueError("hardware checkpoint exceeds byte limit")
sys.stdout.buffer.write(data)
"""


def recover_hardware_checkpoints(sandbox, destination: Path, manifest: dict) -> None:
    """Best effort, at most two capped reads; preserve the original failed status.

    The filesystem read_text API has no caller byte/timeout bound. These fixed
    read-only commands cap data before transfer and run for at most 10s each.
    They never execute the worker/NCU again or classify a checkpoint as a result.
    """
    from modal_hardware_metrics import SOURCE_SHA, NATIVE_SHA
    recovered = manifest["hardware_checkpoint_recovery"] = []
    for remote, local, cap, statuses in (
        ("/tmp/okc-result.json", "hardware-recovered-result.json", 4 * 1024 * 1024,
         ("hardware_running", "hardware_complete", "hardware_unavailable")),
        ("/tmp/okc-hardware-child.json", "hardware-recovered-child.json", 1024 * 1024,
         ("hardware_child_running", "hardware_child_complete", "hardware_child_failed")),
    ):
        record = {"remote_path": remote, "local_file": local, "limit_bytes": cap,
                  "status": "unavailable", "complete_measurement_claimed": False}
        recovered.append(record)
        try:
            if manifest.get("source_sha256") != SOURCE_SHA:
                raise ValueError("Hardware recovery manifest source mismatch")
            process = sandbox.exec("python3", "-c", HARDWARE_READ_CHECKPOINT, remote, str(cap), timeout=10)
            content = process.stdout.read()
            process.stderr.read()
            process.wait()
            record["remote_exit_code"] = process.returncode
            if process.returncode != 0:
                raise RuntimeError("Hardware checkpoint read failed")
            if type(content) is not str or len(content.encode("utf-8")) > cap:
                raise ValueError("Hardware checkpoint exceeds local byte limit")
            checkpoint = json.loads(content)
            if (type(checkpoint) is not dict or checkpoint.get("source_sha256") != SOURCE_SHA
                    or checkpoint.get("status") not in statuses):
                raise ValueError("Hardware checkpoint source/status mismatch")
            if remote == "/tmp/okc-result.json" and checkpoint.get("native_source_sha256") != NATIVE_SHA:
                raise ValueError("Hardware checkpoint native source mismatch")
            save(destination / local, checkpoint)
            record.update(status="recovered_checkpoint", checkpoint_status=checkpoint["status"],
                          bytes=len(content.encode("utf-8")))
        except Exception as exc:
            # Never replace the original exception or promote partial evidence.
            record["error_type"] = type(exc).__name__


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
    attention_v10 = payload.get("attention_v10_screen") is True
    if attention_v10:
        validate_attention_v10_payload(payload)
        attention_sources = attention_screen_sources(payload.get("attention_variant", "v10"))
    folder = Path("/tmp/okc-input")
    folder.mkdir(exist_ok=True)
    for name, source in payload["files"].items():
        allowed = V10_FILES if attention_v10 else {"candidate.py", "baseline.py", "reference.py", "checks.py", "nvfp4.py",
                                                  "decoder_policy_metadata.py"}
        if name not in allowed:
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

    def checkpoint(phase):
        report["checkpoint_phase"] = phase
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = Path("/tmp/okc-result.json.tmp")
        save(temporary, report)
        temporary.replace("/tmp/okc-result.json")
        print(json.dumps({"attention_v10_checkpoint": phase}), flush=True)

    try:
        if torch.cuda.device_count() != 1 or "H200" not in report["gpu"]:
            raise RuntimeError(f"Expected exactly one H200, got {report['gpu']}")
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        import triton
        import triton.language as tl
        report["triton"] = triton.__version__
        if attention_v10:
            probe = load("attention_resource_probe")
            report["attention_v10"] = {"instrumentation": {}, "compilation": {}, "resources": {},
                                       "expected_regular_checks": 128, "checks_are_official": False}
            report["attention_v10"]["candidate_revision"] = payload.get("attention_variant", "v10")
            for name in ("baseline", "candidate"):
                transformed, proof = probe.instrument_source(payload["files"][name + ".py"],
                    expected_sha256=attention_sources[name + ".py"][1], tag=f"{payload['attention_v10_tag']}_{name}")
                if proof != payload["attention_v10_instrumentation"][name]:
                    raise RuntimeError("attention V10 local/remote source transform provenance mismatch")
                (folder / (name + ".py")).write_text(transformed)
                report["attention_v10"]["instrumentation"][name] = proof
        reference, baseline, candidate = (load(name) for name in ("reference", "baseline", "candidate"))
        if attention_v10:
            # Compile without launching attention. A new barrier failure must not erase its diagnostics.
            for name, module in (("baseline", baseline), ("candidate", candidate)):
                compile_started = time.monotonic()
                try:
                    _, capture = probe.capture_compile_call(module._module)
                except probe.CompileCaptureError as exc:
                    capture = exc.capture
                    report["attention_v10"]["compilation"][name] = {
                        "capture": capture, "summary": attention_v10_compiler_summary(probe, capture),
                        "wall_seconds": time.monotonic() - compile_started, "status": "failed"}
                    checkpoint(f"{name}_compile_failed")
                    raise
                summary = attention_v10_compiler_summary(probe, capture)
                report["attention_v10"]["compilation"][name] = {
                    "capture": capture, "summary": summary,
                    "wall_seconds": time.monotonic() - compile_started, "status": "built_without_launch"}
                checkpoint(f"{name}_compiled_before_any_attention_launch")
                if summary["compile_log_gate"] is not True:
                    raise RuntimeError("attention V10 compiler evidence is incomplete or no actual ptxas output was captured")
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
            if payload["problem"] == "muse_decode_layer":
                from decoder_policy_metadata import collect_decoder_policy
                try:
                    policy = collect_decoder_policy(module, states[-1], max_nodes=12000,
                                                    max_items=128, max_depth=10)
                except Exception as exc:
                    policy = {"status": "unavailable", "error_type": type(exc).__name__}
                report.setdefault("prepared_policy", {})[name] = policy
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
            target = attention_v10 and index in (0, 1, 3, 4)
            # Test multiple fresh inputs before capturing either implementation.
            for step in range(2):
                fresh = generate(case, payload["seed"] + 1000 * index + step)
                if attention_v10:
                    row.setdefault("sequence_length_sums", []).append({"phase": "eager", "seed_step": step,
                        "past_tokens": int(fresh["seq_lens"].sum().item())})
                for name, module, state in zip(("baseline", "candidate"), (baseline, candidate), states):
                    data = clone(fresh)
                    if attention_v10 and step == 0:
                        records = []
                        checkpoint(f"case{index}_{name}_before_untimed_launch")
                        output = observe_attention_v10_call(module, lambda: call(module, data, state), records)
                        if index == 0:
                            # _launch<8> has now set this exact kernel's dynamic-shared attribute.
                            report["attention_v10"]["resources"][name] = probe.collect_attention_resources(module)
                        resources = report["attention_v10"]["resources"][name]
                        dispatch = attention_v10_dispatch(records, resources, target=target)
                        row.setdefault("attention_dispatch", {})[name] = dispatch
                        checkpoint(f"case{index}_{name}_resources_and_dispatch_before_sync")
                        if index == 0:
                            require_attention_v10_resources(resources, report["attention_v10"]["compilation"][name]["summary"])
                        if target and not dispatch["activation_gate"]:
                            raise RuntimeError(f"{name} attention V10 target did not activate the exact flat cooperative resident plan")
                    else:
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
                if attention_v10:
                    row["sequence_length_sums"].append({"phase": "graph", "seed_step": repeat,
                        "past_tokens": int(fresh["seq_lens"].sum().item())})
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
            if attention_v10:
                checkpoint(f"case{index}_complete")
        if attention_v10:
            report["attention_v10"]["regular_checks"] = sum(len(row["checks"]) for row in report["cases"])
            if report["attention_v10"]["regular_checks"] != 128:
                raise RuntimeError("attention V10 did not complete exactly 128 oracle checks")
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


def run_hardware_worker(payload: dict) -> None:
    """Stage only the reviewed diagnostic payload, then run its fixed entrypoint."""
    import importlib.util
    expected = {"baseline.py", "reference.py", "checks.py", "nvfp4.py", "hardware_metrics.py"}
    try:
        if set(payload.get("files", {})) != expected or set(payload.get("file_sha256", {})) != expected:
            raise ValueError("Unexpected hardware payload file set")
        folder = Path("/tmp/okc-input")
        folder.mkdir(exist_ok=True)
        for name, source in payload["files"].items():
            if not isinstance(source, str) or sha(source) != payload["file_sha256"][name]:
                raise ValueError(f"Hardware payload hash mismatch: {name}")
            (folder / name).write_text(source)
        spec = importlib.util.spec_from_file_location("hardware_metrics", folder / "hardware_metrics.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run_profile(payload)
    except Exception as exc:
        save(Path("/tmp/okc-result.json"), {"status": "hardware_unavailable", "official": False,
             "diagnostic_only": True, "error": f"{type(exc).__name__}: {exc}"})
        raise


def run_compile_worker(payload: dict) -> None:
    """Only the fixed CPU compile helper and immutable parent enter this sandbox."""
    import importlib.util
    expected = {"baseline.py", "compile_sass.py", "hardware_metrics.py"}
    if set(payload.get("files", {})) != expected or set(payload.get("file_sha256", {})) != expected:
        raise ValueError("Unexpected compile payload files")
    folder = Path("/tmp/okc-input")
    folder.mkdir(exist_ok=True)
    for name, source in payload["files"].items():
        if type(source) is not str or sha(source) != payload["file_sha256"][name]:
            raise ValueError("Compile payload hash mismatch")
        (folder / name).write_text(source)
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("compile_sass", folder / "compile_sass.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(payload)


def finalizer_observed_floor(root: Path) -> dict:
    """Bind a saved higher provider observation; this only increases the debit."""
    from modal_credit_budget import provider_spend
    relative = "experiments/modal/20261104T191321Z-25f35b45/billing-after.json"
    expected = "4626955e09927399d7f10ba88d1f1c524d1aec71f52680a3a5ee3b35db05e4f2"
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Saved provider high-water evidence changed")
    amount = provider_spend(json.loads(raw)["billing"])
    if str(amount) != "4.21235928":
        raise ValueError("Saved provider high-water mismatch")
    return {"path": relative, "sha256": expected, "usd": str(amount)}


def run_finalizer_validation_worker(payload: dict, *, stages=False, coverage=False) -> None:
    """Import exactly the reviewed validation helper, never an ambient module."""
    import importlib.util
    if stages and coverage:
        raise ValueError("Ambiguous finalizer worker mode")
    module_name = "finalizer_coverage" if coverage else "finalizer_stages" if stages else "finalizer_validation"
    expected = ({"candidate.py", "baseline.py", "validation_reference.py", module_name + ".py"}
                if stages or coverage else {"candidate.py", "baseline.py", "finalizer_validation.py"})
    if set(payload.get("files", {})) != expected or set(payload.get("file_sha256", {})) != expected:
        raise ValueError("Unexpected finalizer validation file set")
    folder = Path("/tmp/okc-input")
    folder.mkdir(exist_ok=True)
    for name, source in payload["files"].items():
        if type(source) is not str or sha(source) != payload["file_sha256"][name]:
            raise ValueError("Finalizer validation payload hash mismatch")
        (folder / name).write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, folder / (module_name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_payload(payload)
    module.run(payload)


def run_router_local_ids_worker(payload: dict) -> None:
    """Load the frozen router experiment driver; it validates its full package."""
    import importlib.util
    name = "router_local_ids.py"
    source = payload.get("files", {}).get(name)
    if type(source) is not str or sha(source) != payload.get("file_sha256", {}).get(name):
        raise ValueError("Router worker source/hash mismatch")
    folder = Path("/tmp/okc-input")
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(source)
    spec = importlib.util.spec_from_file_location("router_local_ids", folder / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_payload(payload)
    module.run(payload)


FINALIZER_READ_RESULT = """import sys
with open("/tmp/okc-result.json", "rb") as stream:
    data = stream.read(1048577)
if len(data) > 1048576:
    raise ValueError("finalizer validation result exceeds 1MiB")
sys.stdout.buffer.write(data)
"""


def retrieve_finalizer_validation_result(sandbox, destination: Path, payload: dict, *, recovery=False) -> dict:
    """One fixed-path, remotely capped read. Recovery is never a successful run."""
    process = sandbox.exec("python3", "-c", FINALIZER_READ_RESULT, timeout=10, text=False)
    content = process.stdout.read()
    process.stderr.read()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Finalizer result read exited {process.returncode}")
    if type(content) is not bytes or len(content) > 1024 * 1024:
        raise ValueError("Finalizer result exceeds local byte/type bound")
    result = json.loads(content)
    expected = {"candidate_sha256": payload["source_sha256"],
                "baseline_sha256": payload["baseline_sha256"],
                "worker_sha256": payload["file_sha256"]["finalizer_validation.py"],
                "case_manifest_sha256": payload["case_manifest_sha256"],
                "file_sha256": payload["file_sha256"], "image": payload["image"]}
    if (type(result) is not dict or any(result.get(k) != v for k, v in expected.items())
            or result.get("official") is not False or result.get("diagnostic_only") is not True
            or result.get("status") not in ("running", "incomplete", "failed", "pass")):
        raise ValueError("Finalizer result provenance/status mismatch")
    if result["status"] == "pass":
        import serving_finalizer_validation_worker as finalizer
        if sha(Path(finalizer.__file__).read_text()) != expected["worker_sha256"]:
            raise ValueError("Local finalizer manifest implementation changed")
        cases = finalizer.cases()
        rows = result.get("cases", [])
        if (result.get("planned_cases") != len(cases) or result.get("completed_cases") != len(cases)
                or len(rows) != len(cases) or any(
                    row.get("descriptor") != case or row.get("name") != case["name"]
                    or row.get("seed") != case["seed"] or row.get("pass") is not True
                    for row, case in zip(rows, cases))):
            raise ValueError("Finalizer passing result has incomplete case coverage")
        expected_profiles = {"boundary31": "fast", "boundary32": "large", "boundary33": "large",
                             "fallback_fp32": "eager", "fallback_topk33": "eager", "fallback_hidden16385": "eager"}
        profiles = result.get("profiles", [])
        if (len(profiles) != len(expected_profiles)
                or {p.get("name"): p.get("kind") for p in profiles} != expected_profiles
                or any(p.get("pass") is not True or p.get("cuda_positive_duration_events_only") is not True
                       or p.get("gpu_event_count", 0) <= 0 for p in profiles)):
            raise ValueError("Finalizer passing result has incomplete activation coverage")
        streams, graphs = result.get("two_streams", {}), result.get("two_graphs", {})
        if (streams.get("pass") is not True or streams.get("calls") != 8
                or graphs.get("pass") is not True or graphs.get("graphs") != 2
                or graphs.get("fresh_replays") != 8):
            raise ValueError("Finalizer passing result has incomplete stream/graph coverage")
    local = "result-checkpoint.json" if recovery or result["status"] != "pass" else "result.json"
    save(destination / local, result)
    return result


def retrieve_finalizer_stage_result(sandbox, destination: Path, payload: dict, *, recovery=False) -> dict:
    """Stage diagnostics use the same fixed remote read bound and no PASS status."""
    import serving_finalizer_stage_worker as stages
    if sha(Path(stages.__file__).read_text()) != payload["file_sha256"]["finalizer_stages.py"]:
        raise ValueError("Local stage worker changed after upload")
    process = sandbox.exec("python3", "-c", FINALIZER_READ_RESULT, timeout=10, text=False)
    content = process.stdout.read()
    process.stderr.read()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Stage result read exited {process.returncode}")
    if type(content) is not bytes or len(content) > 1024 * 1024:
        raise ValueError("Stage result exceeds local byte/type bound")
    result = json.loads(content)
    stages.validate_result(result, payload)
    local = ("result.json" if not recovery and result["status"] == "diagnostic_complete"
             else "result-checkpoint.json")
    save(destination / local, result)
    return result


def is_finalizer_payload(payload) -> bool:
    return bool(payload and (payload.get("serving_finalizer_validation") is True
                             or payload.get("serving_finalizer_stages") is True
                             or payload.get("serving_finalizer_coverage") is True))


def retrieve_finalizer_coverage_result(sandbox, destination: Path, payload: dict, *, recovery=False) -> dict:
    """Coverage completion retains the known generic failure and never means PASS."""
    import serving_finalizer_coverage_worker as coverage
    if sha(Path(coverage.__file__).read_text()) != payload["file_sha256"]["finalizer_coverage.py"]:
        raise ValueError("Local coverage worker changed after upload")
    process = sandbox.exec("python3", "-c", FINALIZER_READ_RESULT, timeout=10, text=False)
    content = process.stdout.read()
    process.stderr.read()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Coverage result read exited {process.returncode}")
    if type(content) is not bytes or len(content) > 1024 * 1024:
        raise ValueError("Coverage result exceeds local byte/type bound")
    result = json.loads(content)
    coverage.validate_result(result, payload)
    local = ("result.json" if not recovery and result["status"] == "coverage_complete"
             else "result-checkpoint.json")
    save(destination / local, result)
    return result


def is_component_payload(payload) -> bool:
    return bool(is_finalizer_payload(payload) or (payload and payload.get("serving_router_local_ids") is True))


def retrieve_router_local_ids_result(sandbox, destination: Path, payload: dict, *, recovery=False) -> dict:
    """Bounded router component evidence; completion does not imply a speed gain."""
    import serving_router_local_ids_worker as router
    if sha(Path(router.__file__).read_text()) != payload["file_sha256"]["router_local_ids.py"]:
        raise ValueError("Local router worker changed after upload")
    process = sandbox.exec("python3", "-c", FINALIZER_READ_RESULT, timeout=10, text=False)
    content = process.stdout.read()
    process.stderr.read()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Router result read exited {process.returncode}")
    if type(content) is not bytes or len(content) > 1024 * 1024:
        raise ValueError("Router result exceeds local byte/type bound")
    result = json.loads(content)
    router.validate_result(result, payload)
    local = ("result.json" if not recovery and result["status"] == "screen_complete"
             else "result-checkpoint.json")
    save(destination / local, result)
    return result


@contextmanager
def finalizer_live_recovery(get_sandbox, destination: Path, manifest: dict, payload: dict):
    """Exit before app.run: that context may destroy the sandbox on failure."""
    try:
        yield
    except BaseException:
        sandbox = get_sandbox()
        if sandbox is not None:
            manifest["validation_recovery_inside_app_context"] = True
            try:
                if not (destination / "result.json").exists() and not (destination / "result-checkpoint.json").exists():
                    retrieve = (retrieve_router_local_ids_result if payload.get("serving_router_local_ids") is True
                                else retrieve_finalizer_coverage_result if payload.get("serving_finalizer_coverage") is True
                                else retrieve_finalizer_stage_result if payload.get("serving_finalizer_stages") is True
                                else retrieve_finalizer_validation_result)
                    retrieve(sandbox, destination, payload, recovery=True)
                manifest["validation_checkpoint_recovered"] = True
            except Exception as recovery_error:
                manifest["validation_checkpoint_recovered"] = False
                manifest["validation_retrieval_error_type"] = type(recovery_error).__name__
        raise  # Preserve the original worker/validation failure.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "benchmark", "hardware-profile", "compile-sass", "attention-v10-screen", "attention-v11-screen", "serving-finalizer-validation", "serving-finalizer-stages", "serving-finalizer-coverage", "serving-router-local-ids", "worker"))
    parser.add_argument("--problem", choices=PROBLEMS)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--incumbent", type=Path, help="Optional paired comparison against this same-problem source")
    parser.add_argument("--profile", action="store_true", help="Record CUDA kernel names from separate diagnostic graph replays")
    parser.add_argument("--resources", action="store_true", help="Record cuobjdump resources from module-owned _EXTS shared objects")
    parser.add_argument("--hardware-issue", action="store_true", help="Query the fixed issue/stall and backpressure groups; include each complete available group")
    parser.add_argument("--cases", type=Path, help="JSON list of explicit reference.generate_input kwargs")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weight-seed", type=int, default=7311)
    parser.add_argument("--timeout", type=int, default=300, help="Benchmark sandbox lifetime, 60..900 seconds")
    parser.add_argument("--payload", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print manifest; no Modal call")
    args = parser.parse_args()
    if args.mode == "worker":
        payload = json.loads(Path(args.payload).read_text())
        if payload.get("serving_router_local_ids") is True:
            run_router_local_ids_worker(payload)
        elif payload.get("serving_finalizer_coverage") is True:
            run_finalizer_validation_worker(payload, coverage=True)
        elif payload.get("serving_finalizer_stages") is True:
            run_finalizer_validation_worker(payload, stages=True)
        elif payload.get("serving_finalizer_validation") is True:
            run_finalizer_validation_worker(payload)
        elif payload.get("compile_sass") is True:
            run_compile_worker(payload)
        elif payload.get("hardware_profile") is True:
            run_hardware_worker(payload)
        else:
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
    if args.hardware_issue and args.mode != "hardware-profile":
        parser.error("--hardware-issue requires hardware-profile mode")
    payload = None
    compile_mode = args.mode == "compile-sass"
    if args.mode == "serving-router-local-ids":
        from serving_router_local_ids_worker import build_payload
        if (args.timeout != 600 or args.problem is not None or args.candidate or args.cases
                or args.seed is not None or args.weight_seed != 7311 or args.repeats != 6):
            parser.error("serving-router-local-ids has fixed sources/cases and requires --timeout 600")
        payload = build_payload(ROOT, IMAGE)
        payload["provider_observed_floor_evidence"] = finalizer_observed_floor(ROOT)
        for name, source in payload["files"].items():
            if name.endswith(".py"):
                compile(source, name, "exec")
    if args.mode in ("serving-finalizer-validation", "serving-finalizer-stages", "serving-finalizer-coverage"):
        if args.mode == "serving-finalizer-coverage":
            from serving_finalizer_coverage_worker import build_payload
            fixed_timeout = 300
        elif args.mode == "serving-finalizer-stages":
            from serving_finalizer_stage_worker import build_payload
            fixed_timeout = 300
        else:
            from serving_finalizer_validation_worker import build_payload
            fixed_timeout = 600
        if (args.timeout != fixed_timeout or args.problem is not None or args.candidate or args.cases
                or args.seed is not None or args.weight_seed != 7311 or args.repeats != 6):
            parser.error(f"{args.mode} has fixed sources/cases and requires --timeout {fixed_timeout}")
        payload = build_payload(ROOT, IMAGE)
        payload["provider_observed_floor_evidence"] = finalizer_observed_floor(ROOT)
        for name, source in payload["files"].items():
            compile(source, name, "exec")
    if compile_mode:
        from modal_compile_sass import SOURCE_SHA, validate_payload
        if (args.timeout != 600 or args.problem is not None or args.candidate or args.cases
                or args.seed is not None or args.weight_seed != 7311 or args.repeats != 6):
            parser.error("compile-sass has fixed source/CPU resources and requires --timeout 600")
        source_path = ROOT / "experiments/candidates/muse_moe_w4a16" / f"{SOURCE_SHA}.py"
        files = {"baseline.py": source_path.read_text(),
                 "compile_sass.py": (ROOT / "tools/modal_compile_sass.py").read_text(),
                 "hardware_metrics.py": (ROOT / "tools/modal_hardware_metrics.py").read_text()}
        payload = {"compile_sass": True, "gpu": None, "resource_kind": "cpu_4cpu_32gib",
                   "timeout_seconds": 600, "compile_budget_seconds": 560, "image": IMAGE,
                   "source_sha256": SOURCE_SHA, "files": files,
                   "file_sha256": {name: sha(text) for name, text in files.items()}}
        validate_payload(payload)
        for name, source in files.items(): compile(source, name, "exec")
    if args.mode in ("attention-v10-screen", "attention-v11-screen"):
        if (args.problem is not None or args.candidate or args.cases or args.seed is not None
                or args.weight_seed != 7311 or args.repeats != 6):
            parser.error("attention screen has fixed sources, eight cases, seed121021 and six pairs")
        payload = build_attention_v10_payload(ROOT, uuid.uuid4().hex[:16],
            variant="v11" if args.mode == "attention-v11-screen" else "v10")
    if args.mode == "hardware-profile":
        from modal_hardware_metrics import CASES, FILES, POLICY, PROFILE_SEEDS, SOURCE_SHA, REQUIRED, ISSUE, BACKPRESSURE, validate_payload
        if (args.problem not in (None, "muse_moe_w4a16") or args.candidate or args.cases
                or args.seed is not None or args.weight_seed != 7311 or args.repeats > 6):
            parser.error("hardware-profile has fixed source/cases/seeds; repeats must be 2..6")
        source_path = ROOT / "experiments/candidates/muse_moe_w4a16" / f"{SOURCE_SHA}.py"
        files = {"baseline.py": source_path.read_text(),
                 "hardware_metrics.py": (ROOT / "tools/modal_hardware_metrics.py").read_text()}
        for name in sorted(FILES - files.keys()):
            files[name] = (ROOT / "muse_moe_w4a16" / name).read_text()
        payload = {"problem": "muse_moe_w4a16", "files": files, "image": IMAGE,
                   "hardware_profile": True, "hardware_issue": args.hardware_issue,
                   "hardware_budget_seconds": args.timeout - 40, "hardware_policy": POLICY,
                   "hardware_required_metrics": REQUIRED, "hardware_optional_metrics": ISSUE + BACKPRESSURE if args.hardware_issue else (),
                   "source_sha256": SOURCE_SHA, "file_sha256": {k: sha(v) for k, v in files.items()},
                   "cases": list(CASES), "repeats": args.repeats, "seed": 7202, "weight_seed": 7311,
                   "hardware_input_seeds": list(PROFILE_SEEDS), "comparison_baseline": "single_frozen_incumbent",
                   "incumbent": {"path": str(source_path.relative_to(ROOT)), "sha256": SOURCE_SHA}}
        validate_payload(payload)
        for name, source in files.items():
            compile(source, name, "exec")
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
        if args.problem == "muse_decode_layer":
            files["decoder_policy_metadata.py"] = (ROOT / "tools/decoder_policy_metadata.py").read_text()
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
                   "cases": cases, "repeats": args.repeats, "seed": args.seed if args.seed is not None else 7301, "weight_seed": args.weight_seed,
                   "comparison_baseline": "incumbent" if incumbent else "official_public_baseline",
                   "incumbent": incumbent, "profile": args.profile, "resources": args.resources}
    worker_source = Path(__file__).read_text()
    manifest = {"mode": args.mode, "gpu": None if compile_mode else "H200:1", "timeout_seconds": 120 if args.mode == "smoke" else args.timeout,
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
    from modal_credit_budget import allocation_guard, RESOURCE_CPU
    guard_kwargs = {"resource_kind": RESOURCE_CPU} if compile_mode else {}
    if is_component_payload(payload):
        guard_kwargs["observed_spend_floor"] = payload["provider_observed_floor_evidence"]["usd"]
    with allocation_guard(ROOT, stamp, manifest["timeout_seconds"], **guard_kwargs) as credit:
        if compile_mode and (credit.get("resource_kind") != RESOURCE_CPU or manifest["gpu"] is not None):
            raise RuntimeError("CPU compile resource/credit kind mismatch")
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
            if payload.get("attention_v10_screen") is True:
                from attention_resource_probe import instrument_source
                attention_sources = attention_screen_sources(payload.get("attention_variant", "v10"))
                for name in ("baseline", "candidate"):
                    transformed, proof = instrument_source(payload["files"][name + ".py"],
                        expected_sha256=attention_sources[name + ".py"][1], tag=f"{payload['attention_v10_tag']}_{name}")
                    if proof != payload["attention_v10_instrumentation"][name]:
                        raise RuntimeError("attention V10 frozen instrumented source changed")
                    (destination / f"instrumented_{name}.py").write_text(transformed)
            save(destination / "request.json", {k: v for k, v in payload.items() if k != "files"})
            (destination / "worker.py").write_text(worker_source)
        image = modal.Image.debian_slim(python_version="3.12") if args.mode == "smoke" else modal.Image.from_registry(IMAGE)
        app = modal.App("okc-h200-bounded")
        sandbox = None
        try:
            recovery_scope = (finalizer_live_recovery(lambda: sandbox, destination, manifest, payload)
                              if is_component_payload(payload) else nullcontext())
            # Contexts exit in reverse order: recover while the app still owns
            # a live sandbox, then allow app.run to perform its cleanup.
            with modal.enable_output(), app.run(environment_name=credit["environment"]), recovery_scope:
                sandbox = modal.Sandbox.create("sleep", str(manifest["timeout_seconds"]), app=app, image=image,
                                              gpu=None if compile_mode else "H200", cpu=(4, 4), memory=(32768, 32768), block_network=True,
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
                if (compile_mode or is_component_payload(payload)) and command.returncode:
                    # The worker may fail before creating any result checkpoint.
                    # Preserve its actual exit/logs; recovery failures are separate.
                    manifest["remote_failure"] = {"exit_code": command.returncode, "stderr_log": "stderr.log"}
                    if is_component_payload(payload):
                        save(destination / "manifest.json", manifest)
                    print(stderr[-6000:], file=sys.stderr)
                    raise RuntimeError(f"Remote process exited {command.returncode}; see {destination}")
                if payload:
                    if compile_mode:
                        from modal_compile_sass import retrieve
                        result = retrieve(sandbox, destination)
                        manifest["compile_artifacts_retrieved"] = True
                    elif payload.get("serving_finalizer_validation") is True:
                        result = retrieve_finalizer_validation_result(sandbox, destination, payload)
                        manifest["validation_result_retrieved"] = True
                    elif payload.get("serving_finalizer_stages") is True:
                        result = retrieve_finalizer_stage_result(sandbox, destination, payload)
                        manifest["validation_result_retrieved"] = True
                    elif payload.get("serving_finalizer_coverage") is True:
                        result = retrieve_finalizer_coverage_result(sandbox, destination, payload)
                        manifest["validation_result_retrieved"] = True
                    elif payload.get("serving_router_local_ids") is True:
                        result = retrieve_router_local_ids_result(sandbox, destination, payload)
                        manifest["validation_result_retrieved"] = True
                    else:
                        result = json.loads(sandbox.filesystem.read_text("/tmp/okc-result.json"))
                        save(destination / "result.json", result)
                    if payload.get("attention_v10_screen") is True:
                        save_attention_v10_logs(destination, result)
                    manifest["status"] = result["status"]
                    if payload.get("serving_finalizer_validation") is True and result["status"] != "pass":
                        raise RuntimeError("Finalizer validation incomplete or failed; no automatic retry")
                    if payload.get("serving_finalizer_stages") is True and result["status"] != "diagnostic_complete":
                        raise RuntimeError("Finalizer stage diagnostic incomplete or failed; no automatic retry")
                    if payload.get("serving_finalizer_coverage") is True and result["status"] != "coverage_complete":
                        raise RuntimeError("Finalizer coverage incomplete or failed; no automatic retry")
                    if payload.get("serving_router_local_ids") is True and result["status"] != "screen_complete":
                        raise RuntimeError("Router component screen incomplete or failed; no automatic retry")
                    if compile_mode and result["status"] != "compile_complete":
                        raise RuntimeError("CPU compile diagnostic did not complete; no retry")
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
            if payload and payload.get("attention_v10_screen") is True and sandbox is not None:
                recover_attention_v10_checkpoint(sandbox, destination, manifest)
            if payload and payload.get("hardware_profile") is True and sandbox is not None:
                recover_hardware_checkpoints(sandbox, destination, manifest)
            if compile_mode and sandbox is not None and not manifest.get("compile_artifacts_retrieved"):
                try:
                    from modal_compile_sass import retrieve
                    retrieve(sandbox, destination, recovery=True)
                    manifest["compile_checkpoint_recovered"] = True
                except Exception as recovery_error:
                    manifest["compile_checkpoint_recovered"] = False
                    manifest["compile_retrieval_error_type"] = type(recovery_error).__name__
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
