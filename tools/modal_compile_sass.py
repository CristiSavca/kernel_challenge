"""Fixed CPU-only compilation of the current MoE MAIN translation unit. No execution."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch

from modal_hardware_metrics import run_bounded, command_ok, write_json

SOURCE_SHA = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"
NATIVE_SHA = "85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c"
CPP_SHA = "0214aeb6bfbce490dce0936ae55f2fe5f8dab38a8088655701bf5f01cf0aff75"
IMAGE = "lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af"
FILES = frozenset({"baseline.py", "compile_sass.py", "hardware_metrics.py"})
DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_PRMTSC=1")
EXT_DEFINES = DEFINES + ("-DOKC_MULDEC=1", "-DOKC_DQ7=1")
CUDA_FLAGS = ("-O3", "--use_fast_math", "-Xptxas", "-v,-O3", "-gencode=arch=compute_90a,code=sm_90a",
              "-lineinfo", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3", *EXT_DEFINES)
EXT_NAME = "okc_moe_kperm3_v9_" + hashlib.sha1("_".join(EXT_DEFINES).encode()).hexdigest()[:16] + "_sm90a"
FOLDER = Path("/tmp/okc-input")
BUILD = Path("/tmp/okc-compile-build")
ARTIFACTS = Path("/tmp/okc-compile-artifacts")
RESULT = Path("/tmp/okc-result.json")
CHILD = Path("/tmp/okc-compile-child.json")
BUNDLE = Path("/tmp/okc-compile-artifacts.tar.gz")
MIB = 1024 * 1024
LIMITS = {"compile.stdout.log": 8*MIB, "compile.stderr.log": 8*MIB, "build.ninja": MIB,
          "cuda.cu": 2*MIB, "main.cpp": MIB, "owned.so": 64*MIB,
          "resources.txt": MIB, "sass.txt": 32*MIB, "child.json": MIB, "owned.cubin": 64*MIB}
BUNDLE_LIMIT = 96*MIB
READ_CAPPED = """import sys
limits = {"/tmp/okc-result.json": 2097152, "/tmp/okc-compile-artifacts.tar.gz": 100663296}
limit = limits[sys.argv[1]]
with open(sys.argv[1], "rb") as stream:
    data = stream.read(limit + 1)
if len(data) > limit: raise ValueError("compile retrieval size limit")
sys.stdout.buffer.write(data)
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_strings(source: str) -> dict:
    return {t.id: ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and t.id in ("CUDA_SRC", "CPP_SRC")}


def require_both_big(text: str) -> None:
    if set(re.findall(r"\b_ZN3okc9moe_fusedILb([01])EEEvNS_6ParamsE\b", text)) != {"0", "1"}:
        raise ValueError("Missing BIG0/BIG1 compiled symbols")


def validate_payload(p: dict) -> None:
    if (p.get("compile_sass") is not True or p.get("gpu") is not None
            or p.get("resource_kind") != "cpu_4cpu_32gib" or p.get("image") != IMAGE
            or p.get("compile_budget_seconds") != 560 or p.get("timeout_seconds") != 600):
        raise ValueError("Compile mode is fixed CPU-only 4CPU/32GiB/600s")
    if set(p.get("files", {})) != FILES or set(p.get("file_sha256", {})) != FILES:
        raise ValueError("Unexpected compile payload files")
    for name, value in p["files"].items():
        if type(value) is not str or sha(value.encode()) != p["file_sha256"][name]:
            raise ValueError("Compile payload hash mismatch")
    if p.get("source_sha256") != SOURCE_SHA or sha(p["files"]["baseline.py"].encode()) != SOURCE_SHA:
        raise ValueError("Compile requires frozen247888")
    strings = source_strings(p["files"]["baseline.py"])
    if sha(strings["CUDA_SRC"].encode()) != NATIVE_SHA or sha(strings["CPP_SRC"].encode()) != CPP_SHA:
        raise ValueError("Translation-unit identity mismatch")


def compile_loaded(module, torch, cpp) -> dict:
    """Call only unchanged _ext; guard Python device/tensor APIs, keep build argv intact."""
    if module.PDL is not False or tuple(module.MAIN_DEFINES) != DEFINES[:-1] or module._EXTS:
        raise ValueError("Unexpected fixed parent state")
    observed = {"capability_calls": 0, "load_inline_calls": 0, "ninja_build_calls": 0, "pdl_values": []}
    original_load, original_run = module.load_inline, subprocess.run
    setters = []
    def forbidden(*args, **kwargs):
        raise RuntimeError("Tensor/device execution is forbidden in compile-only mode")
    def capability(*args, **kwargs):
        observed["capability_calls"] += 1
        return (9, 0)
    def capture_run(*args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        is_build = isinstance(command, (list, tuple)) and Path(str(command[0])).name == "ninja" and "-v" in command
        if is_build:
            observed["ninja_build_calls"] += 1
            if observed["ninja_build_calls"] != 1:
                raise RuntimeError("Refusing a second ninja build")
        try:
            completed = original_run(*args, **kwargs)
        except subprocess.CalledProcessError as exc:
            if is_build and exc.output:
                print(exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else exc.output, flush=True)
            raise
        if is_build and completed.stdout:
            print(completed.stdout.decode(errors="replace") if isinstance(completed.stdout, bytes) else completed.stdout, flush=True)
        return completed
    def load(**kwargs):
        observed["load_inline_calls"] += 1
        if (observed["load_inline_calls"] != 1 or kwargs.get("name") != EXT_NAME
                or kwargs.get("cuda_sources") != module.CUDA_SRC or kwargs.get("cpp_sources") != module.CPP_SRC
                or tuple(kwargs.get("extra_cuda_cflags", ())) != CUDA_FLAGS
                or kwargs.get("functions") != ["okc_run", "okc_set_pdl"] or kwargs.get("verbose") is not False):
            raise ValueError("Unchanged _ext build contract mismatch")
        extension = original_load(**kwargs)
        setter = extension.okc_set_pdl
        setters.append((extension, setter))
        def pdl(on):
            if on != 0: raise ValueError("PDL must remain disabled")
            observed["pdl_values"].append(on)
            return setter(on)  # exact source: assignment to host g_okc_pdl only
        extension.okc_set_pdl = pdl
        return extension
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda, "get_device_capability", capability))
            stack.enter_context(patch.object(torch.cuda, "is_available", lambda: False))
            stack.enter_context(patch.object(torch.cuda, "device_count", lambda: 0))
            for name in ("_lazy_init", "init", "current_device", "set_device", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, forbidden))
            for name in ("empty", "empty_like", "zeros", "ones", "full", "tensor", "as_tensor", "arange", "rand", "randn", "randint"):
                stack.enter_context(patch.object(torch, name, forbidden))
            stack.enter_context(patch.object(module, "prepare", forbidden))
            stack.enter_context(patch.object(module, "kernel_fn", forbidden))
            stack.enter_context(patch.object(module, "load_inline", load))
            stack.enter_context(patch.object(subprocess, "run", capture_run))
            extension = module._ext(DEFINES, use_mul=True)
        if (observed["capability_calls"] != 2 or observed["load_inline_calls"] != 1
                or observed["ninja_build_calls"] != 1 or observed["pdl_values"] != [0]
                or list(module._EXTS) != [EXT_DEFINES] or torch.cuda.is_initialized()):
            raise RuntimeError("Compile-only execution/ownership invariant failed")
        observed.update(owned_so=str(Path(extension.__file__).resolve()), cuda_initialized=False,
                        extension_name=EXT_NAME, defines=list(EXT_DEFINES), cuda_flags=list(CUDA_FLAGS))
        return observed
    finally:
        for extension, setter in setters:
            extension.okc_set_pdl = setter


def child(p: dict) -> None:
    validate_payload(p)
    report = {"status": "compile_child_running", "source_sha256": SOURCE_SHA, "native_source_sha256": NATIVE_SHA}
    write_json(CHILD, report)
    try:
        if BUILD.exists(): raise RuntimeError("Fresh compile directory already exists")
        BUILD.mkdir()
        os.environ.update(TORCH_EXTENSIONS_DIR=str(BUILD), MAX_JOBS="4", CUDA_VISIBLE_DEVICES="")
        import torch
        if torch.cuda.is_initialized(): raise RuntimeError("CUDA already initialized")
        # cpp_extension import may call is_available while locating CUDA_HOME.
        with patch.object(torch.cuda, "is_available", lambda: False), \
             patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA init forbidden")):
            import torch.utils.cpp_extension as cpp
            spec = importlib.util.spec_from_file_location("frozen_moe_compile", FOLDER / "baseline.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_home=cpp.CUDA_HOME)
        report.update(compile_loaded(module, torch, cpp))
        report["status"] = "compile_child_complete"
    except Exception as exc:
        report.update(status="compile_child_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(CHILD, report)


def bundle_artifacts() -> dict:
    files = {}
    for path in sorted(ARTIFACTS.iterdir()):
        if path.name not in LIMITS or not path.is_file() or path.is_symlink() or path.stat().st_size > LIMITS[path.name]:
            raise ValueError("Unexpected/oversized compile artifact")
        data = path.read_bytes()
        files[path.name] = {"bytes": len(data), "sha256": sha(data)}
    pending = BUNDLE.with_suffix(".pending")
    with tarfile.open(pending, "w:gz") as archive:
        for name in files: archive.add(ARTIFACTS / name, arcname=name, recursive=False)
    if pending.stat().st_size > BUNDLE_LIMIT: raise ValueError("Compile archive exceeds bound")
    pending.replace(BUNDLE)
    data = BUNDLE.read_bytes()
    return {"path": str(BUNDLE), "bytes": len(data), "sha256": sha(data), "files": files}


def run(p: dict) -> None:
    validate_payload(p)
    start = time.monotonic()
    deadline = start + p["compile_budget_seconds"]
    report = {"status": "compile_running", "official": False, "diagnostic_only": True,
              "source_sha256": SOURCE_SHA, "native_source_sha256": NATIVE_SHA, "cpp_source_sha256": CPP_SHA,
              "gpu": None, "device_execution": False, "versions": {}, "commands": {}}
    def checkpoint():
        report["elapsed_seconds"] = time.monotonic() - start
        write_json(RESULT, report)
    def command(name, argv, cap, limit=None):
        remaining = deadline - time.monotonic() - 5
        if remaining <= 0: raise TimeoutError("Compile diagnostic deadline exhausted")
        result = run_bounded(argv, min(remaining, limit or remaining), cap)
        report["commands"][name] = {k: v for k, v in result.items() if k not in ("stdout", "stderr")}
        return result
    try:
        ARTIFACTS.mkdir(exist_ok=False)
        checkpoint()
        for name, argv in (("nvcc", ["/usr/local/cuda/bin/nvcc", "--version"]),
                           ("ptxas", ["/usr/local/cuda/bin/ptxas", "--version"]),
                           ("cuobjdump", ["/usr/local/cuda/bin/cuobjdump", "--version"]),
                           ("nvdisasm", ["/usr/local/cuda/bin/nvdisasm", "--version"]),
                           ("cxx", [shutil.which("c++") or "/usr/bin/c++", "--version"]),
                           ("ninja", [shutil.which("ninja") or "ninja", "--version"])):
            result = command(name, argv, 65536, 15)
            report["versions"][name] = {"stdout": result["stdout"], "stderr": result["stderr"]}
            checkpoint()
            if not command_ok(result): raise RuntimeError(name + " version unavailable")
        help_result = command("nvdisasm_help", ["/usr/local/cuda/bin/nvdisasm", "--help"], 131072, 15)
        report["nvdisasm_help"] = help_result["stdout"] + help_result["stderr"]
        if not command_ok(help_result) or "--print-line-info-inline" not in report["nvdisasm_help"]:
            raise RuntimeError("Required annotated SASS option unavailable")
        result = command("compile", [sys.executable, str(FOLDER / "compile_sass.py"), "--child", "/tmp/okc-payload.json"], 8*MIB, 400)
        for stream in ("stdout", "stderr"):
            (ARTIFACTS / f"compile.{stream}.log").write_text(result[stream])
        if CHILD.exists():
            if CHILD.stat().st_size > MIB: raise ValueError("Child checkpoint oversized")
            report["child"] = json.loads(CHILD.read_text())
            shutil.copyfile(CHILD, ARTIFACTS / "child.json")
        build = BUILD / EXT_NAME
        for name in ("build.ninja", "cuda.cu", "main.cpp"):
            path = build / name
            if path.is_file():
                if path.stat().st_size > LIMITS[name]: raise ValueError("Generated source oversized")
                shutil.copyfile(path, ARTIFACTS / name)
        checkpoint()
        if not command_ok(result): raise RuntimeError("Single compile failed/timed out/output exceeded; no retry")
        c = report.get("child", {})
        if c.get("status") != "compile_child_complete" or c.get("source_sha256") != SOURCE_SHA or c.get("native_source_sha256") != NATIVE_SHA:
            raise RuntimeError("Compile child incomplete or wrong source")
        owned = Path(c["owned_so"])
        if owned.parent != build or owned.name != EXT_NAME + ".so" or owned.is_symlink() or not owned.is_file() or owned.stat().st_size > LIMITS["owned.so"]:
            raise ValueError("Owned extension identity/size mismatch")
        strings = source_strings(p["files"]["baseline.py"])
        if strings["CUDA_SRC"] not in (ARTIFACTS / "cuda.cu").read_text() or strings["CPP_SRC"] not in (ARTIFACTS / "main.cpp").read_text():
            raise ValueError("Generated translation units do not contain exact source")
        shutil.copyfile(owned, ARTIFACTS / "owned.so")
        dumped = command("resources.txt", ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(owned)], MIB, 30)
        resource_text = dumped["stdout"] + dumped["stderr"]
        (ARTIFACTS / "resources.txt").write_text(resource_text)
        if not command_ok(dumped): raise RuntimeError("Resource dump incomplete; no retry")
        require_both_big(resource_text)
        extract = BUILD / "extracted"
        extract.mkdir()
        extracted = command("extract_cubin", [sys.executable, "-c",
            "import os,sys; os.chdir(sys.argv[1]); os.execv(sys.argv[2], sys.argv[2:])",
            str(extract), "/usr/local/cuda/bin/cuobjdump", "--extract-elf", "all", str(owned)], 65536, 30)
        if not command_ok(extracted): raise RuntimeError("Cubin extraction failed")
        cubins = list(extract.iterdir())
        if (len(cubins) != 1 or cubins[0].suffix != ".cubin"
                or cubins[0].is_symlink() or not cubins[0].is_file() or cubins[0].stat().st_size > LIMITS["owned.cubin"]):
            raise ValueError("Expected exactly one bounded owned cubin")
        report["extracted_cubin_name"] = cubins[0].name
        shutil.copyfile(cubins[0], ARTIFACTS / "owned.cubin")
        dumped = command("sass.txt", ["/usr/local/cuda/bin/nvdisasm", "--print-line-info-inline", str(cubins[0])], 32*MIB, 90)
        sass = dumped["stdout"] + dumped["stderr"]
        (ARTIFACTS / "sass.txt").write_text(sass)
        checkpoint()
        if not command_ok(dumped): raise RuntimeError("Annotated SASS dump incomplete; no retry")
        require_both_big(sass)
        mapping = bool(re.search(r'File\s+"[^"\n]*cuda\.cu"[^\n]*line\s+\d+', sass)
                       or (re.search(r'\.file[^\n]*cuda\.cu', sass) and re.search(r'\.loc\s+\d+\s+\d+', sass)))
        report["source_line_mapping_present"] = mapping
        if not mapping: raise RuntimeError("SASS source-line mapping absent")
        report["status"] = "compile_complete"
    except Exception as exc:
        report.update(status="compile_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            if ARTIFACTS.is_dir(): report["artifact_bundle"] = bundle_artifacts()
        except Exception as exc:
            report.update(status="compile_failed", bundle_error=f"{type(exc).__name__}: {exc}")
        checkpoint()
        print(json.dumps({"status": report["status"], "error": report.get("error")}), flush=True)


def retrieve(sandbox, destination: Path, *, recovery=False) -> dict:
    """Fixed immutable bundle only; cap and verify every member before extraction."""
    def read(path, cap, timeout):
        process = sandbox.exec("python3", "-c", READ_CAPPED, path, timeout=timeout, text=False)
        value = process.stdout.read()
        process.stderr.read()
        process.wait()
        if process.returncode or type(value) is not bytes or len(value) > cap:
            raise ValueError("Compile artifact retrieval failed/oversized")
        return value
    text = read(str(RESULT), 2*MIB, 5)
    result = json.loads(text)
    if (result.get("source_sha256") != SOURCE_SHA or result.get("native_source_sha256") != NATIVE_SHA
            or result.get("cpp_source_sha256") != CPP_SHA or result.get("gpu") is not None or result.get("device_execution") is not False
            or result.get("status") not in ("compile_running", "compile_complete", "compile_failed")):
        raise ValueError("Compile checkpoint identity mismatch")
    bundle = result.get("artifact_bundle")
    if result["status"] == "compile_complete" and (not bundle or result.get("source_line_mapping_present") is not True
            or set(bundle.get("files", {})) != set(LIMITS)):
        raise ValueError("Complete compile lacks required artifacts/line mapping")
    if bundle:
        if bundle.get("path") != str(BUNDLE) or type(bundle.get("bytes")) is not int or not 0 <= bundle["bytes"] <= BUNDLE_LIMIT:
            raise ValueError("Compile bundle metadata invalid")
        data = read(str(BUNDLE), BUNDLE_LIMIT, 15)
        if len(data) != bundle["bytes"] or sha(data) != bundle["sha256"]: raise ValueError("Compile bundle hash/size mismatch")
        records = bundle["files"]
        if not set(records) <= set(LIMITS): raise ValueError("Compile bundle has unexpected files")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len(records) or {m.name for m in members} != set(records): raise ValueError("Duplicate/missing archive members")
            for member in members:
                expected = records[member.name]
                if not member.isfile() or member.size != expected["bytes"] or member.size > LIMITS[member.name]:
                    raise ValueError("Invalid artifact member")
                value = archive.extractfile(member).read(member.size + 1)
                if len(value) != member.size or sha(value) != expected["sha256"]: raise ValueError("Artifact hash mismatch")
                (destination / ("compile-" + member.name)).write_bytes(value)
        (destination / "compile-artifacts.tar.gz").write_bytes(data)
    write_json(destination / ("compile-recovered-result.json" if recovery else "result.json"), result)
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--child", "/tmp/okc-payload.json"]: raise SystemExit("Fixed compile child only")
    child(json.loads(Path(sys.argv[2]).read_text()))
