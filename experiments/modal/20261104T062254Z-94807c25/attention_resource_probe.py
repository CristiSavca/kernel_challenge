"""Local-only host resource export and bounded compiler capture; stdlib only.

Never insert this into a scored candidate. The worker transforms a digest-pinned
source before import, wraps prepare/warmup with capture_compile_call, then calls
collect_attention_resources on the already-built extension. No runner edits.
"""
import ast
import hashlib
import os
import re
import select
import sys
import threading
import types
from pathlib import Path


RESOURCE_FIELDS = (
    "num_regs", "static_shared_bytes", "constant_bytes", "local_bytes",
    "max_threads_per_block", "max_dynamic_shared_bytes", "binary_version", "ptx_version",
    "active_blocks_per_sm", "threads_per_block", "requested_dynamic_shared_bytes",
    "sm_count", "compute_major", "compute_minor", "smem_struct_bytes",
    "default_consumer_regs", "producer_threads", "producer_regs", "initial_regs_per_thread",
)
CPP_EXPORT = "\nstd::vector<int64_t> q8_role_resources_v10();\n"
CUDA_EXPORT = r'''
// LOCAL_RESOURCE_EXPORT_V10: host code only; no device/timed instructions.
std::vector<int64_t> q8_role_resources_v10() {
  // launch<8> selects attn_kernel<8,true> for p.flat: default MODE is0.
  auto fn = attn_kernel<8, true, 0>;
  cudaFuncAttributes a;
  cudaError_t e = cudaFuncGetAttributes(&a, fn);
  TORCH_CHECK(e == cudaSuccess, "attention attributes: ", cudaGetErrorString(e));
  const int smem = (int)sizeof(Smem<8, true>) + 1024;
  const int threads = Cfg<8, true>::NTX;
  int active = 0, device = 0, sms = 0, major = 0, minor = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fn, threads, smem);
  TORCH_CHECK(e == cudaSuccess, "attention occupancy: ", cudaGetErrorString(e));
  e = cudaGetDevice(&device);
  TORCH_CHECK(e == cudaSuccess, "attention device: ", cudaGetErrorString(e));
  e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  TORCH_CHECK(e == cudaSuccess, "attention SM count: ", cudaGetErrorString(e));
  e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  TORCH_CHECK(e == cudaSuccess, "attention major: ", cudaGetErrorString(e));
  e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TORCH_CHECK(e == cudaSuccess, "attention minor: ", cudaGetErrorString(e));
  return {(int64_t)a.numRegs, (int64_t)a.sharedSizeBytes, (int64_t)a.constSizeBytes,
          (int64_t)a.localSizeBytes, (int64_t)a.maxThreadsPerBlock,
          (int64_t)a.maxDynamicSharedSizeBytes, (int64_t)a.binaryVersion, (int64_t)a.ptxVersion,
          active, threads, smem, sms, major, minor, (int64_t)sizeof(Smem<8, true>),
          Cfg<8, true>::CREGS, Cfg<8, true>::NP, Cfg<8, true>::PREGS, Cfg<8, true>::BASEREGS};
}
'''


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def instrument_source(source, *, expected_sha256, tag):
    if type(source) is not str or _digest(source) != expected_sha256:
        raise ValueError("attention source digest mismatch")
    if type(tag) is not str or re.fullmatch(r"[a-z0-9_]{1,48}", tag) is None:
        raise ValueError("invalid diagnostic namespace tag")
    tree = ast.parse(source)
    assignments = {n.targets[0].id: n for n in tree.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    native_name = "ATT7_CUDA_SRC" if "ATT7_CUDA_SRC" in assignments else "CUDA_SRC"
    cpp_name = "ATT7_CPP_SRC" if native_name.startswith("ATT7") else "CPP_SRC"
    fn_name = "_att7" if native_name.startswith("ATT7") else "_build"
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    edits = []
    for name, suffix in ((native_name, CUDA_EXPORT), (cpp_name, CPP_EXPORT)):
        node = assignments[name]
        old = ast.get_source_segment(source, node)
        value = ast.literal_eval(node.value)
        if name == native_name:
            assert "p.flat ? (KernFn)attn_kernel<L, true>" in value
            assert "template <int L, bool FLAT, int MODE = 0>" in value
        new = name + ' = r"""' + value + suffix + '"""'
        edits.append((old, new))
    old = ast.get_source_segment(source, function)
    new = old
    for before, after in (("functions=[\"run\"]", "functions=[\"run\", \"q8_role_resources_v10\"]"),
                          ("verbose=False", "verbose=True"),
                          ('"-O3"', '"-O3", "-Xptxas=-v"'),
                          ('name=f"', 'name=f"diag_' + tag + '_')):
        assert new.count(before) == 1, before
        new = new.replace(before, after, 1)
    edits.append((old, new))
    result = source
    for before, after in edits:
        assert result.count(before) == 1
        result = result.replace(before, after, 1)
    ast.parse(result)
    restored = result
    for before, after in reversed(edits):
        assert restored.count(after) == 1
        restored = restored.replace(after, before, 1)
    assert restored == source
    return result, {"schema": 1, "input_sha256": expected_sha256,
                    "output_sha256": _digest(result), "helper_sha256": _digest(Path(__file__).read_text()),
                    "exact_reverse_proof": True, "template": "attn_kernel<8,true,0>",
                    "changes": ["host-only CUDA export", "host pybind declaration/export",
                                "unique extension namespace", "verbose compiler output", "-Xptxas=-v"]}


_CAPTURE_LOCK = threading.Lock()


class CompileCaptureError(RuntimeError):
    def __init__(self, capture):
        super().__init__("wrapped prepare/compile failed; complete bounded diagnostics are in .capture")
        self.capture = capture


def capture_compile_call(fn, *, max_bytes=8 * 1024 * 1024):
    """Return (fn_result, capture) or re-raise after restoring process FDs.

    Process-wide FD capture is serialized. Output beyond the cap is drained
    but discarded and explicitly invalidates the resource/diagnostic gate.
    Full captured text is returned for the worker to save as an artifact.
    """
    if type(max_bytes) is not int or not 1024 <= max_bytes <= 64 * 1024 * 1024:
        raise ValueError("compiler capture budget must be 1KiB..64MiB")
    with _CAPTURE_LOCK:
        sys.stdout.flush(); sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        reader, writer = os.pipe()
        stop = threading.Event()
        chunks, count = [], [0]
        failure = []
        def drain():
            try:
                while True:
                    ready, _, _ = select.select([reader], [], [], 0.1)
                    if not ready:
                        if stop.is_set():
                            break
                        continue
                    block = os.read(reader, 65536)
                    if not block:
                        break
                    room = max(0, max_bytes - count[0])
                    if room:
                        chunks.append(block[:room])
                    count[0] += len(block)
            except OSError:
                failure.append("capture_read_error")
            finally:
                os.close(reader)
        thread = threading.Thread(target=drain, name="attention-compile-capture", daemon=True)
        thread.start()
        original_error = None
        try:
            os.dup2(writer, 1); os.dup2(writer, 2); os.close(writer)
            value = fn()
        except Exception as error:
            original_error = error
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])
            stop.set()
            thread.join(timeout=5)
        report = {"bytes_total": count[0], "bytes_saved": min(count[0], max_bytes),
                  "limit_bytes": max_bytes, "truncated": count[0] > max_bytes,
                  "capture_error": failure[0] if failure else "reader_timeout" if thread.is_alive() else None,
                  "text": b"".join(chunks).decode("utf-8", errors="replace")}
        report["complete"] = not report["truncated"] and report["capture_error"] is None
        if original_error is not None:
            raise CompileCaptureError(report) from original_error
        return value, report


def compiler_summary(capture):
    text = capture["text"]
    context = None
    warnings = []
    for line in text.splitlines():
        match = re.search(r"Compiling entry function ['\"]([^'\"]+)", line)
        if match:
            context = match.group(1)
        if re.search(r"warning|C7515|serializ", line, re.IGNORECASE):
            explicit = re.search(r"attn_kernelILi8ELb1ELi0EE", line)
            kind = "changed_flat" if explicit or (context and "attn_kernelILi8ELb1ELi0EE" in context) else "other_kernel" if context else "unattributed"
            warnings.append({"scope": kind, "kernel": context[:512] if context else None, "line": line[:1024]})
    complete = capture["complete"] and len(warnings) <= 256
    saw_ptxas = "ptxas" in text
    return {"complete": complete, "saw_ptxas": saw_ptxas,
            "compile_log_gate": complete and saw_ptxas, "warning_count": len(warnings),
            "warnings": warnings[:256], "warnings_omitted": max(0, len(warnings) - 256),
            "attribution_caveat": "context attribution is inferred from sequential compiler output; keep full log"}


def collect_attention_resources(module):
    namespace = vars(module) if type(module) is types.ModuleType else module if type(module) is dict else None
    if namespace is None:
        return {"status": "unavailable", "reason": "unsupported_namespace"}
    extension = namespace.get("_ATT7", namespace.get("_MOD"))
    if extension is None:
        return {"status": "unavailable", "reason": "extension_not_built"}
    export = getattr(extension, "q8_role_resources_v10", None)
    if export is None:
        return {"status": "unavailable", "reason": "resource_export_missing"}
    values = export()
    if type(values) not in (list, tuple) or len(values) != len(RESOURCE_FIELDS) or any(
            type(value) is not int or abs(value) >= 1 << 63 for value in values):
        return {"status": "unavailable", "reason": "invalid_export_scalar_values"}
    result = dict(zip(RESOURCE_FIELDS, values))
    result.update(status="ok", template="attn_kernel<8,true,0>")
    result["two_cta_resource_gate"] = result["active_blocks_per_sm"] >= 2 and result["local_bytes"] == 0
    return result
