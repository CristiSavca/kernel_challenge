"""Assemble the production gate for SM90 and compare its Triton interpreter with two rounded products."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
import triton.language as tl


def source_kernel(source, directory):
    text = source.read_text()
    node = next(f for f in ast.parse(text).body if getattr(f, "name", "") == "_gated_residual_kernel")
    first = node.decorator_list[0].lineno - 1
    path = directory / "gate.py"
    path.write_text("import triton\nimport triton.language as tl\n" + "\n".join(text.splitlines()[first:node.end_lineno]) + "\n")
    spec = importlib.util.spec_from_file_location("gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gate"] = module
    spec.loader.exec_module(module)
    return module._gated_residual_kernel


def compile_cases(kernel):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    cases = []
    for width in [63, 8192, 10091]:
        for index in ["i32", "i64"]:
            constants = dict(WIDTH=width, BLOCK=512)
            signature = dict(Stream="*fp32", Branch="*bf16", Alpha="*fp32", Beta="*fp32",
                             Output="*fp32", count=index, WIDTH="constexpr", BLOCK="constexpr")
            result = triton.compile(ASTSource(kernel, signature, constexprs=constants), target=GPUTarget("cuda", 90, 32),
                                    options=dict(enable_fp_fusion=False))
            ptx = result.asm["ptx"]
            ops = {op: len(re.findall(r"\b" + op + r"(?:\.rn)?\.f32\s", ptx)) for op in ("mul", "add", "fma")}
            assert ops["mul"] >= 2 and ops["add"] >= 1 and ops["fma"] == 0
            assert ".ftz." not in ptx
            cuobjdump = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
            with tempfile.TemporaryDirectory() as t:
                cubin = Path(t) / "gate.cubin"
                cubin.write_bytes(result.asm["cubin"])
                sass = subprocess.check_output([cuobjdump, "--dump-sass", cubin], text=True)
            isa = {op: len(re.findall(r"\b" + op + r"\b", sass)) for op in ("FMUL", "FADD", "FFMA")}
            assert isa["FMUL"] >= 2 and isa["FADD"] >= 1 and isa["FFMA"] == 0
            cases.append(dict(hidden=width, index=index, scalar_ptx_ops=ops, sm90_ops=isa, cubin_bytes=len(result.asm["cubin"]),
                              cubin_sha256=hashlib.sha256(result.asm["cubin"]).hexdigest()))
    return dict(arch="SM90", cases=cases)


def interpret_cases(kernel):
    import numpy as np
    from interpret_decode_kernels import buf

    records = []
    rng = np.random.default_rng(257)
    for rows, width in [(1, 1), (7, 63), (127, 65), (32, 8192), (2048, 127)]:
        n = rows * width
        alpha = rng.uniform(-2, 2, width).astype("f4")
        beta = rng.uniform(-2, 2, width).astype("f4")
        stream = rng.uniform(-100, 100, n).astype("f4")
        branch = (rng.uniform(-100, 100, n).astype("f4").view("u4") >> 16).astype("u2")
        patterns = [0, 0x8000, 1, 0x8001, 0x3F80, 0xBF80, 0x7F7F]
        for i, bits in enumerate(patterns[:n]):
            stream[i] = 0.; alpha[i % width] = -1.; beta[i % width] = .25; branch[i] = bits
        raw = np.full(n + 2, 1234567., dtype="f4")
        real = (branch.astype("u4") << 16).view("f4")
        want = (stream.reshape(rows, width) * alpha + real.reshape(rows, width) * beta).ravel()
        kernel[(triton.cdiv(n, 512),)](buf(stream, tl.float32), buf(branch, tl.bfloat16),
            buf(alpha, tl.float32), buf(beta, tl.float32), buf(raw[1:-1], tl.float32),
            n, width, 512, enable_fp_fusion=False)
        assert np.array_equal(raw[1:-1].view("u4"), want.view("u4")), (rows, width, raw[1:8], want[:7])
        assert raw[0] == raw[-1] == 1234567.
        records.append(dict(rows=rows, hidden=width, matched_fp32_words=n))
    # Every BF16 encoding occurs exactly once. Compare NaN classification and all remaining output bits.
    bits = np.arange(65536, dtype="u2")
    n = bits.size
    stream = np.zeros(n, dtype="f4")
    alpha = np.zeros(1, dtype="f4")
    beta = np.ones(1, dtype="f4")
    output = np.empty_like(stream)
    expected = (bits.astype("u4") << 16).view("f4")
    with np.errstate(invalid="ignore"):
        expected = stream * alpha + expected * beta
        kernel[(triton.cdiv(n,512),)](buf(stream, tl.float32), buf(bits, tl.bfloat16),
            buf(alpha, tl.float32), buf(beta, tl.float32), buf(output, tl.float32), n, 1, 512, enable_fp_fusion=False)
    finite = ~np.isnan(expected)
    assert np.array_equal(np.isnan(expected), np.isnan(output))
    assert np.array_equal(expected[finite].view("u4"), output[finite].view("u4"))
    return dict(cases=records, matched_fp32_words=sum(r["matched_fp32_words"] for r in records),
                exhaustive_BF16_words=n, matched_non_nan_bits=int(finite.sum()), matched_nan_masks=int((~finite).sum()))


if __name__ == "__main__":
    source, mode = Path(sys.argv[1]), sys.argv[2]
    with tempfile.TemporaryDirectory() as tmp:
        k = source_kernel(source, Path(tmp))
        data = compile_cases(k) if mode == "compile" else interpret_cases(k)
    print(json.dumps(dict(triton=triton.__version__, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          mode=mode, physical_H200=False, **data), indent=2))
