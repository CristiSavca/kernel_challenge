"""Compile the frozen sparse-return kernel or run it on host buffers in the Triton interpreter."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

from compile_decode_sm90 import load_jit_module


def compile_cases(fn):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    import subprocess

    cases = []
    assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    for ranks in (1, 2, 4, 8):
        for first, prior, out in [(True, "bf16", "bf16"), (True, "fp32", "fp32"),
                                  (False, "fp32", "fp32"), (False, "fp32", "bf16")]:
            const = dict(SOURCES=ranks, FIRST=first, BLOCK_M=4, BLOCK_N=256)
            sig = dict(zip(fn.arg_names[:7], ["*i32", "*bf16", "*"+prior, "*"+out, "i32", "i32", "i32"]))
            sig.update({k: "constexpr" for k in const})
            binary = triton.compile(ASTSource(fn, sig, constexprs=const), target=GPUTarget("cuda", 90, 32))
            ptx = binary.asm["ptx"]
            assert "add.rn.f32" in ptx or "add.f32" in ptx or ranks == 1 and first
            if out == "bf16" and (ranks > 1 or not first):
                assert "cvt.rn.bf16" in ptx
            with tempfile.TemporaryDirectory() as t:
                src = Path(t) / "reduce.ptx"
                src.write_text(ptx)
                info = subprocess.run([assembler, "-v", "-arch=sm_90a", src, "-o", Path(t)/"reduce.cubin"],
                                      capture_output=True, text=True, check=True).stderr
                assert "0 bytes spill stores, 0 bytes spill loads" in info
            cases.append([ranks,first,prior,out,hashlib.sha256(binary.asm["cubin"]).hexdigest()])
    return dict(compiled=len(cases), zero_spill_and_bf16_rn=True, binaries=cases)


def interpret(fn):
    from interpret_decode_kernels import buf
    rng = np.random.default_rng(167)
    cases = fp32 = bf16 = guards = 0
    for rows, width in [(1,1),(3,255),(5,257),(7,4096)]:
        for sources in (1,2,4,8):
            for mode in ("holes", "empty", "all"):
                active = rng.random((sources, rows)) < 0.5
                if mode == "empty": active[:] = False
                if mode == "all": active[:] = True
                base = 937
                mapping = np.where(active, np.cumsum(active.ravel()).reshape(active.shape)-1+base, -1).astype("i4")
                length = int(active.sum())
                for first in (True,False):
                    for fout in (True,False):
                        # Use representable integers on BF16 stores (RNE tested in Hopper PTX).
                        vals = np.zeros((max(1, length) + 3, width), dtype="f4")
                        vals[:length] = rng.integers(-9, 10, (length, width))
                        start = rng.integers(-4,5,(rows,width)).astype("f4")
                        if not fout:
                            vals[:length] *= np.exp2(rng.integers(-14,15,(length,width))).astype("f4")
                            start = rng.standard_normal((rows,width)).astype("f4")
                        vals[length:] = np.nan
                        expected = np.zeros_like(start) if first else start.copy()
                        for source in range(sources):
                            valid = active[source]
                            expected[valid] += vals[mapping[source,valid]-base]
                        precision = tl.bfloat16 if fout else tl.float32
                        payload = (np.ascontiguousarray(vals).view("u4") >> 16).astype("u2")
                        sentinel = 145
                        dest = np.full(rows*width+33,sentinel,dtype="u2" if fout else "f4")
                        fn[(triton.cdiv(rows,4),triton.cdiv(width,256))](
                            buf(mapping,tl.int32),buf(payload,tl.bfloat16),buf(start,tl.float32),
                            buf(dest[9:],precision),rows,width,base,SOURCES=sources,FIRST=first,BLOCK_M=4,BLOCK_N=256)
                        result = dest[9:9+rows*width].reshape(expected.shape)
                        target = (expected.view("u4") >> 16).astype("u2") if fout else expected
                        np.testing.assert_array_equal(result, target)
                        assert (dest[:9] == sentinel).all() and (dest[9+rows*width:] == sentinel).all()
                        cases += 1; guards += 33
                        if fout: bf16 += expected.size
                        else: fp32 += expected.size
    return dict(cases=cases,fp32_values=fp32,representable_bf16_values=bf16,guard_values=guards)


if __name__ == "__main__":
    mode, source = sys.argv[1], Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as temp:
        fn = load_jit_module(source, Path(temp))._sum_rank_group
        result = (compile_cases if mode == "compile" else interpret)(fn)
    print(json.dumps(dict(triton=triton.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          physical_gpu=False,**result), indent=2))
