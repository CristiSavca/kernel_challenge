"""Check the candidate's actual strided BF16 rank reduction in Triton's interpreter."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import triton
import triton.language as tl

from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def round_bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def run(source):
    rng = np.random.default_rng(150)
    cases = values = guard_values = exact_bf16 = 0
    with tempfile.TemporaryDirectory() as temp:
        fn = load_jit_module(source, Path(temp))._sum_sources
        for size in (4, 8):
            for rows, width in ((1,1), (3,11), (1,255), (3,256), (5,257), (4,4096)):
                for guards in (0, 13):
                    for mode in ("random", "ties", "cancellation"):
                        n = rows * width
                        stride = width + guards
                        if mode == "random":
                            v = np.ldexp(rng.standard_normal((size,n)), rng.integers(-25,26,(size,n))).astype("f4")
                        else:
                            v = np.zeros((size,n), dtype=np.float32)
                            if mode == "ties":
                                v[0] = 1 + (np.arange(n) % 2) / 128
                                v[1] = 1 / 256
                            else:
                                v[0] = 256
                                v[1] = 1
                                v[2] = -256
                                v[3] = np.where(np.arange(n) % 2, -0.5, 0.5)
                        encoded = round_bf16(v)
                        decoded = (encoded.astype(np.uint32) << 16).view("f4")
                        expected = np.zeros(n, dtype="f4")
                        for part in decoded:
                            expected += part
                        # Triton 3.8's CPU implicit BF16 store truncates. Check
                        # every pre-round FP32 result and exact BF16 stores where
                        # no rounding is needed. The SM90 PTX check covers RN.
                        for dt, native, target in ((tl.float32, np.float32, expected),
                                                   (tl.bfloat16, np.uint16, round_bf16(expected))):
                            if dt == tl.bfloat16 and np.any(expected.view("u4") & 0xffff):
                                continue
                            sentinel = native(42333)
                            allocation = np.full((rows,stride), sentinel, dtype=native)
                            fn[(triton.cdiv(n,256),)](
                                buf(encoded,tl.bfloat16), buf(allocation.ravel()[guards//2:],dt),
                                n,width,stride,size,256)
                            start = guards//2
                            out = allocation[:,start:start+width]
                            np.testing.assert_array_equal(out.ravel(), target)
                            assert np.all(allocation[:,:start] == sentinel)
                            assert np.all(allocation[:,start+width:] == sentinel)
                            guard_values += guards * rows
                            if dt == tl.bfloat16:
                                exact_bf16 += n
                        cases += 1
                        values += n
    return dict(cases=cases, fp32_pre_round_values=values, exact_bf16_stores=exact_bf16, untouched_guard_elements=guard_values,
                modes=["rounded random partials", "ties to even", "FP32 cancellation"],
                triton=triton.__version__, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                physical_gpu=False, rounded_bf16_store="checked in generated Hopper PTX")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    print(json.dumps(run(parser.parse_args().source),indent=2))
