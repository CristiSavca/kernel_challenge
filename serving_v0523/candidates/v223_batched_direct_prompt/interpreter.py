"""Replay the submitted segmented byte packer with hostile slot tables and independent scalar slices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def replay(kernel, rng, width, dtype, queries, prefixes, left):
    count = len(queries)
    cuq = np.r_[0, np.cumsum(queries)].astype(np.int32)
    lengths = (np.array(queries) + prefixes).astype(np.int32)
    halos = np.array(prefixes) if left < 0 else np.minimum(prefixes, left)
    used = (queries + halos).astype(np.int32)
    starts = np.r_[0, np.cumsum(used + 256)].astype(np.int32)
    maxk, slots = int(max(lengths)), 589
    table = np.full((count, maxk + 3), -1000000000, dtype=dtype)
    caches = [rng.integers(0, 256, (slots, width + extra), dtype=np.uint8) for extra in (3, 13)]
    current = [rng.integers(0, 256, (cuq[-1], width), dtype=np.uint8) for _ in range(2)]
    out = [np.full((starts[-1] + 2, width), 77, dtype=np.uint8) for _ in range(2)]
    expected = [t.copy() for t in out]
    for b, (q, p, h) in enumerate(zip(queries, prefixes, halos)):
        # The packer is allowed to look up only the prior, still-visible prefix.
        lookup = rng.integers(1, slots, int(h), dtype=dtype)
        table[b, p-h:p] = lookup
        for target, c, x in zip(expected, caches, current):
            i = starts[b] + 1
            target[i:i+h] = c[lookup, :width]
            target[i+h:i+h+q] = x[cuq[b]:cuq[b+1]]
            target[i+h+q:i+h+q+256] = 0
    block = min(32, 4096 // width)
    args = [buf(t, tl.uint8) for t in (*caches, *current)]
    args += [buf(table, tl.int32 if dtype == np.int32 else tl.int64)]
    args += [buf(t, tl.int32) for t in (lengths, cuq, starts, used)]
    args += [buf(t[1:-1], tl.uint8) for t in out]
    kernel[(min(9, triton.cdiv(int(max(used)) + 256, block)),count)](
        *args, caches[0].shape[1], caches[1].shape[1], table.shape[1], ROW=width, BLOCK=block, GUARD=256)
    for actual, want in zip(out, expected):
        np.testing.assert_array_equal(actual, want)
    return int(2 * starts[-1] * width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--compile-fixtures", action="store_true")
    args = ap.parse_args()
    if args.compile_fixtures:
        cases = []
        for width in (16, 32, 64, 128, 256, 512, 1024):
            for idx in ("i32", "i64"):
                types = ("*u8 " * 4 + "*" + idx + " " + "*i32 " * 4 + "*u8 *u8 i32 i32 i32").split()
                cases.append(dict(file="kernels/ops/attention/batched_prompt_kv.py", kernel="_pack_prompt_rows",
                                  label=f"row{width}-{idx}", types=types, num_warps=4, num_stages=3,
                                  constants=dict(ROW=width, BLOCK=min(32,4096//width), GUARD=256)))
        print(json.dumps(cases))
        return
    rng = np.random.default_rng(223)
    cases, total = 0, 0
    with tempfile.TemporaryDirectory() as scratch:
        fn = load_jit_module(args.source, Path(scratch))._pack_prompt_rows
        for width in (16, 32, 64, 128, 256, 512, 1024):
            for dtype in (np.int32, np.int64):
                for q, p, left in (([1,31],[0,512],-1),([17,91,2],[39,7,628],0),
                                   ([43,2,257],[0,300,17],31),([10,67,8,1],[432,0,15,162],127)):
                    total += replay(fn, rng, width, dtype, q, p, left)
                    cases += 1
        for left in (-1, 2047):
            total += replay(fn, rng, 128, np.int32, [11001,5383],[65739,2183],left)
            cases += 1
    print(json.dumps(dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                          triton=triton.__version__, interpreter_cases=cases, checked_kv_bytes=total,
                          guard_rows=256, outer_canaries=True, physical_H200=False),indent=2))


if __name__ == "__main__":
    main()
