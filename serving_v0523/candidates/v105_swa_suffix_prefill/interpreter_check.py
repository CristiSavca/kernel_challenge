"""Compare the production suffix packer with a byte oracle under Triton's CPU interpreter."""
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


def run_case(kernel, width, left, table_dtype, qlen, klen, padded_rows, rng, word):
    batch = len(qlen)
    table_width = max(max(klen), 1)
    bounds = np.array([0, *np.cumsum(qlen)], dtype=np.int32)
    q_capacity = int(bounds[-1]) + padded_rows
    klen = np.asarray(klen, dtype=np.int32)
    src_rows = 113
    ks, vs = (width + 12, width + 8) if word else (width + 13, width + 7)
    k = rng.integers(0, 256, (src_rows, ks), dtype=np.uint8)
    v = rng.integers(0, 256, (src_rows, vs), dtype=np.uint8)
    # Expired logical positions point at a poison row. Live rows are interleaved
    # across physical pages and requests; table rows also have a stride gap.
    k[0] = 255
    v[0] = 255
    table = np.zeros((batch, table_width + 7), dtype=table_dtype)
    capacity = q_capacity + batch * (left + 256) + 256
    def dest():
        return np.full((capacity + 2, width), 181, dtype=np.uint8)
    dk, dv = dest(), dest()
    expect_k, expect_v = dest(), dest()
    used = np.zeros(batch + 1, dtype=np.int32)
    for b in range(batch):
        count = min(int(klen[b]), int(qlen[b]) + left) if qlen[b] else 0
        offset = int(klen[b]) - count
        rows = rng.integers(1, src_rows, count)
        table[b, offset:klen[b]] = rows
        start = 1 + bounds[b] + b * (left + 256)
        used[b] = count
        for orig, expected in ((k, expect_k), (v, expect_v)):
            expected[start:start+count] = orig[rows, :width]
            expected[start+count:start+count+256] = 0
    dummy = 1 + int(bounds[-1]) + batch * (left + 256)
    expect_k[dummy:dummy+256] = expect_v[dummy:dummy+256] = 0
    used[-1] = 1
    cu_q_out = np.full(batch+2, -3, dtype=np.int32)
    cu_k_out = cu_q_out.copy()
    used_out = np.full(batch+1, -3, dtype=np.int32)
    block_rows = min(32, (4096 if word else 2048) // width)
    unit = 4 if word else 1
    def storage(array):
        return buf(array.view(np.int32), tl.int32) if word else buf(array, tl.uint8)
    tile_count = triton.cdiv(min(table_width, max(qlen) + left)+256, block_rows)
    # A small grid exercises multiple streaming iterations per CTA.
    grid = (min(5, tile_count), batch+1)
    kernel[grid](
        storage(k), storage(v),
        buf(table, tl.int32 if table_dtype == np.int32 else tl.int64),
        buf(klen, tl.int32), buf(bounds, tl.int32),
        storage(dk[1:-1]), storage(dv[1:-1]),
        buf(cu_q_out, tl.int32), buf(cu_k_out, tl.int32), buf(used_out, tl.int32),
        ks // unit, vs // unit, table.shape[1], q_capacity, batch,
        LEFT=left, ROW_WIDTH=width // unit, BLOCK_ROWS=block_rows,
    )
    np.testing.assert_array_equal(dk, expect_k)
    np.testing.assert_array_equal(dv, expect_v)
    np.testing.assert_array_equal(cu_q_out, [*bounds, q_capacity])
    np.testing.assert_array_equal(cu_k_out, [*(bounds + np.arange(batch+1)*(left+256)), capacity])
    np.testing.assert_array_equal(used_out, used)
    return dk.nbytes + dv.nbytes


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("source", type=Path)
    source = cli.parse_args().source
    rng = np.random.default_rng(23871)
    cases = 0
    checked = 0
    shapes = [
        (0, [0, 2, 35], [83, 39, 67], 13),
        (8, [17, 0, 73], [26, 0, 322], 0),
        (257, [9, 47], [9, 503], 65),
    ]
    with tempfile.TemporaryDirectory() as temp:
        kernel = load_jit_module(source, Path(temp))._pack_swa_suffix
        for width in (16, 32, 64, 128, 256, 512, 1024):
            for dtype in (np.int32, np.int64):
                for left, q, k, pad in shapes:
                    for word in (False, True):
                        checked += run_case(kernel, width, left, dtype, q, k, pad, rng, word)
                        cases += 1
        for dtype in (np.int32, np.int64):
            for word in (False, True):
                checked += run_case(kernel, 128, 2047, dtype, [0, 7, 128, 561],
                                    [100, 384, 7803, 80111], 192, rng, word)
                cases += 1
    print(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "triton": triton.__version__, "device": "CPU interpreter",
                      "cases": cases, "destination_bytes_checked": checked,
                      "covers": ["zero-length query slots", "unequal K/V/table strides",
                                 "ragged batch", "graph padding", "FP8 byte payloads",
                                 "reused physical slots", "80k prefix", "partial tiles",
                                 "zero guard regions", "untouched storage canaries"]}, indent=2))


if __name__ == "__main__":
    main()
