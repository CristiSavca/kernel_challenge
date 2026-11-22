"""Run the paged suffix kernel against fragmented physical K and V stores on CPU."""
import argparse
import hashlib
import itertools
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


def physical_store(width, page_size, token_gap, page_gap, rng):
    token_pitch = width + token_gap
    page_pitch = page_size * token_pitch + page_gap
    return rng.integers(0, 256, size=37*page_pitch, dtype=np.uint8), page_pitch, token_pitch


def run(kernel, rng, width, page_size, word, index_type, left):
    unit = 4 if word else 1
    q = np.array([0, 7, 39, 521], dtype=np.int32)
    lengths = np.array([777, 7, 4311, 80133], dtype=np.int32)
    batch = len(q)
    cu = np.r_[np.zeros(1, dtype=np.int32), np.cumsum(q, dtype=np.int32)]
    capacity_q = int(cu[-1]) + 59
    counts = np.where(q > 0, np.minimum(lengths, q + left), 0)
    bounds = cu + np.arange(batch+1, dtype=np.int32)*(left+256)
    capacity_k = capacity_q + batch*(left+256) + 256
    table = np.zeros((batch, triton.cdiv(max(lengths), page_size)+5), dtype=index_type)
    k, kp, kt = physical_store(width, page_size, 3*unit, 5*unit, rng)
    v, vp, vt = physical_store(width, page_size, 5*unit, 13*unit, rng)
    # Page zero stands in for inactive/evicted entries. Each live logical page
    # receives an independent physical page, including partial boundary pages.
    k[:kp] = v[:vp] = 255
    dk = np.full((capacity_k+2, width), 205, dtype=np.uint8)
    dv = dk.copy()
    ek, ev = dk.copy(), dv.copy()
    for b in range(batch):
        n = int(counts[b])
        begin = int(lengths[b])-n
        physical = {}
        for logical in range(begin, int(lengths[b])):
            page_no, intra = divmod(logical, page_size)
            if page_no not in physical:
                physical[page_no] = int(rng.integers(1, 37))
                table[b, page_no] = physical[page_no]
            physical_page = physical[page_no]
            dest = 1+int(bounds[b])+logical-begin
            for bank, page_pitch, token_pitch, expected in [(k, kp, kt, ek), (v, vp, vt, ev)]:
                addr = physical_page*page_pitch + intra*token_pitch
                expected[dest] = bank[addr:addr+width]
        stop = 1+int(bounds[b])+n
        ek[stop:stop+256] = ev[stop:stop+256] = 0
    dummy = 1+int(bounds[-1])
    ek[dummy:dummy+256] = ev[dummy:dummy+256] = 0
    q_out = np.full(batch+2, -7, dtype=np.int32)
    k_out = q_out.copy()
    used = np.full(batch+1, -7, dtype=np.int32)
    def raw(x):
        return buf(x.view(np.int32), tl.int32) if word else buf(x, tl.uint8)
    rows = min(32, (4096 if word else 2048)//width)
    kernel[(7, batch+1)](
        raw(k), raw(v), buf(table, tl.int32 if index_type == np.int32 else tl.int64),
        buf(lengths, tl.int32), buf(cu, tl.int32), raw(dk[1:-1]), raw(dv[1:-1]),
        buf(q_out, tl.int32), buf(k_out, tl.int32), buf(used, tl.int32),
        kp//unit, kt//unit, vp//unit, vt//unit, table.shape[1], capacity_q, batch,
        PAGE_SIZE=page_size, LEFT=left, ROW_WIDTH=width//unit, BLOCK_ROWS=rows,
    )
    np.testing.assert_array_equal(dk, ek)
    np.testing.assert_array_equal(dv, ev)
    np.testing.assert_array_equal(q_out, [*cu, capacity_q])
    np.testing.assert_array_equal(k_out, [*bounds, capacity_k])
    np.testing.assert_array_equal(used, [*counts, 1])
    return dk.nbytes+dv.nbytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    source=parser.parse_args().source
    rng=np.random.default_rng(8917)
    cases = 0
    compared = 0
    with tempfile.TemporaryDirectory() as tmp:
        kernel = load_jit_module(source, Path(tmp))._pack_swa_suffix
        for page, width, word, index in itertools.product(
            (1, 7, 128, 160), (16, 128, 512), (False, True), (np.int32, np.int64)
        ):
            left = 2047 if width == 128 else 239
            compared += run(kernel, rng, width, page, word, index, left)
            cases += 1
    print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          triton=triton.__version__, device="CPU interpreter", cases=cases,
          destination_bytes_checked=compared, page_sizes=[1,7,128,160],
          max_logical_context=80133, independent_page_token_strides=True,
          ragged_query_padding=True, partial_page_boundaries=True),indent=2))


if __name__ == "__main__":
    main()
