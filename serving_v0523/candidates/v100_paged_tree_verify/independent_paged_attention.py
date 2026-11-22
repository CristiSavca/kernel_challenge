"""Compare the production packed tree attention on all physical pages to its page-one control."""
from __future__ import annotations

import itertools
import json
import runpy
import sys
from pathlib import Path

import numpy as np


def run(checkout):
    here = Path(__file__).resolve().parent
    sim = runpy.run_path(str(here.parent / 'v70_dflash_selector_tree/cpu_check.py'))
    ptr, ary = sim['Pointer'], sim['array']
    src = checkout / 'python/sglang/kernels/ops/attention/paged_verify_attention.py'
    pid, (_, _, partition, merge) = sim['program'](
        src, ('_tile_geometry', '_plan', '_partition_attention', '_merge_partitions')
    )
    rng = np.random.default_rng(719931)
    cases = 0
    outputs = 0
    for width, page_size, window, splits in itertools.product(
        (4, 8), (16, 128, 160), (-1, 15, 2047), (1, 7)
    ):
        prefixes = np.array([0, page_size - 1, page_size + 1, 2 * page_size - 3, 2048])
        bs, gqa, kv_heads, dim = len(prefixes), 4, 2, 64
        qheads, tile_m, tile_n = gqa * kv_heads, gqa * width, 64 if window >= 0 else 128
        lengths = prefixes.astype(np.int32) + width
        n_page = (lengths.max() + page_size - 1) // page_size
        source_rows = np.arange(2, 2 + bs * n_page)
        pages = np.ascontiguousarray(rng.permutation(source_rows).reshape(bs, n_page), dtype=np.int32)
        rows = int((pages.max() + 1) * page_size)
        all_k = rng.normal(size=(rows, kv_heads, dim)).astype(np.float32)
        all_v = ary(rng.normal(size=all_k.shape)).to('bfloat16')
        # Random fragmented physical pages represent either the full pool or the
        # independently translated SWA pool. Reference token slots use their own IDs.
        token_offsets = np.arange(n_page * page_size)
        flat_slots = pages[:, token_offsets // page_size] * page_size + token_offsets % page_size
        cuq = np.arange(bs + 1, dtype=np.int32) * width
        q = rng.normal(size=(bs * width, qheads, dim)).astype(np.float32)
        ac = np.zeros((bs, width), dtype=np.int32)
        depth = np.zeros_like(ac)
        ac[:, 0] = 1
        for b in range(bs):
            for node in range(1, width):
                parent = int(rng.integers(node))
                ac[b, node] = ac[b, parent] | (1 << node)
                depth[b, node] = depth[b, parent] + 1
        def execute(table, physical_page_size):
            partial = np.full((splits + bs - 1, kv_heads, tile_m, dim), 77., np.float32)
            lse = np.full(partial.shape[:-1], np.nan, np.float32)
            out = np.full(q.shape, np.nan, np.float32)
            for part, kh in itertools.product(range(splits), range(kv_heads)):
                pid[:] = [part, kh, 0]
                partition(
                    *[ptr(x) for x in (q, all_k, all_v, table, lengths, cuq)],
                    None, ptr(depth), ptr(ac), ptr(partial), ptr(lse),
                    table.shape[1], physical_page_size,
                    physical_page_size * kv_heads * dim, kv_heads * dim, dim,
                    physical_page_size * kv_heads * dim, kv_heads * dim, dim, 0, 0,
                    bs, kv_heads, gqa, dim, 0.125 * np.log2(np.e), window, width,
                    tile_m, tile_n, splits, 1 << int(bs - 1).bit_length(),
                )
            for row, kh, packed in itertools.product(range(bs), range(kv_heads), range(tile_m)):
                pid[:] = [row, kh, packed]
                merge(
                    *[ptr(x) for x in (partial, lse, lengths, cuq)], None, ptr(out),
                    0, 0, bs, kv_heads, gqa, dim, window, tile_m, tile_n, splits,
                    1 << (bs - 1).bit_length(), 1 << (splits - 1).bit_length(),
                )
            return out
        ref = execute(flat_slots, 1)
        actual = execute(pages, page_size)
        assert np.isfinite(actual).all()
        assert np.array_equal(ref, actual), (width, page_size, window, splits)
        cases += 1
        outputs += actual.size
    return dict(compared_shapes=cases, matching_output_values=outputs, cpu=True)


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
