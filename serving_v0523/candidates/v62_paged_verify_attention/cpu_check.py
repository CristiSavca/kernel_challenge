"""Execute the Triton attention bodies with checked NumPy-backed CPU pointers."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class BF16:
    pass


def round_bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


class Vector(np.ndarray):
    def to(self, dtype):
        if dtype is BF16:
            return round_bf16(self).view(Vector)
        return self.astype(dtype)


class Pointer:
    def __init__(self, array, offsets=0):
        self.data = array.reshape(-1)
        self.offsets = offsets

    def __add__(self, offsets):
        return Pointer(self.data, self.offsets + offsets)


def bounds(ptr, mask):
    offsets, mask = np.broadcast_arrays(ptr.offsets, np.asarray(mask, dtype=bool))
    chosen = offsets[mask]
    assert np.all((chosen >= 0) & (chosen < ptr.data.size))
    return offsets, mask, chosen


def load(ptr, mask=True, other=0):
    offsets, mask, chosen = bounds(ptr, mask)
    result = np.full(offsets.shape, other, dtype=ptr.data.dtype)
    result[mask] = ptr.data[chosen]
    return result.view(Vector)


def store(ptr, values):
    offsets, mask, chosen = bounds(ptr, True)
    assert np.all(np.isnan(ptr.data[chosen])), "Two programs wrote the same index"
    ptr.data[chosen] = np.broadcast_to(values, offsets.shape)[mask]


def dot(a, b, acc=0, **_):
    return (a @ b + acc).view(Vector)


coords = (0, 0, 0)
ceil_div = lambda r, d: (r + d - 1) // d
tl = SimpleNamespace(
    constexpr=int,
    int32=np.int32,
    int64=np.int64,
    float32=np.float32,
    bfloat16=BF16,
    arange=lambda a, b: np.arange(a, b).view(Vector),
    program_id=lambda axis: coords[axis],
    dot=dot,
    load=load,
    store=store,
    range=lambda a, b, **_: range(a, b),
    cdiv=ceil_div,
    cumsum=np.cumsum,
    min=np.min,
    max=np.max,
    sum=np.sum,
    exp2=lambda x: np.exp2(x).view(Vector),
    log2=np.log2,
    where=np.where,
    maximum=np.maximum,
    minimum=np.minimum,
    full=np.full,
    zeros=lambda shape, dtype: np.zeros(shape, dtype).view(Vector),
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_root", type=Path, help="SGLang checkout with V62 applied")
source_root = parser.parse_args().source_root
file = source_root / "python/sglang/kernels/ops/attention/paged_verify_attention.py"
tree = ast.parse(file.read_text())
tree.body = [node for node in tree.body if getattr(node, "decorator_list", None)]
for n in tree.body:
    n.decorator_list = []
ns = {"tl": tl}
exec(compile(tree, str(file), "exec"), ns)
run_attention = ns["_partition_attention"]
merge = ns["_merge_partitions"]
rng = np.random.default_rng(6222026)


def page_counts(lengths, q_lens, window, tile_n):
    first = np.maximum(0, lengths - q_lens - window) // tile_n if window >= 0 else 0
    return np.where(q_lens > 0, ceil_div(lengths, tile_n) - first, 0)


tested = 0
max_relative_error = 0.0
for case in range(110):
    batch = int(rng.integers(1, 7))
    q_max, groups = [(2, 8), (4, 4), (8, 2), (8, 8), (16, 4)][case % 5]
    heads, dim = case % 3 + 1, 64
    page_size = [1, 16, 64, 128, 160][case % 5]
    window = [-1, 0, 7, 63, 127, 511][case % 6]
    tile_n = 128 if window == -1 else 64
    tile_m = q_max * groups
    partitions = max(1, (case % 13) * 2 + 1)
    block_batch = 1 << (batch - 1).bit_length()
    lengths = rng.integers(0, 535, batch, dtype=np.int32)
    q_lens = np.minimum(lengths, rng.integers(0, q_max + 1, batch, dtype=np.int32))
    if case % 11 == 0:
        q_lens[:] = np.minimum(lengths, q_max)
    if case % 13 == 0:
        lengths[0] = q_lens[0] = 0
    if case % 17 == 0:
        lengths[0], q_lens[0] = 0, q_max
    cuq = np.concatenate((np.zeros(1, dtype=np.int32), q_lens.cumsum(dtype=np.int32)))
    page_bound = max(1, ceil_div(max(lengths), page_size))
    pages = (
        rng.permutation(batch * page_bound)
        .reshape((batch, page_bound))
        .astype(np.int32)
    )
    dims = (batch * page_bound, page_size, heads, dim)
    # Multiples of a sixteenth within [-0.5, 0.5] are exact E4M3 values.
    samples = lambda shape: rng.integers(-8, 9, shape).astype(np.float32) / 16
    k, v = samples(dims), samples(dims)
    q = samples((int(cuq[-1]), heads * groups, dim))
    if window >= 0:
        for b, qlen in enumerate(q_lens):
            expired = np.arange(max(0, lengths[b] - qlen - window))
            physical = pages[b][expired // page_size]
            within_page = expired % page_size
            k[physical, within_page] = np.nan
            v[physical, within_page] = np.nan
    out = np.full_like(q, np.nan)
    scale = float(1.0 / np.sqrt(dim))
    kscale = rng.uniform(0.5, 2.0, (batch, heads)).astype(np.float32)
    vscale = rng.uniform(0.5, 2.0, (batch, heads)).astype(np.float32)
    scales = case % 3 != 0
    partial = np.full(
        (partitions + batch - 1, heads, tile_m, dim), np.nan, dtype=np.float32
    )
    lse = np.full(partial.shape[:-1], np.nan, dtype=np.float32)
    for partition in range(partitions):
        for head in range(heads):
            coords = (partition, head, 0)
            strides = tuple(x // 4 for x in k.strides[:3])
            run_attention(
                Pointer(q),
                Pointer(k),
                Pointer(v),
                Pointer(pages),
                Pointer(lengths),
                Pointer(cuq),
                Pointer(kscale) if scales else None,
                Pointer(partial),
                Pointer(lse),
                page_bound,
                page_size,
                *strides,
                *strides,
                heads,
                1,
                batch,
                heads,
                groups,
                dim,
                scale * np.log2(np.e),
                window,
                q_max,
                tile_m,
                tile_n,
                partitions,
                block_batch,
            )
    for b in range(batch):
        for head in range(heads):
            for row in range(tile_m):
                coords = (b, head, row)
                merge(
                    Pointer(partial),
                    Pointer(lse),
                    Pointer(lengths),
                    Pointer(cuq),
                    Pointer(vscale) if scales else None,
                    Pointer(out),
                    heads,
                    1,
                    batch,
                    heads,
                    groups,
                    dim,
                    window,
                    tile_m,
                    tile_n,
                    partitions,
                    block_batch,
                    16,
                )
    reference = np.zeros_like(q)
    for b, qlen in enumerate(q_lens):
        for j in range(qlen):
            pos = lengths[b] - qlen + j
            if pos < 0:
                continue
            visible = np.arange(max(0, pos - window) if window >= 0 else 0, pos + 1)
            page_ids = pages[b][visible // page_size]
            offsets = visible % page_size
            for head in range(heads):
                q_rows = slice(head * groups, (head + 1) * groups)
                scores = (q[cuq[b] + j, q_rows] @ k[page_ids, offsets, head].T) * scale
                if scales:
                    scores *= kscale[b, head]
                p = np.exp(scores - scores.max(axis=1, keepdims=True))
                normalized = p / p.sum(axis=1, keepdims=True)
                result = normalized @ v[page_ids, offsets, head]
                reference[cuq[b] + j, q_rows] = result * (
                    vscale[b, head] if scales else 1
                )
    err = np.linalg.norm(out - reference) / max(float(np.linalg.norm(reference)), 1e-9)
    max_relative_error = max(max_relative_error, err)
    assert err < 0.002, (case, err, lengths, q_lens)
    tested += 1


for case in range(890):
    batch = int(rng.integers(1, 65))
    lengths = rng.integers(0, 131073, batch, dtype=np.int32)
    q_lens = np.minimum(lengths, rng.integers(0, 17, batch, dtype=np.int32))
    window = [-1, 0, 2047, 32767][case % 4]
    tile_n = [64, 128][case % 2]
    partitions = int(rng.integers(1, 265))
    block_batch = 1 << (batch - 1).bit_length()
    cuq = np.concatenate((np.zeros(1, dtype=np.int32), q_lens.cumsum(dtype=np.int32)))
    reqs, ends, total, width = ns["_plan"](
        Pointer(lengths),
        Pointer(cuq),
        batch,
        partitions,
        window,
        tile_n,
        block_batch,
    )
    expected = page_counts(lengths, q_lens, window, tile_n)
    assert np.array_equal(ends[:batch], np.cumsum(expected))
    slots = set()
    for b, c in enumerate(expected):
        if not c:
            continue
        start, stop = int(ends[b] - c), int(ends[b])
        cells = [split + b for split in range(start // width, (stop - 1) // width + 1)]
        assert max(cells) < partitions + batch - 1
        assert slots.isdisjoint(cells)
        slots.update(cells)

print(
    f"Passed {tested} exact-source attention cases, max relative L2 {max_relative_error:.6f}"
)
print("Passed 890 long-context split/merge ownership plans through 131,072 tokens")
