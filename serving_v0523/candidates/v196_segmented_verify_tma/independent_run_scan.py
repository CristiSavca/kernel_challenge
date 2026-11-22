"""Execute the two production Triton mask helpers with checked CPU pointers and exact-width words."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

STAGE = Path(sys.argv[1])
path = STAGE / "python/sglang/kernels/ops/attention/paged_verify_attention.py"
tree = ast.parse(path.read_text())
stats = Counter()
LIMIT = 33319


class Vector(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


class Pointer:
    def __init__(self, values, offset=0):
        self.values = values
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset+offset)


class TL:
    uint32 = np.uint32
    arange = staticmethod(lambda a,b: np.arange(a,b, dtype=np.int32).view(Vector))
    full = staticmethod(lambda shape,value,dtype: np.full(shape,value,dtype=dtype).view(Vector))
    where = staticmethod(np.where)
    min = staticmethod(np.min)
    minimum = staticmethod(np.minimum)

    @staticmethod
    def sum(a, axis):
        return a.sum(axis=axis,dtype=np.uint32)

    @staticmethod
    def load(p, mask, other):
        indices = p.offset[mask]
        assert all(i>=0 for i in indices) and all(i<len(p.values) for i in indices)
        out = np.full(len(mask),other,dtype=np.int32)
        out[mask] = p.values[indices]
        stats["checked_unmasked_lane_addresses"] += len(indices)
        return out


env = {"tl": TL}
for name in ("_tile_origin_mask","_next_tile_run"):
    node = next(n for n in tree.body if getattr(n,"name", None)==name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),env)


def check(flags, offset):
    n = len(flags)
    # Put distinct valid and invalid non-sentinel origins through every mask word.
    idx = np.arange(n)
    good, bad = np.where(idx%2,1,LIMIT), np.where(idx%3==0,-1,np.where(idx%3==1,0,LIMIT+1))
    values = np.r_[np.full(offset, LIMIT+1),np.where(flags,good,bad)].astype(np.int32)
    ptr = Pointer(values)
    expected = np.zeros(n,dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False,flags,False]))
    for start, end in zip(edges[::2],edges[1::2],strict=True):
        if end-start >= 4:
            expected[start:end] = True
    labels = []
    start = offset
    while start < offset+n:
        end,direct = env["_next_tile_run"](ptr,start,offset+n,LIMIT)
        end=int(end)
        assert start < end <= offset+n
        assert (expected[start-offset:end-offset]==direct).all()
        if labels:
            assert labels[-1] != direct
        labels.append(bool(direct))
        stats["direct_runs" if direct else "indexed_runs"] += 1
        start = end
    stats["sequences"] += 1
    stats["classified_tiles"] += n


for value in range(256):
    pattern = ((np.uint32(value)>>np.arange(8,dtype=np.uint32))&1).astype(bool)
    for offset,i in ((0,26),(5,29),(7,30),(3,31),(1,32),(13,57),(2,61),(0,94)):
        flags = np.zeros(i+12,dtype=bool)
        flags[i:i+8] = pattern
        check(flags,offset)
        check(~flags,offset)

rng=np.random.default_rng(397)
for n in (1,2,3,4,5,15,28,29,30,31,32,33,34,59,61,63,64,65,93,127,128,129,205,511,1024):
    for p in (0.,.04,.3,.6,.78,.96,1.):
        for _ in range(3):
            check(rng.random(n)<p,int(rng.integers(0,20)))
print(json.dumps(dict(candidate="V196", source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tests=stats, physical_h200=False,
    scope="Actual Triton helper function bodies, exact uint32 CPU reductions with checked reads, independently calculated maximal runs >=4. Eight-bit patterns embedded on each side of 29/32-lane cursor boundaries, unaligned request starts, random extents and extreme densities. Short intersections use a separate all-or-indexed branch in the full kernel."),indent=2))
