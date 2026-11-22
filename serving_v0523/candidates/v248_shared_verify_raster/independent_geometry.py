"""Execute both Triton planners and the production interval branches on CPU arrays."""
import argparse
import ast
import copy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from independent_reach import HERE, KERNEL, PARENT, kernel_source


class Vector(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype, copy=False)


def vector(value):
    return np.asarray(value).view(Vector)


class Pointer:
    def __init__(self, values, offset=0):
        self.values = values
        self.offset = offset

    def __add__(self, step):
        return Pointer(self.values, self.offset + step)


class TritonSubset:
    int32 = np.int32
    constexpr = None
    arange = staticmethod(lambda a, b: vector(np.arange(a, b, dtype=np.int32)))
    full = staticmethod(lambda shape, value, dtype: vector(np.full(shape, value, dtype=dtype)))
    where = staticmethod(lambda c, a, b: vector(np.where(c, a, b)))
    cdiv = staticmethod(lambda a, b: vector((a + b - 1) // b))
    cumsum = staticmethod(lambda x: vector(np.cumsum(x, dtype=x.dtype)))
    sum = staticmethod(lambda x, axis=None: vector(np.sum(x, axis=axis, dtype=x.dtype)))
    maximum = staticmethod(lambda a, b: vector(np.maximum(a, b)))
    minimum = staticmethod(lambda a, b: vector(np.minimum(a, b)))

    def program_id(self, dim):
        return self.pid[dim]

    def load(self, pointer, mask=True, other=0):
        ids, mask = np.broadcast_arrays(pointer.offset, mask)
        assert np.all((ids[mask] >= 0) & (ids[mask] < len(pointer.values)))
        return vector(np.where(mask, pointer.values[np.where(mask, ids, 0)], other))

    def store(self, pointer, values, mask=True):
        ids, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
        assert np.all((ids[mask] >= 0) & (ids[mask] < len(pointer.values)))
        pointer.values[ids[mask]] = values[mask]


def contains_assignment(node, var):
    return isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == var


def native_functions(raw):
    tl = TritonSubset()
    ns = {"tl": tl}
    functions = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    for name in ("_tile_geometry", "_plan", "_build_verify_raster", "_partition_attention", "_merge_partitions"):
        if name not in functions:
            continue
        node = copy.deepcopy(functions[name])
        node.decorator_list = []
        if name == "_partition_attention":
            original_loop = next(n for n in node.body if isinstance(n, ast.While))
            part = next(n for n in original_loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "current < req_end")
            end = next(i for i, n in enumerate(part.body) if contains_assignment(n, "tile_end"))
            original_math = part.body[end + 1:]
            assert contains_assignment(original_math[-1], "current") and ast.unparse(original_math[-1].value) == "segment_end"
            part.body = part.body[:end + 1] + ast.parse(
                "segments.append(tuple(map(int, (request, req_start, req_end, tile_start, tile_end))))").body + [original_math[-1]]
            begin = next(i for i,n in enumerate(node.body) if contains_assignment(n, "packed_row"))
            # Only Q/K/V arithmetic and CTA vector setup are omitted. Preserve
            # each branch deciding which tiles, partitions and requests it visits.
            node.body = ast.parse("segments = []").body + node.body[:begin] + [original_loop] + ast.parse("return segments").body
        elif name == "_merge_partitions":
            end = next(i for i,n in enumerate(node.body) if contains_assignment(n, "dims"))
            node.body = node.body[:end] + ast.parse("return tuple(map(int, (req_start, req_end, split_begin, split_end)))").body
        ast.fix_missing_locations(node)
        exec("from __future__ import annotations\n" + ast.unparse(node), ns)
    return tl, ns


def call(fn, **kwargs):
    # Unused Q/K/V arguments from the original signature are placeholders.
    defaults = {name: None for name in fn.__code__.co_varnames[:fn.__code__.co_argcount]}
    defaults.update(kwargs)
    return fn(**defaults)


def base_inputs(lengths, qry, window, partitions):
    cuq = np.insert(np.cumsum(qry, dtype=np.int32), 0, 0)
    return dict(Lengths=Pointer(lengths), CuQ=Pointer(cuq), WINDOW_LEFT=window,
        TILE_N=64 if window >= 0 else 128, PARTITIONS=partitions, B=len(lengths),
        BLOCK_B=1 << (len(lengths)-1).bit_length(), ROWS=4, GQA=8)


def produce(tl, ns, inputs, plan):
    kwargs = {key:inputs[key] for key in ("Lengths", "CuQ", "B", "PARTITIONS", "WINDOW_LEFT", "TILE_N", "BLOCK_B")}
    for pid in range((inputs["PARTITIONS"] + 15) // 16):
        tl.pid = (pid, 0, 0)
        ns["_build_verify_raster"](Plan=Pointer(plan), BLOCK_P=16, **kwargs)


def compare(old, new, inputs, plan, *, exhaustive):
    p, b = inputs["PARTITIONS"], inputs["B"]
    # Scalar, independently derived comparison for every stored cell, including
    # equality at exact partition ends and trailing zero-length requests.
    lengths = inputs["Lengths"].values.astype(np.int64)
    qlen = np.diff(inputs["CuQ"].values)
    tile, win = inputs["TILE_N"], inputs["WINDOW_LEFT"]
    first = np.maximum(lengths-qlen-win, 0)//tile if win >= 0 else np.zeros(b, dtype=np.int64)
    counts = np.where(qlen > 0, (lengths+tile-1)//tile-first, 0)
    offsets = np.insert(np.cumsum(counts), 0, 0)
    width = max(1, (int(offsets[-1]) + p-1)//p)
    owners = np.searchsorted(offsets[1:], np.arange(p)*width, side="right")
    expected = np.concatenate([offsets, first, [width], owners])
    assert np.array_equal(plan, expected)
    axes = (range(p) if exhaustive else sorted({0, 1, p//2, p-2, p-1}))
    counts_out = dict(plan_words=plan.size, partition_pairs=0, nonempty_intersections=0, merge_pairs=0)
    for pid in axes:
        outputs = []
        for i,(tl, ns) in enumerate((old, new)):
            tl.pid = (pid, 0, 0)
            outputs.append(call(ns["_partition_attention"], **{k:v for k,v in inputs.items() if k not in ("ROWS",)},
                **({"Plan":Pointer(plan)} if i else {})))
        assert outputs[0] == outputs[1]
        for r, begin, end, first_tile, last_tile in outputs[1]:
            assert (begin, end) == (offsets[r], offsets[r+1])
            assert (first_tile, last_tile) == (first[r]+max(begin,pid*width)-begin,
                first[r]+min(end,(pid+1)*width)-begin)
        counts_out["partition_pairs"] += 1
        counts_out["nonempty_intersections"] += len(outputs[1])
    for r in range(b):
        for cta in sorted({0, max(0, (int(qlen[r])*8+3)//4-1), (int(qlen[r])*8+3)//4}):
            outputs = []
            for i,(tl, ns) in enumerate((old, new)):
                tl.pid = (r, 0, cta)
                outputs.append(call(ns["_merge_partitions"], **inputs, **({"Plan":Pointer(plan)} if i else {})))
            assert outputs[0] == outputs[1]
            if cta*4 >= qlen[r]*8:
                assert outputs[1] is None
            else:
                assert outputs[1] == (offsets[r], offsets[r+1], offsets[r]//width, (offsets[r+1]-1)//width+1)
            counts_out["merge_pairs"] += 1
    return counts_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", type=Path)
    args = parser.parse_args()
    manifest = HERE/"validation.json"
    if manifest.exists():
        raw = kernel_source(HERE)
        if args.stage:
            assert (args.stage/KERNEL).read_text() == raw
    else:
        assert args.stage is not None
        raw = (args.stage/KERNEL).read_text()
    before = kernel_source(PARENT)
    old, new = native_functions(before), native_functions(raw)
    rng = np.random.default_rng(248)
    totals = dict(plan_words=0, partition_pairs=0, nonempty_intersections=0, merge_pairs=0)
    cases = 0
    lengths_choices = np.array([0,1,7,8,63,64,65,127,128,129,2047,2048,2049,65536,80000,131072], dtype=np.int32)
    for b in (1,2,3,4,5,7,8,10,13,16,22,31,32,63,64):
        for window, p in ((-1,132),(2047,66),(-1,33),(2047,17)):
            reused = np.full(2*b+2+p, -77, dtype=np.int32)
            for mode in ("empty", "mixed", "ends", "uniform"):
                lengths = rng.choice(lengths_choices, size=b)
                if mode == "empty":
                    q = np.zeros(b, np.int32)
                elif mode == "mixed":
                    q = np.minimum(lengths, rng.integers(0, 9, size=b, dtype=np.int32))
                elif mode == "ends":
                    q = np.zeros(b, np.int32)
                    q[[0, b-1]] = 8
                    lengths[[0,b-1]] = np.maximum(lengths[[0,b-1]], 8)
                else:
                    q = np.full(b, 8, np.int32)
                    lengths = np.maximum(lengths, q)
                inp = base_inputs(lengths, q, window, p)
                produce(*new, inp, reused)
                result = compare(old,new,inp,reused,exhaustive=b in (2,7,32) and p in (66,132))
                for k,n in result.items():
                    totals[k] += n
                cases += 1
    output = dict(parent_sha256=json.loads((PARENT/"validation.json").read_bytes())["patch_sha256"],
        parent_kernel_sha256=sha256(before.encode()).hexdigest(), kernel_sha256=sha256(raw.encode()).hexdigest(),
        full_patch_sha256=json.loads(manifest.read_bytes())["patch_sha256"] if manifest.exists() else None,
        tested_batches=[1,2,3,4,5,7,8,10,13,16,22,31,32,63,64],
        paired_shapes=cases, **totals, empty_trailing_and_interior_queries=True, same_buffer_rebuilt_four_times=True,
        oracle="Independent scalar tile count, prefix offsets and right-inclusive owner search, plus production parent/child partition and merge interval branches executed over CPU arrays. Q/K/V tensor-core bodies excluded. Batch one exercises geometry, not serving selection.")
    (HERE/"independent_geometry.json").write_text(json.dumps(output,indent=2)+"\n")
    print(cases, totals)


if __name__ == "__main__":
    main()
