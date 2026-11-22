"""Run the exported device planner and packer against a stable full sort, or assemble their Hopper kernels."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl


def extract(path, names, out, tag):
    lines = path.read_text().splitlines(True)
    nodes = [n for n in ast.parse(path.read_bytes()).body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    text = "import triton\nimport triton.language as tl\n" + "\n".join(
        "".join(lines[n.decorator_list[0].lineno - 1:n.end_lineno]) for n in nodes)
    file = out / (tag + ".py")
    file.write_text(text)
    spec = importlib.util.spec_from_file_location(tag, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


def interpret(k, pack):
    from interpret_decode_kernels import buf

    def ptr(arr):
        return buf(arr, tl.int32)

    rng = np.random.default_rng(310)
    cases = []
    specs = [
        (37, 3, 1, 9, "empty"), (107, 5, 3, 104, "all"),
        (513, 7, 17, 399, "trailing"), (897, 4, 64, 225, "boundary"),
        (1201, 5, 7, 499, "random"), (8192, 8, 32, 12281, "random"),
        (16384, 8, 32, 16896, "uniform-ep"), (16384, 8, 32, 16896, "all"),
        (16384, 8, 32, 16896, "hot"), (16384, 8, 32, 16896, "trailing"),
    ]
    guard = -876543210
    for tokens, topk, experts, capacity, layout in specs:
        slots = tokens * topk
        waves = triton.cdiv(slots, capacity)
        tile = 256
        tiles = triton.cdiv(slots, tile)
        ids = rng.integers(-3, experts + 3, size=(tokens, topk), dtype=np.int32)
        if layout == "empty":
            ids.fill(-1)
        elif layout == "all":
            ids %= experts
        elif layout == "hot":
            ids.fill(experts - 1)
        elif layout == "trailing":
            ids[tokens // 3:] = -1
        elif layout == "boundary":
            ids.fill(-1)
            ids[:, 0] = 0
            ids[capacity-1:capacity+10] = -1
            ids[-12:] = -1
        elif layout == "uniform-ep":
            g = np.array([rng.choice(256, topk, replace=False) for _ in range(tokens)], dtype=np.int32)
            ids = np.where(g < experts, g, -1)
        live = (ids >= 0) & (ids < experts)
        counts = live.sum(axis=1, dtype=np.int32)
        starts = counts.cumsum(dtype=np.int32) - counts
        bounds = np.r_[starts.searchsorted(np.arange(waves) * capacity), tokens].astype(np.int32)
        bins = np.where(live, np.minimum(starts // capacity, waves - 1)[:, None] * experts + ids, waves * experts)
        order = np.argsort(bins.ravel(), kind="stable")
        edges = np.searchsorted(bins.ravel()[order], np.arange(waves * experts + 1))
        span = triton.next_power_of_2(slots)
        buffers = []
        def arena(n):
            raw = np.full(n + 2, guard, dtype=np.int32)
            buffers.append(raw)
            return raw[1:-1]
        prefix = arena(experts * (tiles + 1))
        before = arena(waves * experts)
        actual_edges = arena(waves * experts + 1)
        keys = arena(slots)
        block = triton.next_power_of_2(experts + 1)
        k._count_tiles[(tiles,)](ptr(ids), ptr(prefix), slots, tiles, experts, block, tile)
        k._prefix_tiles[(experts,)](ptr(prefix), tiles, triton.next_power_of_2(tiles))
        k._wave_edges[(waves,)](ptr(ids), ptr(prefix), ptr(bounds), ptr(before), ptr(actual_edges),
                               tiles, experts, topk, waves, block, tile)
        np.testing.assert_array_equal(edges, actual_edges)
        k._place_tiles[(tiles,)](ptr(ids), ptr(starts), ptr(prefix), ptr(before), ptr(actual_edges), ptr(keys),
                                slots, tiles, capacity, topk, experts, waves, block, tile, span)
        live_slots = order[:edges[-1]]
        expected_keys = ids.ravel()[live_slots] * span + live_slots
        np.testing.assert_array_equal(keys[:edges[-1]], expected_keys)
        assert (keys[edges[-1]:] == guard).all()
        checked_words = 0
        if slots <= 4096:
            rows = min(capacity + topk - 1, slots)
            inputs = np.arange(tokens * 3, dtype=np.int32).reshape(tokens, 3)
            for plain in (False, True):
                for wave in range(waves):
                    begin, end = edges[wave*experts], edges[(wave+1)*experts]
                    packed, row_keys, row_map = arena(rows * 3), arena(rows), arena(slots)
                    pack._pack_wave[(min(rows, 32),)](
                        ptr(inputs), ptr(keys), ptr(keys), ptr(actual_edges[wave*experts:]),
                        ptr(packed), ptr(row_keys), ptr(row_map), rows, experts, topk, 3, 4, span, plain, True)
                    selected = live_slots[begin:end]
                    offset = rows - len(selected)
                    np.testing.assert_array_equal(packed.reshape(rows, 3)[offset:], inputs[selected // topk])
                    np.testing.assert_array_equal(row_keys[offset:], ids.ravel()[selected] if plain else expected_keys[begin:end] + span)
                    want = np.full(slots, guard, dtype=np.int32)
                    want[selected] = np.arange(offset, rows)
                    np.testing.assert_array_equal(row_map, want)
                    assert (row_keys[:offset] == guard).all() and (packed[:offset*3] == guard).all()
                    checked_words += len(selected)*3
        assert all((raw[[0, -1]] == guard).all() for raw in buffers)
        cases.append(dict(tokens=tokens, topk=topk, experts=experts, layout=layout, slots=slots,
                          live_rows=int(edges[-1]), edges=len(edges), packed_payload_words=checked_words,
                          guard_words=len(buffers)*2))
    return dict(cases=cases, slots=sum(c['slots'] for c in cases),
                live_keys=sum(c['live_rows'] for c in cases), packed_words=sum(c['packed_payload_words'] for c in cases))


def compile_sm90(k, p):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    rows = []
    for experts, topk, waves in [(1,1,1), (3,3,9), (32,8,8), (64,7,16)]:
        bins=triton.next_power_of_2(experts+1)
        common=dict(EXPERTS=experts,BINS=bins,TILE=256)
        specs = [
            (k._count_tiles, ["*i32"]*2+["i32"]*2, common, 4),
            (k._prefix_tiles, ["*i32","i32"], dict(WIDTH=512 if experts<64 else 2048),4),
            (k._wave_edges, ["*i32"]*5+["i32"], dict(**common,TOPK=topk,WAVES=waves),4),
            (k._place_tiles, ["*i32"]*6+["i32"]*3,dict(**common,TOPK=topk,WAVES=waves,SPAN=131072),4)]
        if experts==32:
            for plain in (False,True):
                for packed in (False,True):
                    specs.append((p._pack_wave, ["*bf16","*i32",("*i32" if packed else "*i64"),"*i32","*bf16","*i32","*i32"],
                        dict(ROWS=16903,EXPERTS=experts, TOPK=topk, HIDDEN=4096,WIDTH=4096,SPAN=131072,
                             PLAIN_IDS=plain,PACKED_ORDER=packed),8))
        for f,types,constants,warps in specs:
            sig = dict(zip(f.arg_names, types + ["constexpr"]*len(constants), strict=True))
            obj=triton.compile(ASTSource(f,sig,constexprs=constants),target=GPUTarget("cuda",90,32),
                               options={"num_warps":warps})
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/"tmp.ptx";path.write_text(obj.asm["ptx"])
                log=subprocess.run([assembler,"-v","-arch=sm_90a",path,"-o",path.with_suffix(".cubin")],check=True,
                                   capture_output=True,text=True).stderr
            spills=tuple(map(int,re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",log).groups()))
            rows.append(dict(kernel=f.__name__,constants=constants,registers=int(re.search(r"Used (\d+) registers",log)[1]),
                             spill_bytes=spills,shared_bytes=obj.metadata.shared))
    return dict(specializations=rows)

if __name__=="__main__":
    source, packing=map(Path,sys.argv[2:4])
    with tempfile.TemporaryDirectory() as td:
        k=extract(source, {"_count_tiles","_prefix_tiles","_counts_before","_wave_edges","_place_tiles"},Path(td),"tiles_under_test")
        p=extract(packing, {"_pack_wave"},Path(td),"packer_under_test")
        result=interpret(k,p) if sys.argv[1]=="interpret" else compile_sm90(k,p)
    print(json.dumps(dict(triton=triton.__version__,tiled_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          wave_source_sha256=hashlib.sha256(packing.read_bytes()).hexdigest(),physical_H200=False,**result),indent=2))
