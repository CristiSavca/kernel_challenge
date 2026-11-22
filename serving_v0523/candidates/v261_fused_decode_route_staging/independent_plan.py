"""Run the frozen V256 and V261 public grouped entry using V66's checked pointer/tensor shims."""
import ast
import hashlib
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
V256 = HERE.parent / "v256_window_backed_mixed_serving"
V66 = HERE.parent / "v66_humming_grouped_decode" / "cpu_check.py"
ADAPTER = "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(candidate):
    patch = (candidate / "patch.diff").read_bytes()
    receipt = json.loads((candidate / "validation.json").read_bytes())
    assert digest(patch) == receipt["patch_sha256"]
    part = next(p for p in patch.decode().split("diff --git ")
                if p.startswith(f"a/{ADAPTER} b/{ADAPTER}\n"))
    lines = part.splitlines()
    assert lines[1] == "new file mode 100644" and lines[3] == "--- /dev/null"
    start = next(i for i, line in enumerate(lines) if line.startswith("@@ "))
    length = int(re.fullmatch(r"@@ -0,0 \+1,(\d+) @@", lines[start]).group(1))
    body = lines[start + 1:]
    assert len(body) == length and all(line.startswith("+") for line in body)
    return "\n".join(line[1:] for line in body) + "\n"


def shims():
    # The original V66 file starts running its checks after these reusable definitions.
    declarations = []
    for node in ast.parse(V66.read_bytes()).body:
        if isinstance(node, ast.If):
            break
        declarations.append(node)
    module = {}
    exec(compile(ast.Module(declarations, type_ignores=[]), str(V66), "exec"), module)
    return module


host = shims()
tensor, tl, Pointer = (host[k] for k in ("tensor", "Language", "Pointer"))
tl.sum = staticmethod(lambda value, axis: tensor(np.asarray(value).sum(axis=axis, dtype=value.dtype)))
padded = lambda n: 1 << (n - 1).bit_length()
torch = SimpleNamespace(bfloat16=np.dtype(np.float32), int32=np.int32, int64=np.int64,
                        empty_like=lambda v, dtype=None: tensor(np.empty_like(v, dtype=dtype)),
                        compiler=SimpleNamespace(is_compiling=lambda: False))


class Kernel:
    def __init__(self, body, state):
        self.body = body
        self.state = state

    def __call__(self, *args, **kw):
        return self.body(*args, **kw)

    def __getitem__(self, grid):
        def run(*args, **kw):
            name = self.body.__name__
            self.state.calls.append(name)
            if name == "_stage_grouped_decode":
                self.state.warps = kw["num_warps"]
            kw = {k:v for k,v in kw.items() if k != "num_warps"}
            pointers = [Pointer(x) if isinstance(x, host["Tensor"]) else x for x in args]
            tl.grid = grid[0]
            # Executing metadata last catches a dependency on its completion.
            order = reversed(range(tl.grid)) if self.state.reverse else range(tl.grid)
            for tl.pid in order:
                self.body(*pointers, **kw)
            if name != "_gather_grouped_decode":
                self.state.metadata = [np.asarray(x).copy() for x in args[2:6]]
        return run


def scope(text, state):
    names = {"_decode_route_expert", "_plan_grouped_decode", "_stage_grouped_decode",
             "_gather_grouped_decode", "humming_decode_eligible", "humming_grouped_decode_outputs"}
    functions = [node for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list.clear()
    functions = ast.parse("from __future__ import annotations").body + functions
    ns = dict(torch=torch, triton=SimpleNamespace(next_power_of_2=padded), tl=tl,
              _GROUPED_COMPUTE="grouped", _MAX_DECODE_SLOTS=2048)
    exec(compile(ast.Module(functions, type_ignores=[]), ADAPTER, "exec"), ns)
    for name in ("_plan_grouped_decode", "_stage_grouped_decode", "_gather_grouped_decode"):
        if name in ns:
            ns[name] = Kernel(ns[name], state)
    # Keep the real public allocation and W13/W2 calls. Numerical GEMMs are
    # unrelated to staging. The fixtures copy the sentinel and live zone through.
    def activate(gate, result, keys, offsets, scales, span):
        first = int(offsets[0])
        result[first:] = gate[first:, :result.shape[1]]
    ns["_expert_scaled_silu_module"] = lambda: SimpleNamespace(run=activate)
    return ns


class Humming:
    @staticmethod
    def forward_layer(layer, *, inputs, outputs, expert_layout, sublayer_name, **kwargs):
        first = int(expert_layout[0])
        if sublayer_name == "w13":
            layer.staged = np.asarray(inputs).copy()
            assert np.isnan(inputs[:first]).all()
            width = inputs.shape[1]
            outputs[first:, :width] = inputs[first:]
            outputs[first:, width:] = inputs[first:]
        else:
            outputs[first:] = inputs[first:]
        layer.phases.append(sublayer_name)


vendor = ModuleType("humming.layer")
vendor.HummingMethod = Humming
sys.modules[vendor.__name__] = vendor


def expected(ids, mapping, global_count, experts, hidden):
    n, topk = ids.shape
    slots, span = ids.size, padded(ids.size)
    loc = []
    for raw in ids.ravel():
        i = int(raw)
        loc.append((int(mapping[i]) if mapping is not None else i) if 0 <= i < global_count else -1)
    live = sorted((e, j) for j,e in enumerate(loc) if 0 <= e < experts)
    begin = slots - len(live)
    remap = np.full(slots, -1, np.int32)
    staged = np.full((slots, hidden.shape[1]), np.nan, hidden.dtype)
    for rank, (_,j) in enumerate(live, begin):
        remap[j] = rank
        staged[rank] = hidden[j // topk]
    counts = Counter(e for e,_ in live)
    offsets = [begin]
    for e in range(experts):
        offsets.append(offsets[-1] + counts[e])
    # Simulate stock padded (bucket,slot) sort separately from the suffix oracle.
    buckets = [(e+1 if 0 <= e < experts else 0) for e in loc] + [0]*(span-slots)
    keys = sorted(bucket * span + j for j,bucket in enumerate(buckets))[span-slots:]
    return [np.array(loc, np.int32).reshape(ids.shape), np.array(offsets, np.int32),
            np.array(keys, np.int32), remap.reshape(ids.shape)], staged


def fixtures():
    rng = np.random.default_rng(26166)
    shapes = [(1,1,1,1024), (3,3,7,1025), (9,7,15,1153), (8,8,32,4096),
              (13,8,31,1024), (17,7,9,1027), (19,8,32,4096), (31,8,32,1024),
              (51,5,7,2049), (32,8,32,4096), (1,1,64,1024), (257,1,32,1024),
              (512,4,32,64), (2,8,32,1023), (4,3,7,4097)]
    for index,(n,k,experts,width) in enumerate(shapes):
        g = 8*experts
        for direction,mapped in itertools.product((False, True), repeat=2):
            mapbuf = None
            if mapped:
                # An arbitrary non-monotone global->local assignment, plus
                # unmapped ids, invalid local values and two ids sharing a bucket.
                mapbuf = tensor(np.full(g,-1,np.int32))
                perm = rng.permutation(g)[:experts]
                mapbuf[perm] = np.arange(experts)
                if g > 4:
                    mapbuf[0], mapbuf[1], mapbuf[2], mapbuf[3] = -9, experts, 0, 0
            for name in ("mixed", "remote", "collision"):
                dtype = np.int64 if (index + mapped + direction) % 2 else np.int32
                ids = rng.integers(-2,g+2,(n,k),dtype=dtype)
                if name == "remote":
                    ids[:] = -1
                    ids.flat[::4] = g+123
                elif name == "collision":
                    ids.flat[:] = np.resize([2,3,2,3,0,1] if mapped else [0,0,experts-1],n*k)
                else:
                    ids.flat[::7] = -7
                    ids.flat[1::11] = np.iinfo(dtype).max
                    ids.flat[2::13] = np.iinfo(dtype).min
                # Distinct row and column signatures, exactly representable in float32.
                h = tensor(np.arange(n*width, dtype=np.float32).reshape(n,width))
                yield tensor(ids), mapbuf, experts, h, direction, name


def main():
    sources = [source(path) for path in (V256,HERE)]
    states = [SimpleNamespace() for _ in sources]
    versions = [scope(text, state) for text,state in zip(sources, states)]
    totals = Counter()
    for ids, mapping, experts, h, reverse, pattern in fixtures():
        g = 8*experts
        oracle, staged = expected(ids, mapping, g, experts, h)
        revised = ids.size <= 256 and 1024 <= h.shape[1] <= 4096
        for i,(ns,state) in enumerate(zip(versions,states)):
            layer = SimpleNamespace(num_local_experts=experts,global_num_experts=g,
                intermediate_size_per_partition=h.shape[1],_expert_map=mapping,phases=[],
                humming_decode_tunings={n:[(2048,"fixed")] for n in ("w13","w2")},w13_global_scale=None)
            assert ns["humming_decode_eligible"](layer,h,ids)
            state.reverse, state.calls = reverse, []
            values, row_map, local = ns["humming_grouped_decode_outputs"](layer,h,ids)
            assert layer.phases == ["w13","w2"]
            assert state.calls == (["_stage_grouped_decode"] if i and revised else
                                   ["_plan_grouped_decode", "_gather_grouped_decode"])
            if i and revised:
                expected_warps = 8 if padded(ids.size) == 256 and h.shape[1] > 2048 else 4
                assert state.warps == expected_warps
                totals[f"fused_{expected_warps}_warps"] += 1
            assert all(np.array_equal(x,y) for x,y in zip(state.metadata,oracle))
            assert np.array_equal(row_map,oracle[3]) and np.array_equal(local,oracle[0])
            assert np.array_equal(layer.staged,staged,equal_nan=True)
            assert np.array_equal(values,staged,equal_nan=True)
            # A non-contiguous argument stays on the normal generic fallback.
            if h.shape[0]>1:
                assert not ns["humming_decode_eligible"](layer,h[:,::-1],ids)
                totals["noncontiguous_hidden_rejections"]+=1
            if ids.size>1:
                strided = tensor(np.empty((ids.size,2),np.int32)[:,0].reshape(ids.shape))
                assert not ns["humming_decode_eligible"](layer,h,strided)
                totals["noncontiguous_id_rejections"]+=1
        totals["parent_candidate_case_pairs"]+=1
        totals["fused_pairs" if revised else "fallback_pairs"]+=1
        totals["metadata_words_per_version"]+=sum(a.size for a in oracle)
        totals["staged_words_per_version"]+=staged.size
        totals["reverse_order_pairs"]+=int(reverse)
        totals["pair_pattern_"+pattern]+=1
    print(json.dumps(dict(full_patch_sha256=[digest((p/"patch.diff").read_bytes()) for p in (V256,HERE)],
            adapter_sha256=[digest(txt.encode()) for txt in sources], v66_shims_sha256=digest(V66.read_bytes()),
            numpy=np.__version__, **totals,
            scope="Real source Python entry, route and gather bodies on NumPy V66 pointer shims; downstream matrices are identity fixtures. Inactive sentinel values compared before and after W13/W2. No H200 execution."),indent=2))


if __name__ == "__main__":
    main()
