"""Execute the actual V79 attention launcher and kernels using checked CPU pointers."""

import argparse
import inspect
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

CANDIDATES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CANDIDATES / "v70_dflash_selector_tree"))
import cpu_check as old


class Tensor:
    def __init__(self, value, dtype=None):
        self.value = np.ascontiguousarray(value)
        self.shape = self.value.shape
        self.dtype = dtype
        self.device = "cpu"

    def stride(self, dim=None):
        strides = tuple(s // self.value.itemsize for s in self.value.strides)
        return strides if dim is None else strides[dim]

    def is_contiguous(self):
        return True


class Launch:
    def __init__(self, kernel, pid):
        self.kernel = kernel
        self.pid = pid

    def __getitem__(self, grid):
        def run(*args, num_warps):
            args = [old.Pointer(v.value) if isinstance(v, Tensor) else v for v in args]
            for coord in np.ndindex(grid):
                self.pid[:] = list(coord) + [0] * (3 - len(coord))
                self.kernel(*args)
        return run


def launcher(path):
    pid, (_, _, part, merge) = old.program(
        path, ("_tile_geometry", "_plan", "_partition_attention", "_merge_partitions")
    )
    torch = SimpleNamespace(
        bfloat16="bf16", float8_e4m3fn="fp8", float32="fp32",
        empty=lambda shape, device, dtype: Tensor(np.full(shape, np.nan, np.float32), dtype),
        empty_like=lambda src, dtype: Tensor(np.full(src.shape, np.nan, np.float32), dtype),
    )
    triton = SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length(),
                             cdiv=lambda n, d: (n+d-1)//d)
    _, (run,) = old.program(path, ("paged_verify_attention",), dependencies={
        "_partition_attention": Launch(part, pid), "_merge_partitions": Launch(merge, pid),
        "torch": torch, "triton": triton, "math": math,
    })
    return run


def parent_attention():
    payload = next(CANDIDATES.glob("v76*/patch.diff")).read_text()
    piece = next(x for x in payload.split("diff --git ") if x.startswith(
        "a/python/sglang/kernels/ops/attention/paged_verify_attention.py "))
    lines = piece.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("@@ "))
    assert all(l.startswith("+") for l in lines[start + 1:])
    return SimpleNamespace(read_text=lambda: "\n".join(l[1:] for l in lines[start+1:]))


def attention_case(run, baseline, width, heads, groups, window, use_tree, padded, rng):
    bs, dim = 3, 64
    qlens = np.array([width, width//2, 0] if padded else [width]*bs)
    prefixes = np.array([63,128,194])
    lengths = prefixes + qlens
    cuq = np.concatenate(([0], np.cumsum(qlens)))
    # Grid-valued input is exactly representable in FP8 for these magnitudes.
    q = rng.integers(-4,5,size=(bs*width,heads*groups,dim)).astype(np.float32) / 4
    table = rng.permutation(bs * lengths.max()).reshape(bs, -1).astype(np.int32)
    k = rng.integers(-4,5,size=(table.size,1,heads,dim)).astype(np.float32) / 4
    v = rng.integers(-4,5,size=k.shape).astype(np.float32) / 4
    ks = rng.choice([.5,1,2], size=(bs,heads)).astype(np.float32)
    vs = rng.choice([.5,1,2], size=(bs,heads)).astype(np.float32)
    depths = np.zeros((bs,width),np.int32)
    masks = np.ones_like(depths)
    for req in range(bs):
        for j in range(1,width):
            parent = rng.integers(j) if use_tree else j-1
            depths[req,j] = depths[req,parent]+1
            masks[req,j] = masks[req,parent] | 1 << j

    args = (Tensor(q,"fp8"),Tensor(k,"fp8"),Tensor(v,"fp8"),Tensor(table),Tensor(lengths),Tensor(cuq))
    kwargs = dict(max_q=width, window_left=window, softmax_scale=.125, k_descale=Tensor(ks),
                  v_descale=Tensor(vs), out=None,num_sms=3,
                  tree_depths=Tensor(depths) if use_tree else None,
                  tree_ancestors=Tensor(masks) if use_tree else None)
    out = run(*args,**kwargs).value
    if width == 8:
        np.testing.assert_array_equal(out, baseline(*args,**kwargs).value)
    assert np.isnan(out[cuq[-1]:]).all()
    ref = np.empty_like(out[:cuq[-1]])
    for req in range(bs):
        prefix = prefixes[req]
        for j in range(qlens[req]):
            minpos = prefix+depths[req,j]-window if window>=0 else 0
            visible = list(range(max(0,min(minpos,prefix)),prefix))
            visible.extend(prefix+t for t in range(j+1)
                           if (masks[req,j] >> t) & 1 and prefix+depths[req,t]>=minpos)
            ids = table[req,visible]
            loc = cuq[req]+j
            for h in range(heads):
                g = slice(h*groups,(h+1)*groups)
                scores = q[loc,g] @ k[ids,0,h].T * (.125*ks[req,h])
                probs = np.exp(scores-scores.max(axis=1,keepdims=True))
                ref[loc,g] = (probs @ v[ids,0,h]) / probs.sum(axis=1,keepdims=True) * vs[req,h]
    err = np.linalg.norm(out[:len(ref)]-ref) / np.linalg.norm(ref)
    assert err < .004, (width,heads,groups,window,use_tree,padded,err)
    return float(err)


def widened_fixture(fn, changes, *args):
    """Extend the frozen V70 test parameters; target implementations stay intact."""
    source = inspect.getsource(fn)
    for previous, replacement in changes:
        assert source.count(previous) == 1
        source = source.replace(previous,replacement)
    scope = vars(old).copy()
    exec(compile(source, fn.__name__, "exec"),scope)
    return scope[fn.__name__](*args)


def main(root):
    ops = root / "python/sglang/kernels/ops"
    path = ops / "attention/paged_verify_attention.py"
    backend = root / "python/sglang/srt/layers/attention/flashattention_backend.py"
    run, base = launcher(path), launcher(parent_attention())
    rng = np.random.default_rng(731)
    errs = [attention_case(run,base,*args,rng)
            for args in itertools.product((8,12,16),(1,2),(4,8),(-1,7),(False,True),(False,True))]
    print(f"{len(errs)} launcher / dense attention cases; max L2 {max(errs):.6f}; "
          f"{len(errs)//3} width-8 V76 bitwise comparisons", flush=True)
    widths = [("for width in range(2, 9):", "for width in (12, 16):")]
    slot_cases = widths + [("np.full((bs, 53)", "np.full((bs, 39 + width)")]
    plans, walks = widened_fixture(old.verify_tree_kernels, slot_cases,
                                    ops / "speculative/dflash_tree.py", rng)
    resets = widened_fixture(old.verify_graph_stage, widths, backend)
    caps = widened_fixture(old.verify_backend_capability, [
        ("assert supported(backend, runner)", "for w in (8, 12, 16):\n"
         "        backend.speculative_num_draft_tokens = w\n        assert supported(backend,runner)\n"
         "    backend.speculative_num_draft_tokens = 8"),
        ('(backend, "speculative_num_draft_tokens", 16)', '(backend, "speculative_num_draft_tokens", 17)'),
        ('(attention, "tp_k_head_num", 1)', '(attention, "tp_q_head_num", 64)'),
    ], backend) + 2
    print(f"{plans} proposals, {walks} accept/slot cases, {resets} replay resets, {caps} capability cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root",type=Path)
    main(parser.parse_args().source_root)
