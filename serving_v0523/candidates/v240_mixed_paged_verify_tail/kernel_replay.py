"""Execute the indexed production split-KV launcher and both Triton bodies on CPU."""
import ast
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
KERNEL = "python/sglang/kernels/ops/attention/paged_verify_attention.py"


def build(source):
    # The V62 emulator checks every active read and every unique store and
    # rounds Triton's BF16 casts to nearest-even.
    original = next(HERE.parent.glob("v62_*/cpu_check.py"))
    tree = ast.parse(original.read_text())
    stop = next(n.lineno for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "parser")
    support = {}
    exec(compile(ast.Module([n for n in tree.body if n.lineno < stop], []), str(original), "exec"), support)
    module = ast.parse(source)
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    # The host BLAS can report spurious IEEE exceptions on strided Accelerate
    # matmul. Use an explicit finite FP32 contraction for the CPU emulator.
    def dot(a, b, acc=0, **kwargs):
        assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(acc).all()
        return (np.einsum("ij,jk->ik", a, b, optimize=False) + acc).view(support["Vector"])

    support["tl"].dot = dot
    device = dict(tl=support["tl"])
    for name in ("_tile_geometry", "_plan", "_indexed_tile", "_accumulate_tiles", "_partition_attention", "_merge_partitions"):
        n = functions[name]
        n.decorator_list = []
        exec(compile(ast.Module([n], []), KERNEL, "exec"), device)

    launches = []

    class Kernel:
        def __init__(self, name, writes):
            self.name, self.writes = name, writes

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, tuple(grid)))
                conv, changed = list(args), []
                for i, x in enumerate(args):
                    if not isinstance(x, torch.Tensor):
                        continue
                    if i in self.writes:
                        assert x.is_contiguous()
                        arr = np.full(x.shape, np.nan, dtype="f4")
                        changed.append((x,arr))
                    else:
                        size = 1 + sum((d - 1)*s for d,s in zip(x.shape,x.stride()))
                        flat = x.as_strided((size,), (1,), x.storage_offset())
                        arr = (flat.float() if x.is_floating_point() else flat).numpy()
                    conv[i] = support["Pointer"](arr)
                for idx in itertools.product(*(range(n) for n in grid)):
                    support["coords"] = idx
                    device[self.name](*conv)
                for tensor, array in changed:
                    tensor.copy_(torch.from_numpy(array))
            return launch

    ns = dict(torch=torch, math=math, TensorDescriptor=None,
              triton=SimpleNamespace(cdiv=support["ceil_div"], next_power_of_2=lambda n:1<<(n-1).bit_length()),
              _partition_attention=Kernel("_partition_attention",(7,8)), _merge_partitions=Kernel("_merge_partitions",(5,)))
    for name in ("_direct_tile_args","paged_verify_attention"):
        exec(compile(ast.Module([functions[name]], []), KERNEL, "exec"), ns)
    return ns["paged_verify_attention"], launches


def dense_reference(q, k, v, pages, lengths, cu, kscale, vscale, softmax_scale):
    heads, dim = k.shape[-2:]
    groups = q.shape[1] // heads
    expected = torch.empty_like(q,dtype=torch.float64)
    for i, n in enumerate(lengths.tolist()):
        lo, hi = cu[i:i+2].tolist()
        slots = pages[i,:n].long()
        keys, vals = [bank[:,0].view(torch.uint8)[slots].view(q.dtype).double() for bank in (k,v)]
        for head in range(heads):
            columns = slice(head*groups,(head+1)*groups)
            scores = (q[lo:hi,columns].double() @ keys[:,head].T) * softmax_scale
            if kscale is not None:
                scores *= kscale[i,head]
            mask = torch.arange(n)[None,None,:] > (n-(hi-lo)+torch.arange(hi-lo))[:,None,None]
            scores.masked_fill_(mask,-float("inf"))
            values = scores.softmax(-1) @ vals[:,head]
            if vscale is not None:
                values *= vscale[i,head]
            expected[lo:hi,columns] = values
    return expected
