"""Trace production host dispatch and run its exact tile kernels through the Triton CPU interpreter."""
import ast
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import triton
import triton.language as tl
from kernel_check import extract
from interpret_decode_kernels import buf


class Tensor(np.ndarray):
    @property
    def is_cuda(self):
        # Exercise the device dispatch with interpreter-backed pointers.
        return True

    def numel(self):
        return self.size

    def is_contiguous(self):
        return self.flags.c_contiguous

    def new_empty(self, shape):
        return np.empty(shape, dtype=self.dtype).view(Tensor)

    def new_full(self, shape, value):
        return np.full(shape, value, dtype=self.dtype).view(Tensor)

    def clamp_max(self, other):
        return np.minimum(self, other)

    def sort(self, stable=False):
        calls["full_sort"] += 1
        order = np.argsort(np.asarray(self), kind="stable" if stable else "quicksort")
        return np.asarray(self)[order].view(Tensor), order.view(Tensor)


def tensor(value):
    return np.asarray(value).view(Tensor)


def search(left, right, out_int32):
    assert out_int32
    return np.searchsorted(left, right).astype(np.int32).view(Tensor)


torch = SimpleNamespace(
    int32=np.dtype("int32"), cumsum=lambda a, dim, dtype: np.cumsum(a, axis=dim, dtype=dtype).view(Tensor),
    arange=lambda n, device, dtype: np.arange(n, dtype=dtype).view(Tensor), searchsorted=search,
    cat=lambda seq: np.concatenate(seq).view(Tensor), div=lambda a,b,rounding_mode: (a // b).view(Tensor),
    where=lambda cond, yes, no: np.where(cond, yes, no).view(Tensor),
)


def host_func(path, name, scope):
    n=next(n for n in ast.parse(path.read_bytes()).body if getattr(n,"name",None)==name)
    exec(compile(ast.Module([n],[]),str(path),"exec"),scope)
    return scope[name]


class DeviceKernel:
    def __init__(self, kernel):
        self.kernel=kernel

    def __getitem__(self, grid):
        def invoke(*args):
            calls[self.kernel.__name__]+=1
            converted=[buf(a,tl.int32) if isinstance(a,Tensor) else a for a in args]
            self.kernel[grid](*converted)
        return invoke


if __name__=="__main__":
    source,wave=map(Path,sys.argv[1:3])
    names=["_count_tiles","_prefix_tiles","_wave_edges","_place_tiles"]
    rng=np.random.default_rng(3101)
    samples=[]
    with tempfile.TemporaryDirectory() as td:
        k=extract(source,set(names)|{"_counts_before"},Path(td),"production_wave_kernels")
        mod=ModuleType("sglang.srt.layers.muse_spark.humming_prompt_route_tiles")
        scope={"torch":torch,"triton":triton, **{n:DeviceKernel(getattr(k,n)) for n in names}}
        mod.tiled_wave_order=host_func(source,"tiled_wave_order",scope)
        plan=host_func(wave,"plan_waves",dict(torch=torch,triton=triton))
        with patch.dict(sys.modules,{mod.__name__:mod}):
            for tokens,topk,experts,dtype,stride in [
                    (1024,8,32,"int32",1),(3073,5,17,"int32",1),(116,8,32,"int32",1),
                    (7937,1,3,"int32",1),(7936,1,3,"int32",1),
                    (1024,8,32,"int64",1),(1024,8,65,"int32",1),(1024,8,32,"int32",2)]:
                arr=rng.integers(-experts,experts+3,(tokens,topk*stride),dtype=dtype)[:,::stride].view(Tensor)
                calls=Counter()
                cap=2*tokens-1
                sb, ss, edges, bounds=plan(arr,experts,cap)
                starts=((arr>=0)&(arr<experts)).sum(axis=1,dtype=np.int32)
                starts=np.cumsum(starts,dtype=np.int32)-starts
                waves=triton.cdiv(arr.size,cap)
                reference=np.where((arr>=0)&(arr<experts), np.minimum(starts//cap,waves-1)[:,None]*experts+arr,waves*experts)
                order=np.argsort(np.asarray(reference).ravel(),kind="stable")
                expected=np.asarray(reference).ravel()[order]
                assert np.array_equal(edges,np.searchsorted(expected,np.arange(waves*experts+1)))
                assert np.array_equal(bounds,np.r_[np.searchsorted(starts,np.arange(waves)*cap),tokens])
                tiled=32<=triton.cdiv(arr.size,256)<=2048 and dtype=="int32" and experts<=64 and stride==1
                if tiled:
                    live=order[:edges[-1]]
                    assert sb is None and np.array_equal(ss[:edges[-1]],arr.ravel()[live]*triton.next_power_of_2(arr.size)+live)
                    assert calls=={n:1 for n in names}
                else:
                    assert np.array_equal(sb,expected) and np.array_equal(ss,order) and calls=={"full_sort":1}
                samples.append(dict(tokens=tokens,topk=topk,experts=experts,dtype=dtype,stride=stride,tiled=tiled,
                                    executed=dict(calls),live=int(edges[-1])))
    print(json.dumps(dict(cases=samples,physical_H200=False),indent=2))
