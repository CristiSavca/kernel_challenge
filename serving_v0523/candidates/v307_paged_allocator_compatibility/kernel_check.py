"""Compile the submitted allocator and interpret its actual slot stores on CPU buffers."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf

PAGES = (1, 3, 64, 160, 256)
CALLERS = ("paged", "unified_sub_pool")
DTYPES = {4: tl.int32, 8: tl.int64}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def launch_geometry(stage):
    """Read both actual Python-side call sites and execute their scalar arguments."""
    root = stage / "python/sglang/srt"
    filename = root / "utils/common.py"
    method = next(n for n in ast.parse(filename.read_bytes()).body if getattr(n, "name", "") == "next_power_of_2")
    namespace = {}
    exec(compile(ast.Module([method], []), str(filename), "exec"), namespace)
    def read(name):
        path = root / "mem_cache/allocator" / (name + ".py")
        calls = [n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Subscript) and ast.unparse(n.func.value) == "alloc_extend_kernel"]
        assert len(calls) == 1 and len(calls[0].args) == 8
        tail = ast.Tuple(calls[0].args[-3:], ast.Load())
        return compile(ast.fix_missing_locations(ast.Expression(tail)), str(path), "eval")
    expressions = [read(name) for name in CALLERS]
    def parameters(page, bs):
        from types import SimpleNamespace
        env = dict(namespace, bs=bs, self=SimpleNamespace(page_size=page))
        values = [eval(exp, env) for exp in expressions]
        assert values[0] == values[1] == (1 << (bs - 1).bit_length(), page, 1 << (page - 1).bit_length())
        return values[0]
    return parameters


def interpret(fn, parameters):
    rng = np.random.default_rng(307)
    counts = Counter()
    for page in PAGES:
        for bs in (1, 5, 32):
            for seqdtype in (np.int32, np.int64):
                for repeat in range(4):
                    # Every old partial page ends at its actual logical offset.
                    # Reuse the same physical pools across trials, including holes.
                    pre = rng.integers(0, 4 * page + 1, bs, dtype=seqdtype)
                    ext = rng.integers(0, 3 * page + 1, bs, dtype=seqdtype)
                    if repeat == 0:
                        pre[:] = 0
                        ext[:] = 0
                    elif repeat == 1:
                        pre[::2] = page - 1
                        ext[::2] = 1
                    elif repeat == 2:
                        pre[:] = page
                        ext[0] = page * (4096 // page + 1) + page - 1
                    else:
                        pre[-1] = 2 * page + (page // 2)
                        ext[-1] = 3 * page + 1
                    seq = pre + ext
                    old_pages = rng.permutation(bs) + 11000
                    last = np.where(pre > 0, old_pages * page + (pre - 1).astype(np.int64) % page, -1).astype(np.int64)
                    ceil_pages = lambda row: (row.astype(np.int64) + page - 1) // page
                    added = ceil_pages(seq) - ceil_pages(pre)
                    n = int(added.sum())
                    free = rng.permutation(max(n,1)).astype(np.int64) + 2
                    expected = []
                    taken = 0
                    for i in range(bs):
                        available = free[taken:taken + int(added[i])]
                        for index in range(int(pre[i]),int(seq[i])):
                            logical_page, offset = divmod(index,page)
                            if logical_page < int(ceil_pages(pre)[i]):
                                expected.append(int(last[i]) + index - int(pre[i]) + 1)
                            else:
                                expected.append(int(available[logical_page - int(ceil_pages(pre)[i])]) * page + offset)
                        taken += len(available)
                    assert taken == n
                    output = np.full(len(expected)+4, -132435, dtype=np.int64)
                    actual_args = parameters(page,bs)
                    args = [buf(x,DTYPES[x.itemsize]) for x in (pre,seq,last,free,output[2:-2])]
                    fn[(bs,)](*args,*actual_args)
                    assert np.array_equal(output[2:-2], expected) and np.all(output[:2] == -132435) and np.all(output[-2:] == -132435)
                    if page & (page - 1) == 0:
                        output[:] = -132435
                        fn[(bs,)](*args,*actual_args[:2])
                        assert np.array_equal(output[2:-2],expected) and np.all(output[[0,1,-2,-1]]==-132435)
                        counts["legacy_abi_cases"] += 1
                    counts["cases"] += 1
                    counts["logical_requests"] += bs
                    counts["slot_indices"] += len(expected)
                    counts["zero_extend_requests"] += int((ext == 0).sum())
                    counts["prior_partial_pages"] += int(((pre % page > 0) & (ext > 0)).sum())
                    counts["new_pages"] += n
    return dict(counts=counts, sequence_types=["int32","int64"], page_sizes=PAGES, bs=[1,5,32], physical_cuda=False)


def assemble(kernel, parameters, work):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    def build(fn, page, bs, typecode, legacy=False):
        bs_upper, page_size, block = parameters(page,bs)
        const = dict(bs_upper=bs_upper,page_size=page_size)
        if "page_block" in fn.arg_names:
            const["page_block"] = None if legacy else block
        signature = dict(zip(fn.arg_names[:5], [typecode,typecode,"*i64","*i64","*i64"]))
        signature.update({k:"constexpr" for k in const})
        image = triton.compile(ASTSource(fn,signature,constexprs=const),target=GPUTarget("cuda",90,32),options={"num_warps":4})
        ptx, obj = work/"probe.ptx",work/"probe.cubin"
        ptx.write_text(image.asm["ptx"])
        command = Path(triton.__file__).parent/"backends/nvidia/bin/ptxas"
        log=subprocess.run([command,"-arch=sm_90a","-v",str(ptx),"-o",str(obj)],capture_output=True,text=True,check=True)
        assert "0 bytes spill stores, 0 bytes spill loads" in log.stderr
        regs=re.search(r"Used (\d+) registers",log.stderr)
        return dict(page=page,bs=bs,seq_ptr=typecode,legacy=legacy,registers=int(regs.group(1)),spill_bytes=0,
                    sm90a_cubin_sha256=digest(obj.read_bytes()))
    rows = [build(kernel,p,b,t) for p in PAGES for b in (1,5,32) for t in ("*i32","*i64")]
    rows += [build(kernel,p,5,"*i32",legacy=True) for p in PAGES if p & (p-1) == 0]
    return dict(builds=rows),build


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    parser.add_argument("--parent",type=Path)
    parser.add_argument("--interpreter",action="store_true")
    args=parser.parse_args()
    source=args.stage/"python/sglang/kernels/ops/memory/allocator.py"
    geometry=launch_geometry(args.stage)
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        fn=load_jit_module(source,work).alloc_extend_kernel
        if args.interpreter:
            result=interpret(fn,geometry)
        else:
            result,build=assemble(fn,geometry,work)
            if args.parent:
                parent=load_jit_module(args.parent,work).alloc_extend_kernel
                ordinary=triton.next_power_of_2
                result["native_parent"]=build(parent,160,32,"*i64")
                result["helper_type"]=type(ordinary).__name__
                # CPU compiler control reproducing the judge's plain-function representation.
                if hasattr(ordinary,"fn"):
                    triton.next_power_of_2 = ordinary.fn
                    try:
                        alternate=work/"plain_parent.py"
                        alternate.write_bytes(args.parent.read_bytes())
                        broken=load_jit_module(alternate,work).alloc_extend_kernel
                        try:
                            build(broken,160,32,"*i64")
                        except Exception as exc:
                            result["plain_parent_error"]=str(exc)
                        else:
                            raise AssertionError("plain helper unexpectedly admitted by the compiler")
                        alternate=work/"plain_fixed.py"
                        alternate.write_bytes(source.read_bytes())
                        fixed=load_jit_module(alternate,work).alloc_extend_kernel
                        result["plain_candidate"]=build(fixed,160,32,"*i64")
                    finally:
                        triton.next_power_of_2=ordinary
    result.update(source_sha256=digest(source.read_bytes()),triton=triton.__version__,physical_cuda=False)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
