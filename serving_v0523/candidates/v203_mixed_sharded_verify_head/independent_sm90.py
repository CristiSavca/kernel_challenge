"""Compile the submitted and stock softcap kernels, or exercise the actual Triton row reduction."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
import triton.language as tl


def sha(x):
    return hashlib.sha256(x).hexdigest()


def load(path, name, directory):
    data = Path(path).read_text()
    node = next(n for n in ast.parse(data).body if getattr(n, "name", "") == name)
    text = "import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n"
    parts = data.splitlines(keepends=True)
    text += "".join(parts[node.decorator_list[0].lineno - 1:node.end_lineno])
    module_name = "native_" + Path(path).stem
    temp = directory / (module_name + ".py")
    temp.write_text(text)
    spec = importlib.util.spec_from_file_location(module_name, temp)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, getattr(module, name)


def instructions(ptx):
    ptx = re.split(r'\.section\s+\.debug', ptx, maxsplit=1)[0]
    return [
        line.strip() for line in ptx.splitlines()
        if line.strip() and not line.lstrip().startswith(("//", ".file", ".loc", ".section", ".b8", "{", "}"))
    ]


def aot(kernels, temp):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    records = []
    for alignment, stride_type in ((False, "i32"), (True, "i32"), (True, "i64")):
        original = None
        for mode, kernel in kernels:
            row = mode != "material"
            signature = {name: "i32" for name in kernel.arg_names}
            for name in kernel.arg_names[:2]:
                signature[name] = "*fp32"
            signature["softcapping_value"] = "fp32"
            signature["row_stride" if row else "output_row_stride"] = stride_type
            constants = {"BLOCK_SIZE": 4096 if row else 1024}
            if mode in ("first_nan", "inherited"):
                constants["FIRST_NAN"] = mode == "first_nan"
            signature.update({name: "constexpr" for name in constants})
            attrs = {(i,): [["tt.divisibility", 16]] for i in (0, 1)} if alignment else {}
            compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants, attrs=attrs),
                                      target=GPUTarget("cuda", 90, 32),
                                      options=dict(num_warps=8 if row else 4, num_stages=3))
            ir, ptx = compiled.asm["ttir"], compiled.asm["ptx"]
            scalar = ["arith.divf", "__nv_tanhf", "arith.mulf"]
            pos = [ir.index(op) for op in scalar]
            assert pos == sorted(pos)
            ops = instructions(ptx)
            # A comment/debug-line can change after adding a constexpr branch.
            # Compare every remaining PTX statement and control-flow label.
            if mode == "stock":
                original = ops
            if mode == "inherited":
                assert ops == original, next(((a, b) for a, b in zip(ops, original) if a != b), None)
            pt, out = temp/"runtime.ptx", temp/"runtime.cubin"
            pt.write_text(ptx)
            run = subprocess.run([assembler, "-arch=sm_90a", "-v", pt, "-o", out], capture_output=True, text=True, check=True)
            assert "0 bytes spill stores, 0 bytes spill loads" in run.stderr
            assert not re.search(r'(?:div|mul)\.approx\.ftz\.f32', ptx)
            nan_pred = [l for l in ops if re.search(r'setp\.(?:neu|nan|num)\.f32',l)]
            records.append(dict(mode=mode, pointers_aligned_16=alignment, stride=stride_type,
                                registers=int(re.search(r"Used (\d+) registers", run.stderr)[1]),
                                zero_spills=True, cubin_bytes=out.stat().st_size, cubin_sha256=sha(out.read_bytes()),
                                scalar_ir=scalar, native_nan_predicates=len(nan_pred),
                                native_tanh_ops=sorted(set(re.findall(r"(?:tanh|ex2|lg2|rcp)\.[a-z.]+f32",ptx))),
                                ptx_instruction_digest=sha("\n".join(ops).encode())))
    return dict(compilations=len(records), rows=records)


def interpreter(kernels):
    import numpy as np
    from interpret_decode_kernels import buf as host_buffer

    def buf(a):
        return host_buffer(a, tl.float32)

    current = kernels[0]
    x = np.array([0., 1.], dtype=np.float32)
    out = np.zeros(2, dtype=np.float32)
    try:
        current[1,](buf(x),buf(out),2,2,0,30.,BLOCK_SIZE=4096,FIRST_NAN=True,num_warps=8)
        direct = True
    except Exception as exc:
        direct = False
        issue = repr(exc)[:600]
    # The CPU interpreter supplies no CUDA libdevice symbols. This fallback
    # replaces only the external tanh, in both the dense and reduced kernels.
    if not direct:
        from triton.runtime.interpreter import TensorHandle
        from triton.language.core import tensor
        def host_tanh(v):
            return tensor(TensorHandle(np.tanh(v.handle.data).astype(np.float32), tl.float32), v.type)
        for mod in modules:
            mod.libdevice.tanh = host_tanh

    mat = kernels[1]
    stock = kernels[2]
    cases = tokens = guard_words = nan_rows = signed_zero_cases = finite_ties = 0
    rng = np.random.default_rng(8203)
    for size, width, valid in ((2,1,1),(3,7,19),(4,4095,9617),(2,4096,8105),
                               (3,4097,12289),(4,8213,17257),(8,8192,49161)):
        rows=9
        for cap in (1., 30., 0.125, -4.):
            local = rng.normal(0, 7, (size, rows, width)).astype(np.float32)
            for i in range(size):
                for row in range(rows):
                    absolute = np.arange(width) + i*width
                    if row == 1:
                        local[i,row] = np.where(absolute%2, -0., 0.).astype(np.float32)
                    if row == 2:local[i,row] = -np.inf
                    if row == 3:local[i,row] = np.where(absolute%3, 1e30, np.inf)
                    if row == 4:local[i,row] = np.nan
                    if row == 5:local[i,row,::997] = np.nan
                    if row == 6 and i==0:local[i,row,-1] = np.nan
                    if row == 7:local[i,row] = -0.
                    if row == 8:
                        local[i,row] = -1000
                        local[i,row,::4096] = 45.
                    local[i,row,absolute>=valid] = np.nan
            masked = np.full((rows, valid+5), np.float32(133.5))
            mat[(triton.cdiv(valid,1024),rows)](buf(local),buf(masked),rows,width,valid,valid+5,cap,BLOCK_SIZE=1024)
            expected = np.argmax(masked[:,:valid],axis=1)
            assert np.all(masked[:,valid:]==np.float32(133.5))
            processed = np.empty((size,rows,2),np.float32)
            for rank in range(size):
                valid_width = max(0,min(width,valid-rank*width))
                for flag in (False,True):
                    result = np.full((rows*2+6,), np.float32(123.))
                    # Triton 3.8 CPU uses numpy.nanargmax for the native preexisting
                    # branch, which throws on an all-NaN tile. Its compiled PTX
                    # equivalence covers that case; the new branch retains it.
                    source = local[rank].copy()
                    if not flag:
                        source[4] = 0.
                    current[(rows,)](buf(source),buf(result[3:-3]),width,valid_width,rank*width,cap,
                                     BLOCK_SIZE=4096,FIRST_NAN=flag,num_warps=8)
                    unwrapped = result[3:-3].reshape(rows,2)
                    assert np.all(result[:3]==123.) and np.all(result[-3:]==123.)
                    guard_words += 6
                    if flag:
                        processed[rank] = unwrapped
                    else:
                        old = np.empty((rows,2),dtype=np.float32)
                        stock[(rows,)](buf(source),buf(old),width,valid_width,rank*width,cap,BLOCK_SIZE=4096,num_warps=8)
                        np.testing.assert_array_equal(old.view("u4"),unwrapped.view("u4"))
                    cases += 1
            winners=processed[np.argmax(processed[:,:,0],axis=0),np.arange(rows),1].astype(np.int64)
            np.testing.assert_array_equal(winners, expected)
            nan_rows += int(np.isnan(masked[:,:valid]).any(axis=1).sum())
            signed_zero_cases += int(np.all(masked[[1,7],:valid]==0,axis=1).sum())
            finite_ties += int(((masked[:,:valid] == np.max(masked[:,:valid],axis=1,keepdims=True)).sum(axis=1)>1).sum())
            tokens += rows
    return dict(native_libdevice_interpreted=direct, interpreter_limitation=None if direct else issue,
                kernel_calls=cases, dense_comparison_rows=tokens, guarded_words=guard_words,rows_with_live_nan=nan_rows,
                signed_zero_rows=signed_zero_cases,rows_with_tied_numeric_winners=finite_ties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("aot","interpreter"))
    parser.add_argument("current",type=Path)
    parser.add_argument("original",type=Path)
    parser.add_argument("material",type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td)
        oldmod,old=load(args.original,"_softcap_row_argmax_kernel",temp)
        newmod,new=load(args.current,"_softcap_row_argmax_kernel",temp)
        matmod,mat=load(args.material,"_gathered_softcap_logits_kernel",temp)
        modules=[newmod,oldmod,matmod]
        if args.mode=="aot":
            data=aot([("stock",old),("inherited",new),("first_nan",new),("material",mat)],temp)
        else:
            data=interpreter([new,mat,old])
        print(json.dumps(dict(triton=triton.__version__,physical_gpu=False,
                              source_sha256={k:sha(getattr(args,k).read_bytes()) for k in ("current","original","material")},
                              **data),indent=2))
