"""Compile the frozen target winner and compare the actual instructions with its parent."""
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
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton.compiler import ASTSource


def function(path, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
    return ''.join(source.splitlines(True)[node.decorator_list[0].lineno - 1:node.end_lineno])


def load(path, math, folder):
    file = folder / (path.stem + '_test.py')
    file.write_text('import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n'
                    + function(math, 'mul_rn_f32') + '\n' + function(path, '_softcap_row_argmax_kernel'))
    spec = importlib.util.spec_from_file_location(file.stem, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = mod
    spec.loader.exec_module(mod)
    return mod._softcap_row_argmax_kernel


def instructions(ptx):
    ptx = ptx.partition('.section')[0]
    return [line.strip() for line in ptx.splitlines() if line.strip()
            and not line.lstrip().startswith(('//', '.file', '.loc', '{', '}'))]


def compile_cases(before, after, numerics, tmp):
    parent, new = load(before, numerics, tmp), load(after, numerics, tmp)
    rows = []
    for stride in ('i32', 'i64'):
        for first in (False, True):
            unscaled = None
            for kernel, scale in ((parent,None), (new,None), (new,0.17677669529663687), (new,1.625)):
                constants = dict(BLOCK_SIZE=4096,FIRST_NAN=first)
                if kernel is new:
                    constants['SCALE'] = scale
                sig = dict(logits_ptr='*fp32', out_ptr='*fp32', row_stride=stride, valid_width='i32',
                           col_offset='i32', softcapping_value='fp32', **{key:'constexpr' for key in constants})
                obj = triton.compile(ASTSource(kernel, sig, constexprs=constants),
                                     target=GPUTarget('cuda',90,32),options=dict(num_warps=8))
                ttir, ptx = obj.asm['ttir'], obj.asm['ptx']
                ops = instructions(ptx)
                if kernel is parent:
                    unscaled = ops
                if scale is None:
                    assert ops == unscaled
                else:
                    order = [ttir.index(op) for op in ('mul.rn.f32','arith.divf','__nv_tanhf','arith.mulf')]
                    assert order == sorted(order)
                    assert any('mul.rn.f32' in op for op in ops)
                pt, cubin = tmp/'case.ptx',tmp/'case.cubin'
                pt.write_text(ptx)
                assembled = subprocess.run([get_ptxas(90).path,'-arch=sm_90a','-v',str(pt),'-o',str(cubin)],capture_output=True,text=True,check=True)
                assert '0 bytes spill stores, 0 bytes spill loads' in assembled.stderr
                rows.append(dict(version='V237' if kernel is new else 'parent', first_nan=first,
                    scale=scale, stride=stride, registers=int(re.search(r'Used (\d+) registers',assembled.stderr)[1]),
                    spills=0, cubin_bytes=cubin.stat().st_size,cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest()))
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','new','numerics'):
        ap.add_argument(arg,type=Path)
    a=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        rows=compile_cases(a.parent,a.new,a.numerics,Path(tmp))
    print(json.dumps(dict(triton=triton.__version__,source_sha256={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in vars(a).items()},
        rows=rows,scaled_ir=['explicit mul.rn.f32','original arith.divf','__nv_tanhf','original arith.mulf'],
        unscaled_ptx_matches_parent=True,physical_H200=False),indent=2))
