"""Exercise the split W2 finalizer against V125's actual unsplit Triton kernel."""
import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runs'))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def bf16_bits(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    return np.where(np.isnan(x), 0x7fc0, rounded).astype(np.uint16)


def constants(h, k, parts, slots, mode):
    norm, weight, first, scale, round_product = mode
    return dict(H=h, TOPK=k, APPLY_NORM=norm, HAS_WEIGHT=weight, WEIGHT_FIRST=first,
                HAS_IDS=scale, HAS_ROW_MAP=True, HAS_EXPERT_SCALE=scale,
                ROUND_PRODUCT=round_product, BLOCK_H=triton.next_power_of_2(h),
                SPLITS=parts, PART_STRIDE=slots*h if parts > 1 else 0)


MODES = [(False, False, False, False, False), (False, False, False, True, True),
         (True, False, False, True, False), (True, True, False, True, False),
         (True, True, True, True, False), (True, True, True, True, True)]


def compare(candidate, parent, rounding):
    # Triton 3.8's CPU interpreter truncates the mantissa in create_fp_cast.
    # Also exercise both source kernels with the SM90 RN-even conversion.
    # The offline assembly below checks the actual device conversion separately.
    if rounding == 'rn-even':
        from triton.runtime import interpreter
        original_cast = interpreter._convert_float

        def device_cast(x, source, target, mode):
            if source == tl.float32 and target == tl.bfloat16 and mode is None:
                return bf16_bits(x)
            return original_cast(x, source, target, mode)

        interpreter._convert_float = device_cast
    rng = np.random.default_rng(12639)
    cases = values = 0
    for (m, k, h), parts, pattern in itertools.product(
            [(3, 3, 73), (8, 8, 4096), (32, 8, 4096)], (2, 4), ('mixed', 'remote', 'local')):
        slots = m*k
        rows = rng.permutation(slots).astype('i4').reshape(m, k)
        if pattern == 'mixed':
            rows.ravel()[1::3] = -1
        elif pattern == 'remote':
            rows.fill(-1)
        ids = rng.integers(0, 7, (m,k), dtype='i4')
        ids[rows < 0] = -1
        live = rows[rows >= 0]
        partials = rng.standard_normal((parts, slots, h), dtype='f4')
        # Cancellation, halfway BF16 ties and non-first split contributions.
        partials[0, :, ::4] = np.float32(2**18+0.5)
        partials[-1, :, ::4] = np.float32(-2**18)
        partials[:, :, 1::4] = 0
        partials[-1, :, 1::4] = np.float32(1+2**-8)
        partials[:, np.setdiff1d(np.arange(slots), live), :] = np.nan
        summed = partials[0].copy()
        for p in partials[1:]:
            np.add(summed, p, out=summed)
        packed = (bf16_bits(summed) if rounding == 'rn-even'
                  else (summed.view('u4') >> 16).astype('u2'))
        scale = bf16_bits(np.array([0.0, .25, .5, .75, 1.0, 1.25, 1.75], 'f4'))
        scores = rng.random((m,k), dtype='f4')
        gamma = rng.standard_normal(h, dtype='f4')
        for mode in MODES:
            def run(fn, number, src, parent=False):
                result = np.full((m,h), 0x7fc0, dtype='u2')
                const = constants(h, k, number, slots, mode)
                if parent:
                    del const['SPLITS'], const['PART_STRIDE']
                fn[(m,)](buf(src, tl.bfloat16 if number == 1 else tl.float32),
                         buf(gamma, tl.float32), buf(scores, tl.float32), buf(result, tl.bfloat16),
                         buf(rows, tl.int32), buf(ids, tl.int32), buf(scale, tl.bfloat16),
                         eps=1e-6, num_warps=8 if h >= 4096 else 4, num_stages=2, **const)
                assert np.all((result & 0x7f80) != 0x7f80)
                if pattern == 'remote':
                    assert np.all(result == 0)
                return result
            expected = run(parent, 1, packed, parent=True)
            np.testing.assert_array_equal(run(candidate, parts, partials), expected)
            if parts == 2:
                np.testing.assert_array_equal(run(candidate, 1, packed), expected)
            cases += 1
            values += m*h
    return dict(fixtures=cases, compared_bf16_outputs=values, baseline='V125 production Triton finalizer', interpreter_bf16=rounding)


def compile_sm90(kernel, work):
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    results = []
    for (h,k,m), parts, mode in itertools.product([(4096,8,8),(4096,8,32),(73,3,3)],
                                                  (1,2,4), (MODES[0],MODES[4],MODES[5])):
        c = constants(h,k,parts,m*k,mode)
        types = ['*bf16' if parts == 1 else '*fp32','*fp32','*fp32','*bf16','*i32','*i32','*bf16','fp32']
        dyn = [n for n in kernel.arg_names if n not in c]
        sig = dict(zip(dyn, types, strict=True)) | {n:'constexpr' for n in c}
        binary = triton.compile(ASTSource(kernel,sig,constexprs=c), target=GPUTarget('cuda',90,32),
                                options=dict(num_warps=8 if h>=4096 else 4, num_stages=2))
        path = work/'kernel.ptx'
        ptx = binary.asm['ptx']
        rounded_casts = sum('cvt.rn.bf16' in line or 'cvt.rn.satfinite.bf16' in line for line in ptx.splitlines())
        assert rounded_casts > 0
        path.write_text(ptx)
        res = subprocess.run([get_ptxas(90).path,'-arch=sm_90a','-v',str(path),'-o',str(work/'test.cubin')],
                             check=True, text=True, capture_output=True)
        stats = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',res.stderr)
        regs = re.search(r'Used (\d+) registers',res.stderr)
        results.append(dict(h=h,k=k,m=m,parts=parts,norm=mode[0],round_product=mode[-1],
                            ptx_round_to_nearest_sites=rounded_casts,spill_stores=int(stats[1]),spill_loads=int(stats[2]),registers=int(regs[1]),
                            cubin_sha256=hashlib.sha256(binary.asm['cubin']).hexdigest()))
    return dict(variants=results)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path)
    ap.add_argument('--parent',type=Path,help='Run CPU interpretation against the V125 source')
    ap.add_argument('--rounding',choices=['interpreter','rn-even'],default='interpreter')
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fn = load_jit_module(args.source,work)._post_expert_norm_weighted_sum_kernel
        if args.parent:
            (work/'baseline').mkdir()
            parent = load_jit_module(args.parent,work/'baseline')._post_expert_norm_weighted_sum_kernel
            report = compare(fn,parent,args.rounding)
            report['parent_sha256'] = hashlib.sha256(args.parent.read_bytes()).hexdigest()
        else:
            report = compile_sm90(fn,work)
    report.update(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  triton=triton.__version__, physical_gpu=False)
    print(json.dumps(report,indent=2))
