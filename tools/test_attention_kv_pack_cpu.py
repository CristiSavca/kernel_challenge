#!/usr/bin/env python3
"""Narrow source/packing proof; no CUDA execution or floating-point converter model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import build_attention_kv_pack_candidate as build

HEADER_SHA = 'e0bebde335eedb623df836803fbef39be81940ba4e760b2a916686336ff6d17d'
HEADER_PATH = Path('/tmp/okc-cuda-fp8-12.8.90.hpp')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cuda(text):
    tree = ast.parse(text)
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))


def conformance(text):
    parent = build.PARENT.read_text()
    assert sha(parent) == build.PARENT_SHA
    restored = text
    for old, new in ((build.OLD, build.NEW), (build.OLD_NAME, build.NEW_NAME),
                     (build.OLD_NOTE, build.NEW_NOTE)):
        restored = build.once(restored, new, old)
    assert restored == parent, 'unplanned source difference'
    assert build.build() == text, 'build not reproducible'
    src, psrc = cuda(text), cuda(parent)
    assert build.once(src, build.NEW, build.OLD) == psrc
    assert '#define OKC_PACK4 0 ' in src
    a = src.index('__device__ __forceinline__ uint32_t e4m3x4f(')
    b = src.index('// D[64x64]', a)
    helper = src[a:b]
    pa = psrc.index('__device__ __forceinline__ uint32_t e4m3x4f(')
    assert helper == psrc[pa:psrc.index('// D[64x64]', pa)]
    # Active helper: x0 low, x1 high; x2 low, x3 high, then upper pair <<16.
    assert '"=h"(lo) : "f"(x0), "f"(x1)' in helper
    assert '"=h"(hi) : "f"(x2), "f"(x3)' in helper
    assert helper.count('cvt.rn.satfinite.e4m3x2.f32') == 4  # two per preprocessor branch
    assert 'return (uint32_t)lo | ((uint32_t)hi << 16);' in helper
    products = re.findall(r'x\[(\d)\] \* sc', build.NEW)
    assert products == list('01234567')
    assert 'to_e4m3(x[e] * sc)' in build.OLD
    assert 'to_e4m3(x[4 + e] * sc)' in build.OLD
    return {'parent_sha256': sha(parent), 'cuda_parent_sha256': sha(psrc),
            'cuda_candidate_sha256': sha(src), 'helper_sha256': sha(helper),
            'full_parent_restoration': True, 'build_reproducible': True,
            'unchanged_flags_launch_stores_norm_rope_and_helpers': True}


def pack_old(v):
    return tuple(sum(v[4*w+e] << (8*e) for e in range(4)) for w in range(2))


def pack_new(v):
    # Opaque conversion-result bytes: this checks layout, not FP8 rounding itself.
    pair = lambda a, b: a | (b << 8)
    return (pair(v[0], v[1]) | (pair(v[2], v[3]) << 16),
            pair(v[4], v[5]) | (pair(v[6], v[7]) << 16))


def run(candidate=build.CANDIDATE, expected_sha=None):
    candidate = Path(candidate)
    text = candidate.read_text()
    digest = sha(text)
    if expected_sha:
        assert digest == expected_sha, (digest, expected_sha)
    source = conformance(text)
    count = 0
    # Every possible output-byte pair in every pair position, others nonzero.
    for slot in range(4):
        for a in range(256):
            for b in range(256):
                v = [0x80, 0x01, 0x7e, 0xff, 0x7f, 0xfe, 0x00, 0x55]
                v[2*slot:2*slot+2] = [a, b]
                assert pack_old(v) == pack_new(v)
                count += 1
    assert pack_old(list(range(8))) != pack_new(list(reversed(range(8))))
    negatives = []
    for label, old, new in (
        ('reversed_output_arguments', 'e4m3x4f(x[0] * sc, x[1] * sc', 'e4m3x4f(x[1] * sc, x[0] * sc'),
        ('changed_product', 'e4m3x4f(x[0] * sc', 'e4m3x4f(x[0]'),
        ('changed_rounding', 'cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;', 'cvt.rz.satfinite.e4m3x2.f32 %0, %2, %1;'),
    ):
        assert old in text
        try:
            conformance(text.replace(old, new, 1))
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f'negative accepted: {label}')
    header = {'sha256': HEADER_SHA, 'checked_local_exact_header': False,
              'version': 'CUDA 12.8.90; evaluator header version unverified'}
    if HEADER_PATH.exists():
        h = HEADER_PATH.read_bytes()
        assert hashlib.sha256(h).hexdigest() == HEADER_SHA
        hs = h.decode()
        a = hs.index('__nv_cvt_float_to_fp8(const float x,')
        b = hs.index('__nv_cvt_float2_to_fp8x2(', a)
        scalar = hs[a:b]
        assert '(__CUDA_ARCH__ >= 890)' in scalar
        assert 'if (saturate == __NV_SATFINITE)' in scalar
        assert 'cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;' in scalar
        assert ': "f"(x), "f"(0.0f)' in scalar
        assert 'res = (__nv_fp8_storage_t)storage;' in scalar
        header['checked_local_exact_header'] = True
    assert candidate.read_text() == text, 'candidate changed during proof'
    return {'status': 'PASS', 'candidate_sha256': digest, 'source': source,
            'opaque_byte_pair_cases': count, 'negative_controls': negatives,
            'header': header,
            'boundary_review': ['signed zero', 'subnormal/underflow ties', 'normal halfway ties',
                                'largest finite/overflow saturation', 'infinity', 'signed NaN'],
            'boundary_reason': 'Both paths apply the same independent-lane RN/satfinite E4M3 PTX operation to identical FP32 products; CPU byte algebra does not emulate conversion.',
            'limits': ['No CUDA compile, GPU execution, SASS count or performance evidence.',
                       'No independent GPU NaN encoding oracle; equivalence uses PTX lane independence and pinned-header evidence.',
                       'Actual evaluator header/compiler combining remain unverified.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, default=build.CANDIDATE)
    p.add_argument('--expected-sha')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = run(args.candidate, args.expected_sha)
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end='')
