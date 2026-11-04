#!/usr/bin/env python3
"""Bounded CPU feasibility only; no candidate, Torch, CUDA, or provider calls.

Python Gaussian -> FP32 -> FP32 scaled -> BF16 RNE is a distribution proxy,
not the Torch/CUDA RNG, official weights, or a proof of their cardinalities.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (230401, 230402, 230403)
UNITS = 4096
UNIT = 512
K = 8192


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16_rne(x):
    u = struct.unpack('<I', struct.pack('<f', x))[0]
    if (u & 0x7f800000) == 0x7f800000 and (u & 0x007fffff):
        # Preserve sign and upper payload; do not round NaN into infinity.
        return (u >> 16) | 0x40
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16) & 0xffff


def prmt(a, b, selectors):
    source = a | (b << 32)
    result = 0
    for i in range(4):
        s = (selectors >> (4 * i)) & 15
        byte = (source >> (8 * (s & 7))) & 255
        if s & 8:
            byte = 255 if byte & 128 else 0
        result |= byte << (8 * i)
    return result


def pack_unit(bits):
    assert len(bits) == UNIT and all(0 <= x < 65536 for x in bits)
    palette = sorted({(x >> 8) & 127 for x in bits})
    if len(palette) > 8:
        # Fixed primary is deliberately still transferred; raw sidecar is exact.
        return {'primary': bytes(768), 'header': bytes(12) + struct.pack('<I', 1),
                'raw': struct.pack('<512H', *bits), 'cardinality': len(palette)}
    lut = {e: i for i, e in enumerate(palette)}
    low = bytes(x & 255 for x in bits)
    codes = [((x >> 15) << 3) | lut[(x >> 8) & 127] for x in bits]
    packed = bytes(codes[i] | (codes[i + 1] << 4) for i in range(0, UNIT, 2))
    dictionary = bytes([x | 128 for x in palette] + [128] * (8 - len(palette)))
    return {'primary': low + packed, 'header': dictionary + bytes(8),
            'raw': None, 'cardinality': len(palette)}


def unpack_unit(blob):
    assert len(blob['primary']) == 768 and len(blob['header']) == 16
    mode = struct.unpack('<I', blob['header'][12:])[0]
    if mode:
        assert mode == 1 and len(blob['raw']) == 1024
        return list(struct.unpack('<512H', blob['raw']))
    assert blob['raw'] is None
    d0, d1 = struct.unpack('<II', blob['header'][:8])
    out = []
    for start in range(0, UNIT, 8):
        lw0, lw1 = struct.unpack('<II', blob['primary'][start:start + 8])
        code = struct.unpack('<I', blob['primary'][512 + start // 2:516 + start // 2])[0]
        for half, low in enumerate((lw0, lw1)):
            selectors = code >> (16 * half)
            high = prmt(d0, d1, selectors & 0x7777)
            sign = (prmt(0x01010101, 0x01010101, selectors) << 7) & 0xffffffff
            high ^= sign
            for select in (0x5140, 0x7362):
                pair = prmt(low, high, select)
                out.extend((pair & 65535, pair >> 16))
    return out


def exact_checks():
    # Every BF16 bit pattern, grouped so each compressed unit uses eight e71s.
    checked = 0
    for base in range(0, 128, 8):
        values = [(sign << 15) | ((base + index) << 8) | low
                  for low in range(256) for sign in range(2) for index in range(8)]
        for start in range(0, len(values), UNIT):
            unit = values[start:start + UNIT]
            packed = pack_unit(unit)
            assert packed['raw'] is None and packed['cardinality'] == 8
            assert unpack_unit(packed) == unit
            checked += len(unit)
    # All four-element sign/index selectors for a noncontiguous dictionary.
    palette = (0, 1, 7, 31, 63, 64, 126, 127)
    d0, d1 = struct.unpack('<II', bytes(x | 128 for x in palette))
    for selector in range(65536):
        high = prmt(d0, d1, selector & 0x7777)
        high ^= (prmt(0x01010101, 0x01010101, selector) << 7) & 0xffffffff
        expected = sum((palette[(selector >> (4 * i)) & 7] |
                       (((selector >> (4 * i)) & 8) << 4)) << (8 * i) for i in range(4))
        assert high == expected
    fallback_units = [
        [((i % 9) << 8) | ((i & 1) << 15) | (i & 255) for i in range(UNIT)],
        [((i % 128) << 8) | ((i & 1) << 15) | (i & 255) for i in range(UNIT)],
        ([0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xffc1] * 64),
    ]
    for unit in fallback_units:
        assert unpack_unit(pack_unit(unit)) == unit
    assert pack_unit(fallback_units[0])['raw'] is not None
    assert pack_unit(fallback_units[1])['raw'] is not None
    # Exact RNE ties: even-down, odd-up, negative ties, signed zero, infinities.
    for u, expected in [(0x3f808000, 0x3f80), (0x3f818000, 0x3f82),
                        (0xbf808000, 0xbf80), (0xbf818000, 0xbf82),
                        (0, 0), (0x80000000, 0x8000),
                        (0x7f800000, 0x7f80), (0xff800000, 0xff80)]:
        assert bf16_rne(struct.unpack('<f', struct.pack('<I', u))[0]) == expected
    return {'all_bf16_patterns_roundtripped': checked,
            'four_code_selectors_checked': 65536,
            'adversarial_units_checked': len(fallback_units),
            'fallback_required_for_cardinalities': [9, 128],
            'rne_endpoint_checks': 8,
            'scope': 'Bit representation only: no CUDA PRMT execution, MMA/FP32 sum-order or speed proof.'}


def run():
    assert __debug__, 'assertions required'
    proof = exact_checks()
    summaries = []
    for seed in SEEDS:
        rng = random.Random(seed)
        scale = f32(1 / math.sqrt(K))
        cardinality, widths = Counter(), Counter()
        sample_hash = hashlib.sha256()
        fallback = 0
        for _ in range(UNITS):
            values = [bf16_rne(f32(f32(rng.gauss(0.0, 1.0)) * scale)) for _ in range(UNIT)]
            sample_hash.update(struct.pack('<512H', *values))
            es = {(x >> 8) & 127 for x in values}
            cardinality[len(es)] += 1
            widths[max(es) - min(es) + 1] += 1
            blob = pack_unit(values)
            assert unpack_unit(blob) == values
            fallback += blob['raw'] is not None
        fraction = fallback / UNITS
        summaries.append({'seed': seed, 'units': UNITS, 'weights': UNITS * UNIT,
                          'bf16_bits_sha256': sample_hash.hexdigest(),
                          'e71_cardinality_histogram': dict(sorted(cardinality.items())),
                          'e71_contiguous_span_histogram': dict(sorted(widths.items())),
                          'dictionary_overflow_units': fallback,
                          'dictionary_overflow_fraction': fraction,
                          'mean_requested_weight_bytes_per_unit': 784 + 1024 * fraction,
                          'requested_weight_bytes_ratio_to_768': (784 + 1024 * fraction) / 768,
                          'dictionary_only_unit_fraction': 1 - fraction})
    all_fallback = sum(x['dictionary_overflow_units'] for x in summaries)
    return {'status': 'CPU_FEASIBILITY_ONLY', 'date': '2026-11-04',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_sha256': '97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2',
            'generator': {'kind': 'Python random.Random.gauss distribution proxy',
                          'order': 'FP32 Gaussian, FP32 inverse sqrt(8192), FP32 product, BF16 RNE',
                          'K': K, 'seeds': list(SEEDS), 'units_per_seed': UNITS,
                          'weights_per_unit': UNIT,
                          'not_actual_torch_cuda_rng_or_judged_weights': True,
                          'not_proof_of_actual_eligibility': True},
            'layout': {'primary_payload_bytes': 768, 'fixed_header_bytes': 16,
                       'overflow_raw_sidecar_bytes': 1024,
                       'header': 'eight e71|0x80 bytes, uint32 raw-unit index, uint32 mode',
                       'codes': 'lane-major consecutive four-bit sign/index nibbles; differs from parent nibble order',
                       'modeled_raw_index': 0,
                       'raw_storage': 'test model returns raw sidecar separately; not a production allocator'},
            'exact_checks': proof, 'proxy_samples': summaries,
            'combined': {'units': len(SEEDS) * UNITS, 'weights': len(SEEDS) * UNITS * UNIT,
                         'overflow_units': all_fallback,
                         'overflow_fraction': all_fallback / (len(SEEDS) * UNITS)},
            'limits': ['No candidate implementation, hardware allocation, compiler or provider calls.',
                       'The fixed primary is transferred even for raw fallback units; extra request bytes are 1024 per overflow.',
                       'No latency, occupancy, cache reuse, numerical MMA correctness or speed forecast follows.',
                       'Unit dictionary removes corrective arithmetic mathematically but changes FP32 accumulation association versus the parent.',
                       'Generic fallback and actual weight qualification are mandatory; distribution samples cannot justify omitting them.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/iterations/20261104-dense-unit-dictionary-feasibility.json')
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'combined': result['combined'],
                      'overflow_by_seed': [x['dictionary_overflow_units'] for x in result['proxy_samples']],
                      'report': str(args.output)}, sort_keys=True))
