#!/usr/bin/env python3
"""CPU-only exact arithmetic audit of existing NVFP4 decode and proposed bases.

No candidate changes/execution, CUDA compilation, GPU work, or timing claims.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/candidates/muse_moe_w4a16/ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a.py"
OUT = ROOT / "experiments/iterations/20261104-nvfp4-decode-math.json"


def pow2(exponent):
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**(-exponent))


def e4m3(byte):
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        return None
    magnitude = mantissa * pow2(-9) if exponent == 0 else (8+mantissa) * pow2(exponent-10)
    return -magnitude if byte & 128 else magnitude


def e2m1(code):
    magnitude = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
                 Fraction(2), Fraction(3), Fraction(4), Fraction(6)][code & 7]
    return -magnitude if code & 8 else magnitude


def bf16_value(bits):
    exponent, mantissa = (bits >> 7) & 255, bits & 127
    assert exponent < 255
    magnitude = mantissa * pow2(-133) if exponent == 0 else (128+mantissa) * pow2(exponent-134)
    return -magnitude if bits & 32768 else magnitude


def exact_bf16_bits(value, negative_zero=False):
    # Our rational values are binary and have <=8 significant bits. Verify both
    # FP32 and BF16 conversions are exact, not just close after rounding.
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    assert bits & 65535 == 0, (str(value), hex(bits))
    result = bits >> 16
    if value == 0 and negative_zero:
        result |= 32768
    assert bf16_value(result) == value
    return result


def main():
    products = 0
    subnormal_products = 0
    max_significant_bits = 0
    prmt_accepted = []
    prmt_subnormal_failures = []
    bases = [(0, 0), (1, -1), (1, 0), (3, -1), (1, 1), (3, 0), (1, 2), (3, 1)]
    for scale_byte in range(256):
        scale = e4m3(scale_byte)
        if scale is None:
            continue  # NaN scales require existing fallback semantics, no equality claim.
        sb = exact_bf16_bits(scale * pow2(64), bool(scale_byte & 128))
        # PRMT SCD/SCDK's rebias is valid only on the checked positive normal range.
        if 8 <= scale_byte <= 126:
            assert ((scale_byte << 4) + 0x5c00) == sb
            prmt_accepted.append(scale_byte)
        elif scale_byte < 8:
            direct = (scale_byte << 4) + 0x5c00
            assert direct != sb
            prmt_subnormal_failures.append(dict(scale_byte=scale_byte, actual_scale=str(scale),
                naive_prmt_scale=str(bf16_value(direct)*pow2(-64))))
        # Reusable scale and 3*scale bases are exact BF16 at the tensor operand's
        # internal exponent. Do not pre-store these as expanded model weights.
        one_bits = exact_bf16_bits(scale * pow2(-62), bool(scale_byte & 128))
        three_bits = exact_bf16_bits(3 * scale * pow2(-62), bool(scale_byte & 128))
        for code in range(16):
            q = e2m1(code)
            qb = ((code & 7) << 6) | ((code & 8) << 12)
            assert bf16_value(qb) == q * pow2(-126)
            expected = q * scale * pow2(-62)
            negative = bool((code >> 3) ^ (scale_byte >> 7))
            expected_bits = exact_bf16_bits(expected, negative)
            actual = bf16_value(qb) * bf16_value(sb)
            assert exact_bf16_bits(actual, negative) == expected_bits
            factor, exponent = bases[code & 7]
            selected = 0 if factor == 0 else bf16_value(one_bits if factor == 1 else three_bits)
            alternative = selected * pow2(exponent) * (-1 if code & 8 else 1)
            assert exact_bf16_bits(alternative, negative) == expected_bits
            if expected:
                n = abs(expected.numerator)
                while n % 2 == 0:
                    n //= 2
                max_significant_bits = max(max_significant_bits, n.bit_length())
            products += 1
            subnormal_products += int((scale_byte & 127) in range(1, 8))
    finite_values = {e4m3(x) for x in range(256)} - {None}
    normal_triple_unrepresentable = [x for x in range(8, 127) if 3*e4m3(x) not in finite_values]
    result = dict(parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        scope="Exact scalar operand arithmetic only; no generated ISA, runtime performance or decoder-output proof.",
        finite_scale_patterns=254, products_checked=products,
        products_with_nonzero_e4m3_subnormal_scales=subnormal_products,
        maximum_product_significant_bits=max_significant_bits,
        operand_nonzero_range=[str(pow2(-72)), str(2688*pow2(-62))],
        positive_normal_prmt_scale_patterns=len(prmt_accepted),
        invalid_prmt_zero_subnormal_examples=prmt_subnormal_failures,
        scale_three_reuse_exact=True,
        positive_normal_scales_whose_triple_is_not_e4m3=len(normal_triple_unrepresentable),
        triple_e4m3_counterexamples=normal_triple_unrepresentable[:8],
        universal_table_bytes=dict(full_single=256*16*2, sign_factored_single=128*8*2,
            full_same_scale_pair=256*256*4, sign_factored_same_scale_pair=128*64*4),
        parent_dynamic_shared=231168, parent_optin_limit=232448,
        existing_scale_lut_bytes=1024,
        sign_factored_single_table_shared_if_reusing_lut=231168-1024+2048,
        gpu_used=False, candidate_modified=False)
    assert products == 4064 and subnormal_products == 224 and max_significant_bits == 6
    OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
