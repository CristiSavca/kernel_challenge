"""Compare native two-source BF16 RNE with FP32 RNE followed by BF16 RNE.

Exact integer magnitudes use one minimum BF16 subnormal as the unit. Inputs
are finite. Exact cancellation produces zero; both negative zeros keep sign.
"""
import json
from itertools import product


def units(exponent, mantissa):
    return ((128 + mantissa) << (exponent - 1)) if exponent else mantissa


def rounded(value, precision):
    shift = max(0, value.bit_length() - precision)
    high = value >> shift
    remainder = value - (high << shift)
    if shift and (remainder > 1 << (shift - 1) or
                  (remainder == 1 << (shift - 1) and high & 1)):
        high += 1
    result = high << shift
    return result if result < 128 << 254 else None


def check():
    cases = overflow = cancelled = subnormal = 0
    for big_exp in (0, 1, 2, 8, 16, 64, 127, 200, 253, 254):
        for small_exp in range(big_exp + 1):
            for m, n in product((0, 1, 63, 64, 126, 127), repeat=2):
                a, b = units(big_exp, m), units(small_exp, n)
                for sign_a, sign_b in product((-1, 1), repeat=2):
                    v = abs(sign_a * a + sign_b * b)
                    direct = rounded(v, 8)
                    promoted = rounded(v, 24)
                    staged = None if promoted is None else rounded(promoted, 8)
                    assert direct == staged, (big_exp, small_exp, m, n, sign_a, sign_b)
                    cases += 1
                    overflow += direct is None
                    cancelled += direct == 0
                    subnormal += direct is not None and 0 < direct < 128
    return dict(finite_input_pairs=cases, overflow_outputs=overflow,
                zero_magnitude_outputs=cancelled, subnormal_outputs=subnormal,
                all_exponent_gaps=list(range(255)), precision_bits=[8, 24])


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
