"""Compare native and rank-major BF16 RMS reduction orders using finite inputs.

Run with Python and NumPy. All BF16 squares in the fixture are exact in FP32.
Both paths use the same host reciprocal square root.
"""
import hashlib
import json

import numpy as np


def bf16_words(values):
    bits = np.asarray(values, dtype="float32").view("uint32")
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("uint16")


def as_float(words):
    return (words.astype("uint32") << 16).view("float32")


def native_sum(inputs):
    rows, _ = inputs.shape
    sums = np.zeros((rows, 128), dtype="float32")
    strips = inputs.reshape(rows, -1, 128, 4)
    for strip in range(strips.shape[1]):
        for element in range(4):
            values = strips[:, strip, :, element]
            sums += values * values
    warps = sums.reshape(rows, 4, 32)
    for distance in (16, 8, 4, 2, 1):
        warps = warps[:, :, :distance] + warps[:, :, distance:]
    roots = warps[:, :, 0]
    return (roots[:, 0] + roots[:, 2]) + (roots[:, 1] + roots[:, 3])


def rank_major_sums(inputs, ranks):
    rows, width = inputs.shape
    shards = inputs.reshape(rows, ranks, width // ranks).transpose(1, 0, 2).copy().reshape(-1, 4)
    row_ids = np.arange(rows).reshape(-1, 1)
    shard_vectors = width // (4 * ranks)
    sums = np.zeros((rows, 128), dtype="float32")
    for vector in range(0, width // 4, 128):
        columns = vector + np.arange(128, dtype=np.int64)
        addresses = (columns // shard_vectors * rows + row_ids) * shard_vectors + columns % shard_vectors
        block = shards[addresses]
        for element in range(4):
            sums += block[:, :, element] * block[:, :, element]
    warp_starts = np.arange(0, 128, 32)
    for distance in (16, 8, 4, 2, 1):
        # The out-of-range shuffle reads the originating lane. Only lane zero's root is consumed.
        lanes = np.arange(32)
        sources = np.where(lanes + distance < 32, lanes + distance, lanes)
        sums += np.concatenate([sums[:, start + sources] for start in warp_starts], axis=1)
    roots = sums[:, warp_starts]
    ordinary = (roots[:, 0] + roots[:, 2]) + (roots[:, 1] + roots[:, 3])
    adjacent = (roots[:, 0] + roots[:, 1]) + (roots[:, 2] + roots[:, 3])
    return ordinary, adjacent


def reciprocal_rms(sum_of_squares):
    return np.float32(1) / np.sqrt(sum_of_squares / np.float32(4096) + np.float32(1e-6))


def main():
    rng = np.random.default_rng(284)
    def samples(shape, low_exponent, high_exponent):
        mantissas = rng.uniform(0.5, 1, shape)
        scaled = np.ldexp(mantissas, rng.integers(low_exponent, high_exponent, shape))
        return as_float(bf16_words(scaled * rng.choice([-1, 1], shape)))

    gain = samples(4096, -4, 4)
    cases = roots = words = mismatches = changed_roots = changed_words = early_rounding = 0
    digest = hashlib.sha256()
    for ranks in (4, 8):
        for rows in (1, 8, 17, 32, 128, 256):
            inputs = samples((rows, 4096), -7, 8)
            native = native_sum(inputs)
            fused, adjacent = rank_major_sums(inputs, ranks)
            assert np.array_equal(native, fused)
            normalized = inputs * reciprocal_rms(native)[:, None]
            reference = bf16_words(gain * normalized)
            actual = bf16_words(gain * (inputs * reciprocal_rms(fused)[:, None]))
            wrong_tree = bf16_words(gain * (inputs * reciprocal_rms(adjacent)[:, None]))
            wrong_cast = bf16_words(gain * as_float(bf16_words(normalized)))
            cases += 1
            roots += len(native)
            words += reference.size
            mismatches += int(np.count_nonzero(reference != actual))
            changed_roots += int(np.count_nonzero(adjacent != native))
            changed_words += int(np.count_nonzero(wrong_tree != reference))
            early_rounding += int(np.count_nonzero(wrong_cast != reference))
            digest.update(actual.tobytes())
    assert mismatches == 0 and min(changed_roots, changed_words, early_rounding) > 0
    print(json.dumps({
        "numpy": np.__version__, "seed": 284, "pytorch_ref": "v2.14.1",
        "cuda_source_sha256": "aca7054f068c01c4e7a3ba598764747e8cbdce2d06f4b6206b3b67eb3562eae4",
        "cases": cases, "root_words": roots, "observed_BF16_words": words,
        "matched_candidate_errors": mismatches, "adjacent_tree_different_FP32_roots": changed_roots,
        "adjacent_tree_different_BF16": changed_words, "premature_BF16_errors": early_rounding,
        "sha256": digest.hexdigest(), "physical_H200": False,
    }, indent=2))


if __name__ == "__main__":
    main()
