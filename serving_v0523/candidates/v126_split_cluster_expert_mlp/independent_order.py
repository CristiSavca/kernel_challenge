"""Independent ownership and FP32 reduction-order check for the shipped V126 header."""

from pathlib import Path
import hashlib
import json

import numpy as np


DIR = Path(__file__).resolve().parent


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def add_rows(values):
    acc = np.zeros(values.shape[1:], dtype=np.float32)
    for row in values:
        acc = np.add(acc, row)
    return acc


def recombine(values, splits, gate):
    pieces = [values[s::splits] for s in range(splits)] if gate else np.split(values, splits)
    return add_rows(np.stack([add_rows(piece) for piece in pieces]))


def counterexamples(splits):
    epsilon = 2.0**-27
    # One nonzero BF16 input times an E2M1 +/-1 coefficient in each live K16.
    gate = np.zeros((256, 1), dtype=np.float32)
    gate[[0, 1, 4], 0] = [1, epsilon, -1]
    down = np.zeros_like(gate)
    down[[0, 256 // splits, 256 // splits + 1], 0] = [1, -1, epsilon]
    result = {}
    for name, values in [("gate", gate), ("down", down)]:
        full = bf16(add_rows(values)).item()
        split = bf16(recombine(values, splits, name == "gate")).item()
        assert (full, split) == ((0, epsilon) if name == "gate" else (epsilon, 0))
        result[name] = dict(full=full, split=split)
    return result


def synthetic_dots():
    rng = np.random.default_rng(97532)
    counts = {p: [0, 0] for p in (2, 4)}
    max_abs = {p: [0.0, 0.0] for p in (2, 4)}
    e2m1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    scales = np.array([.0625, .125, .25, .5, 1., 2.], dtype=np.float32)
    total = 0
    for _ in range(64):
        # Each synthetic K16 has valid BF16 x dequantized BF16 operands. Its
        # FP32 NumPy sum is shared across schedules to isolate intertile order.
        x = bf16(rng.standard_normal((256, 256, 16), dtype=np.float32))
        weight = rng.choice(e2m1, x.shape) * rng.choice(scales, (*x.shape[:2], 1))
        tiles = (x * weight).sum(axis=2, dtype=np.float32)
        baseline = bf16(add_rows(tiles))
        total += baseline.size
        for p in (2, 4):
            for i, gate in enumerate((True, False)):
                trial = bf16(recombine(tiles, p, gate))
                counts[p][i] += int(np.count_nonzero(baseline != trial))
                max_abs[p][i] = max(max_abs[p][i], float(np.max(np.abs(baseline-trial))))
    return dict(dot_products=total, k=4096, bf16_changes_gate_down=counts,
                largest_absolute_change_gate_down=max_abs)


def check_ownership(p, hidden, inner):
    # Every rank maps a disjoint band of activation columns. Every split-rank
    # produces every real staged row of its own output-column/shard exactly once.
    width = inner // (8 * p)
    activation_counts = np.zeros((8, inner), dtype=np.uint8)
    for split in range(p):
        for rank in range(8):
            for start in range(0, width, 64 * (8 // p)):
                for warp in range(8):
                    band = start + 64 * (warp // p)
                    if band >= width:
                        continue
                    for frag in range(warp % p, 4, p):
                        for lane in range(32):
                            for i in range(4):
                                row = (lane % 4) * 2 + i % 2
                                col = (split * 8 + rank) * width + band + frag * 16 + lane // 4 + (i // 2) * 8
                                activation_counts[row, col] += 1
    assert np.all(activation_counts == 1)
    for split in range(p):
        down_k = [(split * 8 + peer) * width + k for peer in range(8) for k in range(width)]
        assert down_k == list(range(split * inner // p, (split + 1) * inner // p))
    owned = np.zeros((8, hidden), dtype=np.uint8)
    for rank in range(8):
        for warp in range(8):
            for band in range(warp * 64, hidden // 8, 512):
                for frag in range(4):
                    for lane in range(32):
                        for i in range(4):
                            row = 2 * (lane % 4) + i % 2
                            col = rank * hidden // 8 + band + frag * 16 + lane // 4 + (i // 2) * 8
                            owned[row, col] += 1
    assert np.all(owned == 1)
    return int(activation_counts.size + p * owned.size)


if __name__ == "__main__":
    delta = (DIR / "delta.diff").read_text().split("diff --git ")[1]
    raw = "".join(line[1:] for line in delta.split("@@", 2)[2].splitlines(keepends=True) if line.startswith("+"))
    sha = hashlib.sha256(raw.encode()).hexdigest()
    manifest = json.loads((DIR / "validation.json").read_text())
    assert sha == manifest["source_hashes"]["python/sglang/kernels/jit/csrc/muse_spark/expert_split_mlp.cuh"]
    for text in ("k = stripe * 16", "k += Splits * 16", "rank * width + band", "row < tile.count",
                 "(split * kCluster + peer) * inner", "for (uint32_t s = 0; s < Splits; ++s)"):
        assert text in raw
    cells = sum(check_ownership(p, h, width * 8 * p) for p in (2, 4)
                for h in (512, 1536, 4096) for width in (64, 128, 192, 256))
    print(json.dumps(dict(source_sha=sha, cuda_execution=False, owned_cells=cells,
                          exact_examples={p: counterexamples(p) for p in (2, 4)},
                          random_intertile_order=synthetic_dots()), indent=2))
