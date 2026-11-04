"""CPU-only PTX fragment and in-place query staging proof for the V5 experiment.

This checks byte address geometry against the documented m16n8k32 matrix
layout. It does not compile CUDA or establish GPU correctness/performance.
Run from any directory with Python 3; no third-party packages are required.
"""

import ast
from pathlib import Path
import random


def e4m3(bits):
    sign = -1 if bits & 128 else 1
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    assert (exponent, mantissa) != (15, 7)
    return sign * (mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2**(exponent - 7))


def verify():
    candidate = Path(__file__).with_name("native_q1_fp8_qk_v5.py")
    ast.parse(candidate.read_text())
    rng = random.Random(20261104)
    # Use exactly representable nonzero E4M3 values of both signs; encode each
    # element independently so head/dimension/key mixing cannot pass trivially.
    codes = list(range(8, 72)) + list(range(136, 200))
    q = [[rng.choice(codes) for _ in range(64)] for _ in range(16)]
    k = [[[rng.choice(codes) for _ in range(64)] for _ in range(64)] for _ in range(2)]
    q_stage = bytearray(16 * 128)
    k_stage = bytearray(64 * 128)
    owned_raw_chunks = set()
    written = set()
    for row in range(16):
        for part in range(8):
            raw = set(range(row * 128 + part * 16, row * 128 + part * 16 + 16))
            assert not owned_raw_chunks & raw
            owned_raw_chunks |= raw
            dst = row * 128 + part * 16
            assert set(range(dst, dst + 8)) <= raw
            q_stage[dst:dst + 8] = bytes(q[row][part * 8:part * 8 + 8])
    for key in range(64):
        for head in range(2):
            for dim in range(64):
                addr = key * 128 + (((head * 4 + (dim >> 4)) ^ (key & 7)) << 4) + (dim & 15)
                assert addr not in written
                written.add(addr)
                k_stage[addr] = k[head][key][dim]
    assert len(written) == 8192

    outputs = 0
    for head in range(2):
        for half in range(2):
            for nt in range(4):
                result = [[0.0 for _ in range(8)] for _ in range(16)]
                for ks in range(2):
                    a = [[None for _ in range(32)] for _ in range(16)]
                    b = [[None for _ in range(8)] for _ in range(32)]
                    for lane in range(32):
                        g, tig = lane >> 2, lane & 3
                        qa = []
                        for hf in range(2):
                            dim = 32 * ks + 16 * hf + 4 * tig
                            addr = (head * 8 + g) * 128 + (dim >> 3) * 16 + (dim & 7)
                            qa.append(list(q_stage[addr:addr + 4]))
                        ar = qa[0] + [0] * 4 + qa[1] + [0] * 4
                        key = 32 * half + 8 * nt + g
                        br = []
                        for hf in range(2):
                            dim = 32 * ks + 16 * hf + 4 * tig
                            addr = key * 128 + (((head * 4 + (dim >> 4)) ^ (key & 7)) << 4) + (dim & 15)
                            br.extend(k_stage[addr:addr + 4])
                        # Independent PTX element-index formulas, not the
                        # candidate's vector-address construction above.
                        for i, bits in enumerate(ar):
                            row = g if i < 4 or 8 <= i < 12 else g + 8
                            col = 4 * tig + (i & 3) + (16 if i >= 8 else 0)
                            assert a[row][col] is None
                            a[row][col] = e4m3(bits)
                        for i, bits in enumerate(br):
                            row = 4 * tig + (i & 3) + (16 if i >= 4 else 0)
                            assert b[row][g] is None
                            b[row][g] = e4m3(bits)
                    assert all(v is not None for row in a for v in row)
                    assert all(v is not None for row in b for v in row)
                    for row in range(16):
                        for col in range(8):
                            result[row][col] += sum(a[row][d] * b[d][col] for d in range(32))
                for lane in range(32):
                    g, tig = lane >> 2, lane & 3
                    for element in range(2):
                        col = 2 * tig + element
                        key = 32 * half + 8 * nt + col
                        expected = sum(e4m3(q[head * 8 + g][d]) * e4m3(k[head][key][d]) for d in range(64))
                        assert result[g][col] == expected
                        assert result[g + 8][col] == 0
                        outputs += 1
    assert outputs == 16 * 64
    return {"matrix_outputs_checked": outputs, "cache_bytes_checked": len(written),
            "query_raw_bytes_disjoint": len(owned_raw_chunks), "cuda_checked": False}


if __name__ == "__main__":
    print(verify())
