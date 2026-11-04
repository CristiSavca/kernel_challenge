#!/usr/bin/env python3
"""CPU inverse packing proof. Requires NumPy; never imports the CUDA candidate.

Only AST-whitelisted preparation helpers run with a NumPy tensor adapter. The
inverse checks read the actual bytes returned by those candidate helpers and
recover original weight/scale coordinates independently of their reshape code.
"""
import ast
import hashlib
import operator
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/candidates/muse_moe_w4a16/ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a.py"
CANDIDATE = ROOT / "muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py"


class Tensor:
    """Small CPU adapter for the exact integer packing operations under test."""
    def __init__(self, value):
        self.a = np.asarray(value)

    @property
    def shape(self):
        return self.a.shape

    @property
    def device(self):
        return "cpu"

    def numel(self):
        return self.a.size

    def to(self, dtype):
        return Tensor(self.a.astype(dtype))

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.a))

    def reshape(self, *shape):
        return Tensor(self.a.reshape(*shape))

    def view(self, *shape):
        if len(shape) == 1 and shape[0] in (np.uint8, np.int32, np.int64):
            assert self.a.flags.c_contiguous
            return Tensor(self.a.view(shape[0]))
        return self.reshape(*shape)

    def permute(self, *dims):
        return Tensor(self.a.transpose(dims))

    def index_select(self, dim, index):
        return Tensor(np.take(self.a, index.a, axis=dim))

    def __getitem__(self, index):
        if isinstance(index, Tensor):
            index = index.a
        return Tensor(self.a[index])


def raw(x):
    return x.a if isinstance(x, Tensor) else x


for name, fn in {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
                 "and": operator.and_, "or": operator.or_, "rshift": operator.rshift,
                 "lshift": operator.lshift, "ge": operator.ge}.items():
    setattr(Tensor, "__" + name + "__", lambda self, other, fn=fn: Tensor(fn(self.a, raw(other))))
    setattr(Tensor, "__r" + name + "__", lambda self, other, fn=fn: Tensor(fn(raw(other), self.a)))


class CpuTorch:
    uint8, int32, int64 = np.uint8, np.int32, np.int64

    @staticmethod
    def arange(*args, device=None):
        assert device in (None, "cpu")
        return Tensor(np.arange(*args, dtype=np.int64))

    @staticmethod
    def stack(values, dim=0):
        return Tensor(np.stack([raw(v) for v in values], axis=dim))

    @staticmethod
    def cat(values, dim=0):
        return Tensor(np.concatenate([raw(v) for v in values], axis=dim))

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        assert device in (None, "cpu")
        return Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def where(pred, yes, no):
        return Tensor(np.where(raw(pred), raw(yes), raw(no)))


def literal(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def packing_namespace(tree):
    allowed = {"_stored_pos", "_stored_pos_mul", "_pack_codes_nat", "_wgs_kperm_index",
               "_wgs_kperm_codes", "_wgs_kperm_rows", "_wgs_copy"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in allowed]
    assert {n.name for n in nodes} == allowed
    ns = {"torch": CpuTorch, "DQ7": True}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CANDIDATE), "exec"), ns)
    ns["_POS"] = ns["_stored_pos"]()
    ns["_POS_MUL"] = ns["_stored_pos_mul"]()
    return ns


def permutation(p):
    lg = (p >> 6) & 3
    j = (p >> 4) & 3
    t = (p >> 1) & 3
    # i uses original bit 3 as its upper bit.
    i = (p & 1) | (((p >> 3) & 1) << 1)
    return 16 * (4 * t + j) + 4 * lg + i


def check_actual_bytes(ns, code_bytes, scales, rows, positions, label):
    result = ns["_wgs_copy"](Tensor(code_bytes), Tensor(scales), Tensor(rows), positions).a
    units = len(rows) // 256
    pieces = result.reshape(units, 16, 4, 9216)
    words = pieces[..., :8192].copy().view(np.uint32).reshape(units, 16, 4, 4, 4, 32, 4)
    ub, st, wg, lg, wp, lane, j = np.indices(words.shape, sparse=True)
    g, t = lane // 4, lane % 4
    checked_codes = 0
    for register, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for e in range(2):
            got = np.zeros(words.shape, dtype=np.uint32)
            for bit in range(4):
                got |= ((words >> positions[(register, e, bit)]) & 1) << bit
            logical_row = ub * 256 + wg * 64 + wp * 16 + g + h * 8
            original_row = rows[logical_row]
            original_k = st * 256 + 16 * (4 * t + j) + 4 * lg + 2 * q + e
            expected = (code_bytes[original_row, original_k // 2] >> (4 * (original_k % 2))) & 15
            assert np.array_equal(got, np.broadcast_to(expected, got.shape)), (label, register, e)
            checked_codes += got.size
    packed_scales = pieces[..., 8192:].reshape(units, 16, 4, 4, 8, 4, 4, 2)
    ub, st, wg, wp, g, t, j, h = np.indices(packed_scales.shape, sparse=True)
    original_row = rows[ub * 256 + wg * 64 + wp * 16 + g + 8 * h]
    original_block = 16 * st + 4 * t + j
    expected_scales = scales[original_row, original_block]
    assert np.array_equal(packed_scales, expected_scales), label
    print(f"PASS {label}: actual {result.nbytes} packed bytes -> {checked_codes} exact FP4 codes, {packed_scales.size} exact scale bytes")


def main():
    parent_bytes, candidate_bytes = PARENT.read_bytes(), CANDIDATE.read_bytes()
    assert hashlib.sha256(parent_bytes).hexdigest() == PARENT.stem
    p, c = ast.parse(parent_bytes), ast.parse(candidate_bytes)
    preserved = ("_pack_codes", "_pack_scales", "_fat_order", "_unit_order", "_pack_codes_nat",
                 "_scales_prmt_exact", "_main_defines", "kernel_fn", "prepare", "_make_scratch")
    for name in preserved:
        assert ast.dump(function(p, name), include_attributes=False) == ast.dump(function(c, name), include_attributes=False), name
    for name in ("PDL", "PRMT_SCALES", "MAIN_DEFINES", "TINY_DEFINES", "TINY_MAX_M",
                 "TINY2_DEFINES", "TINY2_MAX_M", "MID_DEFINES", "MID_MAX_M", "MAIN_FALLBACK_DEFINES", "MAIN_NS", "DQ7"):
        assert literal(p, name) == literal(c, name), name
    source = literal(c, "CUDA_SRC")
    assert '#define OKC_WGMIN 9' in source and '#define OKC_WGSSC2 1' in source
    assert 'static_assert(OKC_WGSSC2 == 1' in source
    for name in ("MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES", "MAIN_FALLBACK_DEFINES"):
        assert not any("WGMIN" in flag or "WGSSC2" in flag for flag in literal(c, name))
    assert 'wgs_kperm_x(src, kc)' in source  # dormant nonbatched gather is also valid
    assert 'WGS_PIECE = 9216' in source and 'k < BW_E ? 2' in source
    assert '#define OKC_WGSFCI 3' in source and '#define OKC_WGSGS 2' in source
    # Construction of entries 0..11 is the unchanged state literal.
    states = []
    for tree in (p, c):
        states.append(next(n.value for n in function(tree, "_prepare_layout").body
                           if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets)))
    assert ast.dump(states[0], include_attributes=False) == ast.dump(states[1], include_attributes=False)
    print("PASS unchanged dispatch/defines/precision/fallback packing and entries0..11; WGMIN9 excludes non-WGS natural-fat aliases")

    ns = packing_namespace(c)
    pi = permutation(np.arange(256))
    assert np.array_equal(ns["_wgs_kperm_index"]("cpu").a, pi)
    assert sorted(pi.tolist()) == list(range(256))
    for chunk in range(32):
        b = 8 * ((chunk >> 1) & 3) + 2 * (chunk >> 3) + (chunk & 1)
        assert pi[8 * chunk:8 * chunk + 8].tolist() == [2 * b + 64 * t + e for t in range(4) for e in range(2)]
    gate = np.arange(4096).reshape(-1, 8)
    natural_a_rows = np.stack((gate, gate + 4096), axis=1).reshape(-1)
    permuted_a_rows = ns["_wgs_kperm_rows"](Tensor(natural_a_rows)).a
    expected_channels = (np.arange(4096) // 256) * 256 + pi[np.arange(4096) % 256]
    expected_a_rows = np.stack((expected_channels.reshape(-1, 8), (expected_channels + 4096).reshape(-1, 8)), axis=1).reshape(-1)
    assert np.array_equal(permuted_a_rows, expected_a_rows)
    assert sorted(permuted_a_rows.tolist()) == list(range(8192))
    assert np.all(expected_channels // 256 == np.arange(4096) // 256)
    print("PASS exact gather map, gate/up pairing, and unchanged256-column B-readiness groups")

    row = np.arange(8192, dtype=np.uint32)[:, None]
    col = np.arange(2048, dtype=np.uint32)[None, :]
    codes = ((row * 37 + col * 11 + ((row ^ col) >> 3)) & 255).astype(np.uint8)
    scales = ((row * 13 + np.arange(256, dtype=np.uint32)[None, :] * 7) & 255).astype(np.uint8)
    # Complete W13/W2 experts: every hidden channel, all 16 K stages, every
    # output unit/warp/lane, and all code/scale bits are decoded below.
    a_rows = permuted_a_rows
    b_rows = np.arange(4096)
    for name in ("_POS", "_POS_MUL"):
        check_actual_bytes(ns, codes, scales, a_rows, ns[name], "W13 " + name)
        check_actual_bytes(ns, codes, scales, b_rows, ns[name], "W2 " + name)
    # Both old and cached construction give the exact same BF16 pair bits.
    for s0 in range(0x08, 0x7F):
        for s1 in range(0x08, 0x7F):
            expected = (s0 * 16 + 0x5C00) | ((s1 * 16 + 0x5C00) << 16)
            got = ((s0 | (s1 << 16)) * 16 + 0x5C005C00) & 0xFFFFFFFF
            assert got == expected
    print("PASS all14161 allowed PRMT scale pairs; precision is unchanged, reduction-order correctness remains GPU-untested")
    print("candidate_sha256", hashlib.sha256(candidate_bytes).hexdigest())


if __name__ == "__main__":
    main()
