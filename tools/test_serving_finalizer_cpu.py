"""Source and host-array checks; this does not compile or execute Triton/CUDA."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REL = "python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py"
SOURCE = ROOT / "serving/sglang" / REL
BASE = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
BF16 = "bf16"


def bf16(a):
    a = np.asarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    finite = (u & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    rounded = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    bits = np.where(finite, rounded, u) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


class Arr(np.ndarray):
    def to(self, dtype):
        return arr(bf16(self) if dtype == BF16 else np.asarray(self, dtype=dtype))


def arr(a):
    return np.asarray(a).view(Arr)


class Ptr:
    def __init__(self, data, offset=0, dtype=np.float32):
        self.data = data.reshape(-1)
        self.offset = offset
        self.dtype = SimpleNamespace(element_ty=dtype)

    def __add__(self, offset):
        return Ptr(self.data, self.offset + offset, self.dtype.element_ty)


class TL:
    float32 = np.float32
    constexpr = object()
    pid = (0, 0)

    @classmethod
    def program_id(cls, axis):
        return cls.pid[axis]

    @staticmethod
    def arange(a, b):
        return arr(np.arange(a, b))

    @staticmethod
    def zeros(shape, dtype):
        return arr(np.zeros(shape, dtype=dtype))

    @staticmethod
    def static_range(n):
        return range(n)

    @staticmethod
    def load(p, mask=True, other=0):
        offsets, valid = np.broadcast_arrays(p.offset, mask)
        assert np.all((offsets[valid] >= 0) & (offsets[valid] < p.data.size))
        out = np.full(offsets.shape, other, dtype=p.data.dtype)
        out[valid] = p.data[offsets[valid]]
        return arr(out)

    @staticmethod
    def store(p, value, mask=True):
        offsets, valid, values = np.broadcast_arrays(p.offset, mask, value)
        assert np.all((offsets[valid] >= 0) & (offsets[valid] < p.data.size))
        p.data[offsets[valid]] = values[valid]

    @staticmethod
    def sum(a, axis):
        return arr(np.sum(a, axis=axis, dtype=np.float32))

    @staticmethod
    def rsqrt(a):
        return arr(np.float32(1) / np.sqrt(a, dtype=np.float32))


def run():
    source = SOURCE.read_text()
    old = subprocess.check_output(
        ["git", "show", f"{BASE}:{REL}"], cwd=ROOT / "serving/sglang", text=True
    )
    tree, original = ast.parse(source), ast.parse(old)
    kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_small_")]
    assert len(kernels) == 2
    restored = ast.parse(source)
    restored.body = [n for n in restored.body if not (isinstance(n, ast.FunctionDef) and n.name.startswith("_small_"))]
    wrapper = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == "fused_post_expert_norm_weighted_sum")
    selector = next(n for n in wrapper.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "small_cuda" for t in n.targets))
    selection = compile(ast.Expression(selector.value), str(SOURCE), "eval")
    fast = next(n for n in wrapper.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "small_cuda")
    assert wrapper.body.index(selector) > next(i for i, n in enumerate(wrapper.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "tokens == 0")
    wrapper.body.remove(selector)
    wrapper.body.remove(fast)
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    launches = [n for n in ast.walk(fast) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
    assert len(launches) == 2
    assert all(any(k.arg == "enable_fp_fusion" and isinstance(k.value, ast.Constant) and k.value.value is False for k in n.keywords) for n in launches)
    for expression in ("norm_weight.contiguous()", "topk_scores.contiguous()", "slot_expert_ids.contiguous()"):
        assert expression in ast.unparse(fast)
    torch = SimpleNamespace(bfloat16=BF16, float16="f16", float32="f32", float64="f64", int32="i32", int64="i64")
    domain_cases = 0
    for t, k, h, device, dtype in itertools.product((1, 8, 31, 32), (0, 1, 8, 32, 33), (0, 511, 4096, 16384, 16385), ("cpu", "cuda"), (BF16, "f32")):
        x = SimpleNamespace(is_cuda=device == "cuda", dtype=dtype, device=device)
        scores = SimpleNamespace(shape=(t, k), device=device, dtype="f32")
        actual = eval(selection, {"torch": torch, "_MIN_TOKENS_FOR_FUSED": 32}, dict(x=x, tokens=t, topk=k, hidden=h, topk_scores=scores, norm_weight=None, slot_expert_ids=None))
        assert actual == (device == "cuda" and dtype == BF16 and t < 32 and 0 < k <= 32 and 0 < h <= 16384)
        domain_cases += 1
    def tensor(shape, dtype="f32", device="cuda", count=None):
        return SimpleNamespace(shape=shape, dtype=dtype, device=device, numel=lambda: int(np.prod(shape)) if count is None else count)
    base_env = dict(x=SimpleNamespace(is_cuda=True, dtype=BF16, device="cuda"), tokens=3, topk=8, hidden=511,
                    topk_scores=tensor((3, 8)), norm_weight=tensor((511,), BF16), slot_expert_ids=tensor((24,), "i32"))
    directed = [(None, None, True)]
    for name, choices in {
        "topk_scores": [tensor((8, 3)), tensor((3, 8), device="cpu"), tensor((3, 8), dtype="complex")],
        "norm_weight": [tensor((1, 511)), tensor((511,), device="cpu"), tensor((511,), dtype="complex")],
        "slot_expert_ids": [tensor((23,), "i32"), tensor((24,), "f32"), tensor((24,), "i32", "cpu")],
    }.items():
        directed.extend((name, value, False) for value in choices)
    directed.append(("slot_expert_ids", tensor((8, 3), "i64"), True))
    for name, value, expected in directed:
        e = dict(base_env)
        if name is not None:
            e[name] = value
        assert eval(selection, {"torch": torch, "_MIN_TOKENS_FOR_FUSED": 32}, e) == expected
    env = {"tl": TL}
    for kernel in kernels:
        kernel.decorator_list = []
        exec(compile(ast.Module([kernel], type_ignores=[]), str(SOURCE), "exec"), env)
    first, second = [env[n.name] for n in kernels]
    rng = np.random.default_rng(20261104)
    cases = 0
    with np.errstate(all="ignore"):
        for t, k, h in itertools.product((1, 3), (1, 3, 8), (7, 511, 4096)):
            for weighted, weight_first, apply, has_ids in itertools.product((False, True), repeat=4):
                x = bf16(rng.normal(size=(t, k, h)))
                w = bf16(rng.normal(size=h))
                s = rng.normal(size=(t, k)).astype(np.float32)
                ids = rng.choice([-1, 0, 4], size=(t, k)).astype(np.int32)
                if has_ids:
                    x[ids < 0] = np.nan  # invalid storage must never be consumed
                inv, out = np.empty((t, k), np.float32), np.empty((t, h), np.float32)
                args = dict(TOPK=k, H=h, APPLY_NORM=apply, HAS_WEIGHT=weighted, WEIGHT_FIRST=weight_first, HAS_IDS=has_ids)
                for tok, slot in itertools.product(range(t), range(k)):
                    TL.pid = tok, slot
                    first(Ptr(x), Ptr(w), Ptr(ids), Ptr(inv), eps=np.float32(1e-8), BLOCK_H=1 << (h - 1).bit_length(), **args)
                for tok, block in itertools.product(range(t), range((h + 255) // 256)):
                    TL.pid = tok, block
                    second(Ptr(x), Ptr(w), Ptr(s), Ptr(ids), Ptr(inv), Ptr(out, dtype=BF16), BLOCK_H=256, **args)
                f = np.where(ids[..., None] >= 0, x, np.float32(0)) if has_ids else x.copy()
                if apply:
                    if weighted and weight_first:
                        f = f * w
                    variance = np.sum(f * f, axis=-1, dtype=np.float32) / np.float32(h)
                    f = f * (np.float32(1) / np.sqrt(variance + np.float32(1e-8)))[..., None]
                    if weighted and not weight_first:
                        f = f * w
                weighted_values = f * s[..., None]
                expected = np.zeros((t, h), np.float32)
                for slot in range(k):
                    expected += weighted_values[:, slot]
                expected = bf16(expected)
                # NumPy's padded and unpadded reductions can differ, just as
                # actual Torch/Triton trees can: this is a loose host check.
                np.testing.assert_allclose(out, expected, atol=0.03125, rtol=0.016, equal_nan=True)
                cases += 1
        nonfinite = 0
        for weight, score, eps in itertools.product((1.0, np.inf, np.nan), (1.0, np.inf, np.nan), (0.0, 1e-8)):
            x, w, s, ids = np.array([np.nan], np.float32), np.array([weight], np.float32), np.array([score], np.float32), np.array([-1], np.int32)
            inv, out = np.empty(1, np.float32), np.empty(1, np.float32)
            args = dict(TOPK=1, H=1, APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True)
            TL.pid = 0, 0
            first(Ptr(x), Ptr(w), Ptr(ids), Ptr(inv), eps=np.float32(eps), BLOCK_H=1, **args)
            second(Ptr(x), Ptr(w), Ptr(s), Ptr(ids), Ptr(inv), Ptr(out, dtype=BF16), BLOCK_H=256, **args)
            f = np.float32(0) * w[0]
            expected = bf16(f * (np.float32(1) / np.sqrt(f * f + np.float32(eps))) * s[0])
            np.testing.assert_equal(out[0], expected)
            nonfinite += 1
    return {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "original_sha256": hashlib.sha256(old.encode()).hexdigest(), "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "numpy_version": np.__version__, "original_fallback_and_large_path_AST_unchanged": True, "metadata_cases": domain_cases, "directed_auxiliary_metadata_cases": len(directed), "contiguous_auxiliary_copies_present": True, "actual_kernel_host_array_cases": cases, "nonfinite_zero_slot_cases": nonfinite, "limits": ["Actual kernel ASTs execute through a bounded NumPy pointer/mask adapter, not a Triton compiler.", "Host BF16 and rsqrt/reduction behavior are proxies; loose host tolerance is not the GPU/serving acceptance tolerance.", "No GPU numerical, graph, concurrency, resource or performance result is established."]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
