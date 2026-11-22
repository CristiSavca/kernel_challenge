"""Exercise the staged candidate's actual Triton/Python bodies with the V66 CPU shims."""
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

stage = Path(sys.argv[1]).resolve()
v66 = Path(__file__).resolve().parents[1] / "v66_humming_grouped_decode/cpu_check.py"
# Reuse the pinned language emulator, fake Humming and test vectors.
fixture = v66.read_text()
setup = fixture.split("for experts, slots in itertools.product", 1)[0]
ns = {"__name__": "cpu_fixture", "__file__": str(v66)}
exec(compile(setup, str(v66), "exec"), ns)
scope = ns["scope"]
tensor, pointer = ns["tensor"], ns["Pointer"]


class StageLanguage(ns["Language"]):
    minimum = staticmethod(np.minimum)
    max = staticmethod(lambda x, axis: tensor(np.max(x, axis=axis)))
    sum = staticmethod(lambda x, axis: tensor(np.sum(x, axis=axis, dtype=x.dtype)))
    gather = staticmethod(lambda x, idx, axis: tensor(np.take(x, idx, axis=axis)))
    program_id = classmethod(lambda cls, axis: cls.coords[axis])


stage_scope = {"tl": StageLanguage}
ns["extract"](ns["adapter"], ("_stage_grouped_decode",), stage_scope)


class StageLaunch:
    def __getitem__(self, grid):
        def run(*args, **opts):
            args = [pointer(arg) if isinstance(arg, ns["Tensor"]) else arg for arg in args]
            order = list(itertools.product(*(range(n) for n in grid)))
            rng.shuffle(order)
            opts.pop("num_warps")
            for StageLanguage.coords in order:
                stage_scope["_stage_grouped_decode"](*args, **opts)
        return run


scope["envs"] = SimpleNamespace(SGLANG_ENABLE_MUSE_SPARK_HUMMING_FUSED_STAGE=SimpleNamespace(get=lambda: True))
scope["triton"].cdiv = lambda a, b: (a + b - 1) // b
scope["_stage_grouped_decode"] = StageLaunch()
rng = np.random.default_rng(21)
# This established test runs the real Python dispatch, simulated BF16 GEMMs,
# real Triton finalization and independently computed per-route BF16 reference.
start = fixture.index("for tokens, topk, experts, hidden, inner in (")
end = fixture.index("model_path = ", start)
exec(compile(fixture[start:end], str(v66), "exec"), ns)
assert ns["kernel_cases"] == 48

cases = 0
for slots in (1, 31, 32, 127, 128, 129, 256, 508, 513, 1024, 2047, 2048):
    for experts in (1, 7, 32, 64):
        for pattern in ("invalid", "concentrated", "mixed", "all"):
            mapped = pattern in ("invalid", "mixed")
            count = experts * 3 if mapped else experts
            ids = rng.integers(-1, count + 1, slots, dtype=np.int64)
            if pattern == "concentrated":
                ids[:] = experts - 1
            if pattern == "invalid":
                ids[:] = np.where(np.arange(slots) % 2, -1, 2**40)
            if pattern == "all":
                ids[:] = rng.integers(0, experts, slots)
            mapping = np.full(count, -1, dtype=np.int32)
            mapping[rng.choice(count, experts, replace=False)] = np.arange(experts)
            safe = np.clip(ids, 0, count - 1)
            local = np.where((ids >= 0) & (ids < count), mapping[safe] if mapped else ids, -1)
            topk = 4 if slots % 4 == 0 else 1
            width = 13
            source = tensor(rng.integers(0, 65536, size=(slots // topk, width), dtype=np.uint16))
            span = 1 << (slots - 1).bit_length()
            tile = min(span, 128)
            local_out, offsets, keys, rows = [
                tensor(np.full(n, -777, dtype=np.int32)) for n in (slots, experts + 1, slots, slots)
            ]
            out = tensor(np.full((slots, width), 65535, dtype=np.uint16))
            StageLaunch()[(experts, (slots + tile - 1) // tile)](
                tensor(ids), tensor(mapping), source, local_out, offsets, keys, rows, out, slots,
                GLOBAL_EXPERTS=count, EXPERTS=experts, HAS_MAP=mapped, TOPK=topk, HIDDEN=width,
                SLOT_SPAN=span, TILE=tile, ROWS=4, WIDTH=16, num_warps=8,
            )
            live = np.flatnonzero(local >= 0)
            ordered = live[np.argsort(local[live], kind="stable")]
            first = slots - len(ordered)
            expected_rows = np.full(slots, -1)
            expected_rows[ordered] = np.arange(first, slots)
            np.testing.assert_array_equal(local_out, local)
            np.testing.assert_array_equal(rows, expected_rows)
            np.testing.assert_array_equal(keys[first:], (local[ordered] + 1) * span + ordered)
            np.testing.assert_array_equal(out[first:], source[ordered // topk])
            np.testing.assert_array_equal(np.diff(offsets), np.bincount(local[live], minlength=experts))
            assert offsets[0] == first and offsets[-1] == slots
            assert (keys[:first] == -777).all() and (out[:first] == 65535).all()
            cases += 1
print(json.dumps({"staging_cases": cases, "finalizer_cases": ns["kernel_cases"],
                  "bf16_route_values": ns["values_compared"], "numpy": np.__version__}, indent=2))
