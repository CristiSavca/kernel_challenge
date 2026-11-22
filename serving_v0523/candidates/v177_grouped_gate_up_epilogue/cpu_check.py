"""Replay real sorted-cohort, in-place sparse return and finalizer control on CPU."""
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
path = here.parent / "v169_grouped_expert_tile_cohorts/cpu_check.py"
spec = importlib.util.spec_from_file_location("prior", path)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
prefix = stage / "python/sglang/srt/layers/muse_spark"
jit = ModuleType("sglang.kernels.jit.utils")
jit.load_jit = None
with patch.dict(sys.modules, {"sglang.kernels.jit.utils": jit}):
    paired = prior.module("paired", prefix / "prefill_gate_up.py")

original_fixture, original_bank = prior.fixture, prior.v168.prior.bank
out = Counter()
cases = {}
torch.set_num_threads(1)


def fixture(counts):
    f = original_fixture(counts)
    f.env["run_gate_up"] = paired.run_gate_up
    return f


def bank(f, rank, size, hidden):
    e = original_bank(f, rank, size, hidden)
    e.hidden_size = hidden
    e.w13_weight = torch.empty(1, dtype=torch.int32)
    e.w13_weight_scale = torch.empty(1, dtype=torch.uint8)

    class Launch:
        previous_input = None
        def run(self, inputs, active, ids, offsets, scales, maps, handle, shared, width, sms):
            assert active.data_ptr() != inputs.data_ptr()
            assert offsets[0] == 0 and offsets[-1] == len(ids)
            assert list(ids.unique_consecutive()) == [j for j in range(len(offsets)-1) if offsets[j] < offsets[j+1]]
            out["attempts"] += 1
            if mode == "map_failure":
                return False
            if mode == "second_cohort_failure" and self.previous_input == inputs.untyped_storage().data_ptr() and width == 176:
                # The saved parent fixture accounts successful routes by projection.
                # Its abandoned first call is now superseded by the complete ordinary rerun.
                assert e.gemms[-1][0] == "w13"
                e.gemms.pop()
                out["partial_retry"] += 1
                self.previous_input = None
                return False
            self.previous_input = inputs.untyped_storage().data_ptr() if width < 176 else None
            # Stand in only for the WGMMA contraction and checked scalar CPU activation.
            gate = torch.empty((len(inputs), 2*active.shape[1]), dtype=torch.bfloat16)
            f.env["HummingMethod"].forward_layer(e, inputs=inputs, outputs=gate, expert_layout=offsets,
                compute_config="grouped", tuning_config=width, sublayer_name="w13")
            e.humming_prefill_silu.run(gate, active, ids, scales)
            out["fused_rows"] += len(ids)
            out["fused_calls"] += 1
            return True

    variants = {w: (None, 2, 100, w or 176) for w in (None,136,152,176)}
    if mode == "missing_width":
        variants[136] = None
    addr = ((e.w13_weight.data_ptr() + int(mode == "moved_weight"), e.w13_weight.shape),
            (e.w13_weight_scale.data_ptr(), e.w13_weight_scale.shape))
    e.humming_prefill_gate_up = (Launch(), None, addr, variants, 132)
    return e


prior.fixture = fixture
prior.v168.prior.bank = bank
for mode in ("active", "map_failure", "missing_width", "moved_weight", "second_cohort_failure"):
    cases[mode] = dict(prior.numerical())
    assert cases[mode]["cases"] == 50
assert out["partial_retry"] > 0
print(json.dumps(dict(phase="CPU grouped host adapter with native CUDA boundary emulated",
    cases=cases, events=out, adapter_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (prefix/"prefill_gate_up.py",prefix/"muse_spark_humming_moe.py")}), indent=2))
