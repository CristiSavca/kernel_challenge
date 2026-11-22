"""Drive V102's SP mode and W4A16 dispatch from full-size meta tensors."""

import json
import sys
import tempfile
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v98_graph_grouped_prefill"))
from entry_check import Dispatch, function, rank_test, selected


class Mode(IntEnum):
    EXTEND = 1
    DECODE = 2
    TARGET_VERIFY = 3


def main(stage):
    root = stage / "python/sglang"
    layer = root / "srt/layers/muse_spark"
    environment = dict(ForwardMode=Mode, layernorm_sp_enabled=lambda: True)
    is_sp = function(root / "srt/layers/layernorm_sp.py", "runs_sp", environment)
    triton = SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b, next_power_of_2=lambda a: 1 << (a-1).bit_length())
    capacity = function(layer / "humming_prefill_waves.py", "wave_capacity", {"triton": triton})
    extend, decode, verify = Mode.EXTEND, Mode.DECODE, Mode.TARGET_VERIFY
    cases = [
        (8191, extend, True, True, True, False, "indexed"),
        (8192, extend, True, True, True, False, "wave"),
        (16384, extend, True, True, True, False, "wave"),
        (16384, extend, True, True, False, False, "wave"),
        (16384, extend, True, False, False, False, "wave"),
        (16384, extend, False, True, True, False, "eager"),
        (32768, extend, True, True, True, False, "eager"),
        (16384, extend, True, True, False, True, "indexed"),
        (8192, decode, True, True, False, False, "indexed"),
        (8192, verify, True, True, False, False, "indexed"),
        (128, verify, True, True, False, False, "indexed"),
    ]
    for rows, mode, enabled, capture, breakable, compiling, want in cases:
        runtime = SimpleNamespace(
            compiler=SimpleNamespace(is_compiling=lambda: compiling),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capture), bfloat16=torch.bfloat16,
            empty_like=torch.empty_like, contiguous_format=torch.contiguous_format,
        )
        post = object()
        expert = SimpleNamespace(
            _post_expert_norm=post, _post_expert_norm_weight=post, use_humming=True,
            humming_graph_prefill=enabled, global_num_experts=256, num_local_experts=32,
            intermediate_size_per_partition=4096, _grouped_humming_and_combine=selected("eager"),
        )
        env = dict(torch=runtime, local_humming_expert_ids=lambda layer, ids: ids,
                   humming_expert_outputs=selected("indexed"), grouped_prefill_waves=selected("wave"),
                   wave_capacity=capacity, get_forward=lambda: SimpleNamespace(sp_active=is_sp(mode)),
                   _HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256, is_in_breakable_cuda_graph=lambda: breakable)
        entry = function(layer / "muse_spark_fp4_moe.py", "_forward_w4a16", env)
        ids = torch.empty(rows, 8, dtype=torch.int32, device="meta")
        hidden = torch.empty(rows, 4096, dtype=torch.bfloat16, device="meta")
        try:
            entry(expert, hidden, None, ids, post)
        except Dispatch as outcome:
            assert outcome.args == (want,), (rows, mode, outcome.args, want)
        else:
            raise AssertionError("production entry did not select a dispatch")
    with tempfile.TemporaryDirectory() as location:
        mp.spawn(rank_test, args=(root, str(Path(location) / "vote")), nprocs=2)
    print(json.dumps({"production_dispatch_cases": len(cases), "gloo_ranks": 2, "readiness_cases": 6,
                      "full_geometry_wave_rows": [8192, 16384], "passed": True}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
