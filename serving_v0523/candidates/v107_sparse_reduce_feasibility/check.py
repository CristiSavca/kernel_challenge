"""Execute the pinned H200 collective selector and compare exact BF16 boundaries."""

import ast
from collections import namedtuple
from enum import Enum
from functools import cache
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import NamedTuple, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def pinned(file):
    return subprocess.check_output(["git", "-C", str(ROOT / "sglang"), "show",
                                    f"{PIN}:python/sglang/{file}"]).decode()


def definitions(source, names):
    nodes = [n for n in ast.walk(ast.parse(source))
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    return "\n".join(ast.unparse(n) for n in nodes)


def chosen_algorithms():
    kinds = Enum("AllReduceAlgo", ["ONE_SHOT_PUSH", "TWO_SHOT_PUSH", "ONE_SHOT_PULL", "TWO_SHOT_PULL"])
    config = namedtuple("Choice", "algo use_graph use_multicast", defaults=(False, False))
    env = dict(KB=1024, MB=1024**2, cache=cache, Optional=Optional, NamedTuple=NamedTuple)
    code = definitions(pinned("srt/distributed/device_communicators/configs/custom_all_reduce_v2.py"),
                       {"Range", "Heuristic", "AllReduceConfig", "_pack_heuristic", "_sm90_configs"})
    exec("from __future__ import annotations\n" + code, env)
    driver = SimpleNamespace(config=env["_sm90_configs"](132)[8])
    env.update(AllReduceConfig=config, AllReduceAlgo=kinds)
    exec("from __future__ import annotations\n" + definitions(
        pinned("srt/distributed/device_communicators/custom_all_reduce_v2.py"), {"_pick_config"}), env)
    def pick(n):
        result = env["_pick_config"](driver, n, can_use_graph=True)
        return result.algo.name + ("_MULTICAST" if result.use_multicast else "")
    return pick


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + 0x7fff + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xffff0000)).view(np.float32)


def main():
    pick = chosen_algorithms()
    p = bf16([1, 1])
    q = bf16([1, 256])
    w = bf16([1, -1])
    primary = bf16(bf16(p + q) @ w)
    swapped = bf16(bf16(p @ w) + bf16(q @ w))
    assert (primary, swapped) == (-254, -255)

    def norm(x):
        return bf16(x / np.sqrt(np.mean(x*x, dtype=np.float32) + np.float32(1e-6)))
    a, b = bf16([1, 0]), bf16([0, 1])
    per_route = bf16(norm(a) + norm(b))
    after_sum = norm(bf16(a + b))
    assert np.all(per_route != after_sum)
    half = bf16(np.array([2**-10], dtype=np.float32))
    scaled_norm, skipped_scale = norm(bf16(4*half)), norm(half)
    assert np.all(scaled_norm != skipped_scale)

    shape = {}
    for m in [1, 8, 16, 32, 64, 128, 256]:
        expert = m * 4096 * 2
        attn = m * 8192 * 2
        shape[m] = {"expert": pick(expert), "attention": pick(attn),
                    "expert_payload_KiB": expert/1024, "attention_payload_KiB": attn/1024,
                    "local_expert_result_write_plus_read_MiB_62_layers": 2*expert*62/2**20,
                    "one_shot_expert_peer_payload_MiB_62_layers": 7*expert*62/2**20}
    assert shape[8]["expert"] == shape[8]["attention"] == "ONE_SHOT_PUSH"
    assert shape[32]["expert"] == "TWO_SHOT_PULL" and shape[32]["attention"] == "TWO_SHOT_PULL_MULTICAST"
    observed = json.loads((ROOT / "runs/abd217baf0ed49fd.json").read_bytes())["result"]["rungs"][0]
    period = observed["acceptance"]/observed["tok_s_per_user_p50"]*1000
    output = {"modes": shape, "bf16_unpool_before_reduce": [float(primary),float(swapped)],
              "bf16_norm_before_vs_after_route_sum": [per_route.tolist(),after_sum.tolist()],
              "actual_W2_scale_vs_skipped_before_RMS": [scaled_norm.tolist(),skipped_scale.tolist()],
              "uniform_top8_mean_active_EP_ranks_per_row":8*(1-math.comb(224,8)/math.comb(256,8)),
              "extra_unpool_weight_MiB_per_rank_per_layer_for_row_groups":
                  {g:(g-1)*8 for g in [1,2,4,8]},
              "illustrative_1user_verify_ms_from_mean_acceptance_and_median_decode_rate":period,
              "microseconds_required_saved_per_layer_for_rung_speedup":
                  {factor:1000*period*(1-1/factor)/62 for factor in [2,4]}}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
