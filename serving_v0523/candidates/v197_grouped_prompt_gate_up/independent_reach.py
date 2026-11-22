"""Replay ordinary model dispatch from an applied V195 parent and price the work against public/own measurements."""
import ast
import itertools
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(sys.argv[1])
V195 = "0481ee37be4bff6d675a4a8fb6144ab73091977e"
MODEL = "python/sglang/srt/models/muse_spark.py"
FP4 = "python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py"
HUM = "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
LOAD = "sglang.srt.layers.muse_spark.muse_spark_humming_moe"
NS = types.SimpleNamespace


def source(path):
    return subprocess.check_output(["git", "-C", STAGE, "show", f"{V195}:{path}"], text=True)


def install_methods(filename, cls, names, namespace):
    module = ast.parse(source(filename))
    body = next(c.body for c in module.body if isinstance(c, ast.ClassDef) and c.name == cls)
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        code = ast.Module([node], [])
        exec(compile(code, filename, "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)


class Selection(Exception):
    pass


def dispatch():
    sp = NS()
    torch = NS(bfloat16="bf16", contiguous_format=True, compiler=NS(), cuda=NS(), empty_like=lambda source, **_: source)
    env = {"torch": torch, "layernorm_sp": sp, "get_parallel": lambda: NS(tp_group="tp8"),
           "_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT": 256}
    install_methods(MODEL, "MuseSpark1xMoE", ["_forward_sp_experts","_forward_sp_replicated_experts"], env)
    install_methods(FP4, "MuseSpark1xFp4Experts", ["forward","_forward_w4a16"], env)
    module = types.ModuleType(LOAD)
    def selected(name):
        def op(*args, **kwargs):
            raise Selection(name)
        return op
    module.humming_decode_eligible = selected("decode_test")
    module.humming_grouped_decode_outputs = selected("decode_impl")
    module.humming_expert_outputs = selected("indexed")
    module.local_humming_expert_ids = lambda self, ids: ids

    class Experts:
        use_w4a16 = use_humming = True
        global_num_experts = 256
        _post_expert_norm = post_expert_norm = object()
        _grouped_humming_and_combine = selected("grouped_prompt")
        _forward_w4a16 = env["_forward_w4a16"]
        __call__ = env["forward"]

    class Model:
        experts = Experts()
        sp_replicate_expert_input = True
        pre_expert_norm = None
        _forward_sp_replicated_experts = env["_forward_sp_replicated_experts"]

    counts = {"grouped_prompt": 0, "indexed": 0}
    with patch.dict(sys.modules, {LOAD: module}):
        for rows, dtype, norm, capture, breakable, compiled in itertools.product(
            (1,256,8191,8192,8193,16384), ("bf16","fp16"), (False,True), (False,True), (False,True), (False,True)
        ):
            gathers = []
            def gather(kind, n):
                assert n == rows
                gathers.append(kind)
                return NS(dtype=dtype, numel=lambda: n * 8)
            sp.sp_exit_gather = gather
            sp.sp_num_tokens = lambda: rows
            torch.compiler.is_compiling = lambda: compiled
            torch.cuda.is_current_stream_capturing = lambda: capture
            env["is_in_breakable_cuda_graph"] = lambda: breakable
            Model.experts._post_expert_norm_weight = 1 if norm else None
            try:
                env["_forward_sp_experts"](Model(), "input", "scores", "ids", "router")
            except Selection as e:
                name = e.args[0]
            else:
                raise AssertionError("Missing dispatch")
            expected = rows >= 8192 and dtype == "bf16" and norm and not compiled and (not capture or breakable)
            assert name == ("grouped_prompt" if expected else "indexed") and gathers == ["input","scores","ids"]
            counts[name] += 1
    return counts


def trace():
    logs = list((ROOT / "experiments/modal_node/20261113T204343Z-bench").glob("*v41__server.log"))
    assert len(logs) == 1
    pattern = re.compile(r"Prefill batch, #new-seq: \d+, #new-token: (\d+)")
    groups = []
    for line in logs[0].read_text().splitlines():
        if "Cache flushed successfully!" in line:
            groups.append([])
        elif groups and "TP0 EP0" in line and (m := pattern.search(line)):
            groups[-1].append(int(m[1]))
    baseline = json.loads((ROOT/"serving_v0523/candidates/v193_scheduler_cache_poll/trace.json").read_bytes())["V41"]["rounds"]
    assert len(groups) == len(baseline) == 7
    reports = []
    for batches, observed in zip(groups, baseline):
        eligible = [n for n in batches if n >= 8192]
        assert len(batches) == observed["prefill_scheduler"]["passes"] and sum(batches) == observed["real_prefill_rows"]
        # Global top eight over eight EP ranks give a rank-average one routed row per input token.
        bytes_saved = sum(eligible) * 4 * (2 * 4096) * 2 * 62
        equivalent = bytes_saved / 4.8e12
        reports.append(dict(users=observed["concurrent_users"], rep=observed["rep"], logical_grouped_passes=len(eligible),
                     all_passes=len(batches), eligible_logical_rows=sum(eligible), all_rows=sum(batches),
                     rank_average_62layer_intermediate_bytes=bytes_saved, transfer_equivalent_s=equivalent,
                     transfer_equivalent_as_fraction_of_wall=equivalent/observed["wall_s"],
                     net_ms_per_layer_call_for_5pct_raw=1000 * observed["wall_s"] * (1 - 1/1.05) / (62*len(eligible)),
                     net_ms_per_layer_call_for_2_833x_raw=1000 * observed["wall_s"] * (1 - 1/2.8330618315744744) / (62*len(eligible))))
    return reports


def main():
    prep = source(HUM)
    for line in ('grouped[-1][2] | {"use_stream_k": False, "use_tma_c": False}',
                 'global_scale = global_scale[:, 0]', 'destination.mul_(scales.unsqueeze(1))', 'silu_and_mul(gate_up, activations)',
                 'route_experts, slots = local_ids.flatten().sort(stable=True)', 'first_local = offsets[0].item()'):
        assert line in prep
    assert "_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT = 256" in source(FP4)
    probe = ROOT/"experiments/modal_serving/20261113T065259Z/result.json"
    hw = json.loads(probe.read_bytes())
    sample = next(r for r in hw["cases"] if r["T"] == 16384)
    assert hw["gpu"] == "NVIDIA H200" and sample["local_routes"] == 16224
    bytes_saved = 4 * sample["local_routes"] * 8192 * 2
    tiles = json.loads((ROOT/"serving_v0523/candidates/v169_grouped_expert_tile_cohorts/generated.json").read_bytes())
    return dict(parent_tree=V195, entry=dispatch(), native_prompt_W13=tiles["8192_176.cu"], archive=trace(),
                H200=dict(source=str(probe.relative_to(ROOT)), gpu=hw["gpu"], original_Marlin_whole_call_us=sample["stock_us"],
                    dense_BF16_reference_us=sample["dense_ceiling"]["us"], local_routes=sample["local_routes"],
                    eliminated_gate_up_bytes=bytes_saved, transfer_equivalent_ms_at_4_8_TB_s=bytes_saved/4.8e9,
                    gated_full_tile_estimate_original_Marlin_62layer_s=62*sample["stock_us"]/1e6))


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
