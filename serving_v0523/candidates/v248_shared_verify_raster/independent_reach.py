"""Count V247 planner executions and compare separate historical H200 measurements."""
import ast
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import median


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent / "v247_coalesced_serving_decode"
KERNEL = "python/sglang/kernels/ops/attention/paged_verify_attention.py"
PROFILES = ROOT / "experiments/candidates/muse_attention_decode/profiles"
V41 = ROOT / "experiments/modal_node/20261113T204343Z-bench"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def kernel_source(candidate):
    patch = candidate / "patch.diff"
    manifest = json.loads((candidate / "validation.json").read_bytes())
    assert digest(patch) == manifest["patch_sha256"]
    section = patch.read_text().split(f"diff --git a/{KERNEL} b/{KERNEL}\n", 1)[1].split("\ndiff --git ", 1)[0]
    assert "new file mode" in section
    lines = section.split("\n@@ ", 1)[1].splitlines()[1:]
    assert all(s.startswith("+") or not s for s in lines)
    return "\n".join(s[1:] for s in lines if s) + "\n"


def measured_profiles(path):
    archive = json.loads(path.read_bytes())
    cases = {}
    for e in archive["profiles"]:
        c = e["case"]
        if not c["load"].startswith("verify "):
            continue
        assert c["heads"] == "tp8" and c["page"] == 1 and c["q_len"] == 8
        rows = e["report"].splitlines()
        total = float(re.search(r"([\d.]+) µs", next(s for s in rows if s.startswith("total ")))[1])
        planner = [float(re.search(r"([\d.]+) µs", s)[1]) for s in rows if "prepare_varlen_num_blocks_kernel" in s]
        cases[c["batch"], c["layer"]] = (total, planner[0] if planner else None)
    return cases


def main():
    parent = kernel_source(PARENT)
    stage = ast.parse(parent)
    fns = {n.name: n for n in stage.body if isinstance(n, ast.FunctionDef)}
    plan = fns["_plan"]
    calls = lambda node, func: [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == func]
    assert len(calls(plan, "tl.load")) == 3
    assert len(calls(plan, "tl.cumsum")) == len(calls(plan, "tl.sum")) == 1
    assert len(calls(fns["_partition_attention"], "_plan")) == len(calls(fns["_merge_partitions"], "_plan")) == 1
    assert "if first_row >= q_length * GQA:\n        return" in ast.unparse(fns["_merge_partitions"])
    assert "triton.cdiv(num_sms * (1 if window_left >= 0 else 2), kv_heads)" in ast.unparse(fns["verify_partition_workspace"])
    assert "min(4, 1 << programs_per_sm.bit_length() - 1)" in ast.unparse(fns["paged_verify_attention"])
    source_anchors = {n: fns[n].lineno for n in ("_tile_geometry", "_plan", "_partition_attention", "_merge_partitions",
                                                  "verify_partition_workspace", "paged_verify_attention")}
    archive = next(V41.glob("*v41__server.log"))
    info = next(V41.glob("*v41__server_info.json"))
    load = next(V41.glob("*v41__load.json"))
    config = json.loads(info.read_bytes())["internal_states"][0]
    buckets = config["decode_cuda_graph_bs"]
    assert buckets[0] == 1 and buckets[-1] == 32 and config["speculative_num_draft_tokens"] == 8

    by_running = defaultdict(list)
    for line in archive.read_text().splitlines():
        if "Decode batch," not in line:
            continue
        b = int(re.search(r"#running-req: (\d+)", line)[1])
        accepted = float(re.search(r"accept len: ([\d.]+)", line)[1])
        throughput = float(re.search(r"gen throughput \(token/s\): ([\d.]+)", line)[1])
        graph = re.search(r"cuda graph: (True|False)", line)[1] == "True"
        by_running[b].append((graph, b * accepted * 1000 / throughput))
    assert sum(map(len, by_running.values())) == 147
    assert all(graph for rows in by_running.values() for graph, _ in rows)
    # Raw throughput sensitivity holding the recorded TTFT and scheduling gaps
    # fixed for each user. Remove every post-TTFT interval, far more work
    # than the candidate touches. Serving can reschedule in a real run.
    sessions = defaultdict(list)
    measurements = json.loads(load.read_bytes())
    for req in measurements["requests"]:
        sessions[req["load"], req["rep"], req["user"]].append(req)
    by_rep = defaultdict(list)
    for (load_level, rep, _), reqs in sessions.items():
        assert sorted(req["turn"] for req in reqs) == list(range(13))
        end = max(req["t0"]+req["dur"] for req in reqs)
        after = sum(req["dur"]-req["ttft"] for req in reqs)
        assert end > after > 0
        by_rep[load_level, rep].append((end, end-after))
    instant_post = {}
    for (level, rep), pairs in sorted(by_rep.items()):
        assert len(pairs) == level
        observed = max(end for end, _ in pairs)
        reduced = max(end for _, end in pairs)
        archive_wall = next(r["wall_s"][rep] for r in measurements["rungs"] if r["concurrency"] == level)
        assert abs(observed-archive_wall) < .006
        instant_post[f"{level} users, rep {rep}"] = dict(observed_s=round(observed,3), fixed_ttft_and_gap_s=round(reduced,3),
            raw_throughput_factor_if_all_post_ttft_time_vanished=round(observed/reduced,4))

    fa3 = next(PROFILES.glob("e97e9e5f*8727d7d1d2da427c.json"))
    custom = next(PROFILES.glob("87a9bddc*c21d391dbf604bb5.json"))
    stock_cases = measured_profiles(fa3)
    fast_cases = measured_profiles(custom)
    rows = {}
    for running in sorted(by_running):
        b = min(k for k in buckets if k >= running)
        merger_rows = min(4, 1 << (max(1, b * 128 // 132).bit_length() - 1))
        full_partition_ctas, sliding_partition_ctas = 264, 132
        merge_live = running * 128 // merger_rows
        merge_padded = b * 128 // merger_rows
        active_plan_invocations = 16 * full_partition_ctas + 46 * sliding_partition_ctas + 62 * merge_live
        padded_plan_invocations = 16 * full_partition_ctas + 46 * sliding_partition_ctas + 62 * merge_padded
        duration_proxy = median([n for _, n in by_running[running]])
        row = dict(sample_count=len(by_running[running]), captured_capacity=b, running=running, rows_per_merge_cta=merger_rows,
            eligible_batch_dimension=b>1, per_forward_plan_grids=2 if b>1 else 0,
            producer_ctas_across_two_geometries=14 if b>1 else 0,
            stored_int32_plan_bytes_per_bucket=4*(4*b+4+132+66) if b>1 else 0,
            minimum_parent_plan_invocations_if_padding_q_length_zero=active_plan_invocations,
            parent_plan_invocations_if_padding_q_length_eight=padded_plan_invocations,
            parent_logical_i32_loads_if_padding_q_length_zero=active_plan_invocations*3*b,
            parent_logical_i32_loads_if_padding_q_length_eight=padded_plan_invocations*3*b,
            v41_step_duration_proxy_ms=round(duration_proxy, 3))
        if (running, "full") in fast_cases:
            weighted = lambda cases, index: sum(layers * cases[running, mode][index] for mode, layers in (("full", 16), ("sliding", 46)))
            custom_us = weighted(fast_cases, 0)
            stock_us = weighted(stock_cases, 0)
            prepare_us = weighted(stock_cases, 1)
            row["standalone_comparators_us_per_62_layers"] = dict(optimized_full_attention_including_sliding=round(custom_us, 1),
                fa3_aggregate_attention_wrapper=round(stock_us, 1), fa3_preparation_kernel_alone=round(prepare_us, 1))
            row["sensitivity_percent_of_v41_proxy"] = dict(total_optimized_attention=round(custom_us / (duration_proxy*10), 3),
                total_fa3_preparation=round(prepare_us / (duration_proxy*10), 3))
        rows[str(running)] = row
    child = kernel_source(HERE)
    child_methods = {n.name:n for n in ast.parse(child).body if isinstance(n, ast.FunctionDef)}
    refresh = ast.unparse(child_methods["refresh_verify_rasters"])
    assert "triton.cdiv(partitions, 16)" in refresh and "size = 2 * batch + 2 + partitions" in refresh
    assert "1 < batch <= 64" in refresh
    out = dict(full_patch_sha256=digest(HERE/"patch.diff"), kernel_sha256=sha256(child.encode()).hexdigest(),
        parent_sha256=digest(PARENT/"patch.diff"), parent_kernel_sha256=sha256(parent.encode()).hexdigest(),
        parent_kernel=KERNEL, parent_function_lines=source_anchors, enabled_geometry=dict(tp=8, hkv=2, gqa=8,
            width=8, sms=132, full_layers=16, sliding_layers=46, tile_n_full=128, tile_n_sliding=64),
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in (archive, info, load, fa3, custom)},
        v41_source_reports=147, v41_reports_with_more_than_one_running=sum(len(a) for b,a in by_running.items() if b>1),
        graph_buckets=buckets, two_plan_logical_bytes_for_one_metadata_object_per_reported_bucket=sum(
            4*(4*b+4+132+66) for b in buckets if b>1), by_observed_running_count=rows, fixed_chronology_sensitivity=instant_post,
        interpretation=[
            "V41 GPU log entries sample the scheduler on a reporting interval. Their final running-request count and graph flag do not identify every batch in the interval. The steady-batch proxy combines reported accept length, ending concurrency and interval generation throughput; it can include prefill, idle or batch churn.",
            "Static parent grid counts assume eligible ordinary width-eight verification on every layer, H200 132 SMs and two KV heads per GPU. A graph dummy can have zero or full query width. Each plan reads three B-wide int32 input vectors; logical CTA reads are not HBM traffic.",
            "The two standalone H200 attention implementations have different kernels from V247. FA3 has a separately timed metadata kernel and the custom C++ attention does not. Neither profile measures V247's inlined planner or V248's incremental effect.",
            "Fixing each user's observed TTFT and gaps while removing all post-TTFT intervals models a hypothetical raw throughput ceiling for that fixed chronology. It does not predict a rerun that changes scheduling or an official frontier score.",
            "A fourfold serving gain would require 75% of the original elapsed time to disappear at fixed throughput definition. Reusing partition metadata retains all KV traffic, matrix products, value merges and the other model layers. Existing hardware traces do not support a fourfold attribution.",
        ])
    assert math.isclose(rows["32"]["standalone_comparators_us_per_62_layers"]["fa3_preparation_kernel_alone"], 137.2)
    path = HERE / "independent_reach.json"
    path.write_text(json.dumps(out, indent=2)+"\n")
    print(f"{out['v41_source_reports']} archived decode samples, {len(rows)} running counts")


if __name__ == "__main__":
    main()
