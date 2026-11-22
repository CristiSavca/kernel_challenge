"""Join physical eight-H200 client requests to timed flush boundaries and the official V64 control."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "serving_v0523/runs"
METRIC = re.compile(r"full token usage: ([\d.]+), swa token usage: ([\d.]+)")
FLUSH = re.compile(r"^\[.*? TP0 EP0\] Cache flushed successfully!$", re.M)
SCOPED_EVENT = re.compile(r"^.*?TP0 EP0\].*?(?:restor|backup|evict|retract).*?$", re.M | re.I)
SETTINGS = ("enable_hierarchical_cache", "hicache_write_policy", "hicache_io_backend",
            "hicache_mem_layout", "hicache_storage_backend", "page_size", "max_running_requests",
            "speculative_num_draft_tokens", "chunked_prefill_size", "enable_cache_report",
            "enable_metrics", "enable_forward_pass_metrics", "enable_request_time_stats_logging")


def source(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}


def phase_targets(window, score):
    """Illustrative uniform speedup at this load using the archived wall-time split."""
    cold = window["cold_wall_to_final_first_text_s"]
    wall = window["wall_s"]
    later = wall - cold
    remaining_factor = 4.0 / score
    target = wall / remaining_factor
    factors = {}
    for later_factor in (1, 2, 3):
        available = target - later / later_factor
        factors[str(later_factor)] = cold / available if available > 0 else None
    return dict(concurrency=window["concurrency"], rep=window["rep"],
                first_wave_fraction=cold / wall, target_wall_s=target,
                cold_factor_if_later_time_zero=cold / target,
                required_cold_factor_by_later_speedup=factors)


def trace_window(rows, section, capacity):
    rows = sorted(rows, key=lambda r: (r["user"], r["turn"]))
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user"]].append(row)
    later = 0
    prior_output_uncached = 0
    for turns in by_user.values():
        assert [r["turn"] for r in turns] == list(range(13))
        for previous, current in zip(turns, turns[1:]):
            assert current["cached"] == previous["prompt"] + 1
            prior_output_uncached += previous["completion"] - 1
            later += 1
    first = min(row["t0"] for row in rows)
    end = max(row["t0"] + row["dur"] for row in rows)
    cold = [turns[0] for turns in by_user.values()]
    last_cold_text = max(row["t0"] + row["ttft"] for row in cold)
    first_cold_text = min(row["t0"] + row["ttft"] for row in cold)
    cold_rows = sum(row["prompt"] - row["cached"] for row in cold)
    early_later = sum(r["turn"] > 0 and r["t0"] < last_cold_text for r in rows)
    assert not early_later
    # Union of outstanding HTTP intervals. The client is the only clock here.
    idle = 0
    last_end = first
    for row in sorted(rows, key=lambda r: r["t0"]):
        idle += max(0, row["t0"] - last_end)
        last_end = max(last_end, row["t0"] + row["dur"])
    uncached = sum(r["prompt"] - r["cached"] for r in rows)
    generated = sum(r["completion"] for r in rows)
    logical_rows = uncached + generated
    wall = end - first
    cold_wall = last_cold_text - first
    usage = METRIC.findall(section)
    assert usage
    # This conditional exposure assumes all participating pools back every logical row once.
    # Number/length of backed SWA radix nodes, overlaps, evictions and retransfers are unobserved.
    per_pool = {"target_full": 4096, "target_swa": 11776, "private_draft": 2560}
    calls = logical_rows * len(per_pool) * 2
    return dict(
        concurrency=len(cold), rep=rows[0]["rep"], requests=len(rows), later_turns=later,
        missing_from_previous_prompt=0, previous_completion_after_first_total=prior_output_uncached,
        total_cached_rows=sum(r["cached"] for r in rows), uncached_prompt_rows=uncached,
        generated_rows=generated, first_turn_new_rows=cold_rows,
        one_copy_logical_full_rows=logical_rows, nominal_full_capacity_minus_logical=capacity-logical_rows,
        wall_s=round(wall, 6), cold_wall_to_final_first_text_s=round(cold_wall, 6),
        first_text_spread_s=round(last_cold_text-first_cold_text, 6), client_empty_s=round(idle, 6),
        later_arrivals_before_last_cold_text=early_later, cold_logical_prompt_rows_per_s=round(cold_rows/cold_wall, 3),
        entire_post_cold_removal_speedup_bound=round(wall/cold_wall, 6),
        observed_full_swa_peak_locked_or_used=[max(float(v[i]) for v in usage) for i in (0, 1)],
        timed_tp0_transfer_or_retraction_log_lines=len(SCOPED_EVENT.findall(section)),
        all_pools_once_example=dict(bytes_per_rank=logical_rows * sum(per_pool.values()),
            small_copies_per_rank=calls, exposed_us_per_copy_needed={
            str(target): round(1e6*wall*(1-1/target)/calls, 4) for target in (1.0163, 2, 3)}))


def trace(directory, label):
    path = ROOT / "experiments/modal_node" / directory
    files = {k: next(path.glob(f"*__{label}__{k}")) for k in ("load.json", "server.log", "server_info.json")}
    info = json.loads(files["server_info.json"].read_bytes())["internal_states"][0]
    data = json.loads(files["load.json"].read_bytes())
    grouped = defaultdict(list)
    for row in data["requests"]:
        grouped[(row["load"], row["rep"])].append(row)
    sections = FLUSH.split(files["server.log"].read_text())[1:]
    assert len(sections) >= len(grouped)
    windows = [trace_window(rows, section, info["memory_usage"]["token_capacity"])
               for (_, rows), section in zip(sorted(grouped.items()), sections)]
    return dict(sources=[source(path) for path in files.values()], settings={k: info[k] for k in SETTINGS},
                token_capacity={k: info["memory_usage"][k] for k in ("token_capacity", "token_capacity_swa")},
                matched_flush_sections=len(windows), all_flush_sections=len(sections),
                unpaired_log_sections=len(sections)-len(windows), windows=windows)


def main():
    official = RUNS / "abd217baf0ed49fd.json"
    result = json.loads(official.read_bytes())["result"]
    assert result["passed"] and result["frontier"]["score"] == 1.4119
    v41 = trace("20261113T204343Z-bench", "v41")
    v44 = trace("20261113T213641Z-bench", "v44")
    assert [sum(win["later_turns"] for win in tr["windows"]) for tr in (v41,v44)] == [708,36]
    assert all(not w["timed_tp0_transfer_or_retraction_log_lines"] for tr in (v41,v44) for w in tr["windows"])
    output = dict(official=dict(source=source(official), score=result["frontier"]["score"],
        paired_frontier=result["frontier"]["points"], response_summary=[{
            k: row[k] for k in ("concurrency","cache_hit","ttft_first_p90_s","ttft_later_p90_s","reps")
        } for row in result["rungs"]], physical_capacity=[d for d in result["launch_settings"]["derived"]
            if "memory_usage.token_capacity" in d["setting"]]),
        v41=v41, v44=v44, public_control=source(RUNS / "v134_hicache_control.json"))
    score = output["official"]["score"]
    output["four_x_uniform_sensitivity"] = dict(
        stock_target=4.0, official_score=score, extra_uniform_factor=4.0 / score,
        additional_factor_if_only_n_frontier_points_improve={
            str(n): (4.0 / score) ** (4.0 / n) for n in range(1, 5)
        },
        score_if_only_two_frontier_points_quadruple=score * (4.0 ** 2) ** 0.25,
        archive="v41", windows=[phase_targets(window, score) for window in v41["windows"]],
        assumption="Scales the archived first-wave and remaining elapsed time independently. "
                   "The score is a four-load frontier, while V41 reports wall time on separate hardware runs. "
                   "A uniform factor at every load is one scenario. These are time budgets, not GPU predictions.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
