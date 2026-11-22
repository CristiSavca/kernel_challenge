"""Cross the official score threshold; re-time only measured intervals in our own H200 trace."""
import json
import math
import runpy
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, mean

ROOT = Path(__file__).resolve().parents[3]
SCORED = ROOT / "serving_v0523/runs/abd217baf0ed49fd.json"
TRACE = next((ROOT / "experiments/modal_node/20261113T204343Z-bench").glob("*v41__load.json"))
frontier = runpy.run_path(str(ROOT / "tools/okc_node_loadgen.py"))["frontier"]


def measured_intervals(trace):
    users = defaultdict(list)
    for row in trace["requests"]:
        users[row["load"], row["rep"], row["user"]].append(row)
    rounds = defaultdict(list)
    for (load, rep, _), rows in users.items():
        rows.sort(key=lambda r: r["turn"])
        assert [r["turn"] for r in rows] == list(range(13))
        assert all(0 <= r["ttft"] <= r["dur"] and r["completion"] == r["asked"] for r in rows)
        cold = rows[0]["ttft"]
        later = sum(r["ttft"] for r in rows[1:])
        after = sum(r["dur"] - r["ttft"] for r in rows)
        gap = max(r["t0"] + r["dur"] for r in rows) - cold - later - after
        assert gap >= -1e-7
        out = sum(r["completion"] for r in rows)
        total = sum(r["prompt"] for r in rows) + out
        rounds[load, rep].append((cold, later, after, max(gap, 0), out, total))
    return rounds


def rerate(rounds, factors):
    rates = defaultdict(list)
    busy = defaultdict(float)
    outputs = defaultdict(int)
    walls = defaultdict(list)
    for (load, _), users in sorted(rounds.items()):
        assert len(users) == load
        durations = [sum(t / f for t, f in zip(u[:3], factors)) for u in users]
        makespan = max(d + u[3] for d, u in zip(durations, users))
        rates[load].append(sum(u[5] for u in users) / (8 * makespan))
        busy[load] += sum(durations)
        outputs[load] += sum(u[4] for u in users)
        walls[load].append(makespan)
    return {load: (mean(rates[load]), outputs[load] / busy[load], walls[load]) for load in sorted(rates)}


def crossover(score):
    if score(1_000_000) < 4:
        return None
    low, high = 1, 1_000_000
    for _ in range(60):
        mid = (low + high) / 2
        if score(mid) < 4:
            low = mid
        else:
            high = mid
    return high


def main():
    result = json.loads(SCORED.read_bytes())["result"]
    trace = json.loads(TRACE.read_bytes())
    stock = result["frontier"]["points"]
    rungs = result["rungs"]
    official = result["frontier"]["score"]
    rounds = measured_intervals(trace)
    base = rerate(rounds, (1, 1, 1))
    assert len(rounds) == 7 and len(trace["requests"]) == 767
    for actual in trace["rungs"]:
        rebuilt_y, rebuilt_x, walls = base[actual["concurrency"]]
        assert abs(rebuilt_y - actual["tok_s_per_gpu"]) < .06
        assert abs(rebuilt_x - actual["tok_s_per_user_e2e"]) < .006
        assert all(abs(w - ref) <= .006 for w, ref in zip(walls, actual["wall_s"]))

    def at_stock(curve):
        return [frontier(curve, p["tok_s_per_user"]) / p["stock_tok_s_per_gpu"] for p in stock]

    original_local = at_stock(rungs)
    calibration = [p["ratio"] / raw for p, raw in zip(stock, original_local)]
    assert abs(geometric_mean(p["ratio"] for p in stock) - official) < .0001

    def normalized(curve):
        return geometric_mean(r * c for r, c in zip(at_stock(curve), calibration))

    def model(cold=1, later=1, after=1):
        new = rerate(rounds, (cold, later, after))
        rows = []
        factors = []
        for rung in rungs:
            load = rung["concurrency"]
            sy, sx = [new[load][i] / base[load][i] for i in (0, 1)]
            rows.append({"tok_s_per_gpu": rung["tok_s_per_gpu"] * sy,
                         "tok_s_per_user_e2e": rung["tok_s_per_user_e2e"] * sx})
            factors.append({"users": load, "throughput_multiplier": sy, "user_speed_multiplier": sx})
        peak = max(r["tok_s_per_gpu"] for r in rows)
        return {"scaled_V64_rungs": factors, "modeled_frontier": normalized(rows),
                "peak_raw_estimate": peak, "ceiling_at_any_user_speed": peak / stock_mean,
                "uncalibrated_frontier": geometric_mean(at_stock(rows)),
                "fixed_speed_geomean_proxy": official * geometric_mean(f["throughput_multiplier"] for f in factors)}

    inf = math.inf
    phases = dict(first_ttft_only=(inf, 1, 1), later_ttft_only=(1, inf, 1),
                  all_ttft=(inf, inf, 1), post_first_only=(1, 1, inf),
                  first_ttft_and_post_first=(inf, 1, inf))
    stock_mean = geometric_mean(p["stock_tok_s_per_gpu"] for p in stock)
    peak = max(r["tok_s_per_gpu"] for r in rungs)
    both_axes = lambda factor: normalized([{"tok_s_per_gpu": r["tok_s_per_gpu"] * factor,
                                            "tok_s_per_user_e2e": r["tok_s_per_user_e2e"] * factor}
                                           for r in rungs])
    return dict(official=dict(score=official, factor_to_4=4 / official, fraction_removed_uniform_fixed_speed=1-official / 4,
                             stock_geomean=stock_mean, V64_max_raw=peak, required_max_raw=4 * stock_mean,
                             required_max_raw_factor=4 * stock_mean / peak, ceiling_at_current_max_raw=peak / stock_mean,
                             required_fraction_removed_if_each_anchor_hits_4={p["concurrency"]: 1 - p["ratio"] / 4 for p in stock}),
                model=dict(loadgen_unadjusted_baseline=geometric_mean(original_local),
                           per_anchor_calibration=calibration, coupled_uniform_measured_rung_multiplier_to_4=crossover(both_axes),
                           phase_limits={k: model(*v) for k,v in phases.items()},
                           phase_crossovers_to_4=dict(equal_first_ttft_and_post_first=crossover(lambda z: model(z, 1, z)["modeled_frontier"]),
                               all_ttft_and_post_first_equally=crossover(lambda z: model(z,z,z)["modeled_frontier"]),
                               cold_factor_with_2x_post_first=crossover(lambda z: model(z, 1, 2)["modeled_frontier"]),
                               post_first_factor_with_2x_cold=crossover(lambda z: model(2, 1, z)["modeled_frontier"]),
                               cold_factor_with_post_first_fixed=crossover(lambda z: model(z, 1, 1)["modeled_frontier"]),
                               post_first_factor_with_cold_fixed=crossover(lambda z: model(1, 1, z)["modeled_frontier"]))),
                sources=dict(scored=str(SCORED.relative_to(ROOT)), trace=str(TRACE.relative_to(ROOT)),
                             trace_requests=len(trace["requests"]), rounds=len(rounds), anchor_rule="serving_v0523/RULES.md",
                             frontend="tools/okc_node_loadgen.py::frontier"))


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
