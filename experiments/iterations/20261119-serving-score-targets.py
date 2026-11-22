"""Translate 2x and 3x challenge scores into measurable whole-service phase targets."""
import hashlib
import json
import runpy
from pathlib import Path

root = Path(__file__).resolve().parents[2]
candidates = root / "serving_v0523/candidates"
runs = root / "serving_v0523/runs"
phase_source = next(candidates.glob("v195_*/score_headroom.py"))
curve_source = next(candidates.glob("v265_*/audit.py"))
helpers = runpy.run_path(str(phase_source))
trace_path = helpers["TRACE"]
scored_path = helpers["SCORED"]
check_paths = [runs / name for name in ("ac89fddc4ee147ab.json", "7fcdf9abf39b486a.json")]
read = lambda path: json.loads(path.read_bytes())
trace = read(trace_path)
groups = helpers["measured_intervals"](trace)
rerate = helpers["rerate"]
baseline = rerate(groups, (1, 1, 1))
loads = sorted(baseline)
estimate, calibration = runpy.run_path(str(curve_source))["calibrate"](read(scored_path))
assert loads == [1, 4, 16, 32] and len(trace["requests"]) == 767
for rung in trace["rungs"]:
    gpu_rate, user_rate, walls = baseline[rung["concurrency"]]
    assert abs(gpu_rate - rung["tok_s_per_gpu"]) < 0.06
    assert abs(user_rate - rung["tok_s_per_user_e2e"]) < 0.006
    assert max(abs(a-b) for a,b in zip(walls, rung["wall_s"], strict=True)) < 0.006


def phase_score(cold, streaming):
    # The first and later turns have separate TTFT. Gaps and all later-turn TTFT stay fixed.
    projected = rerate(groups, (cold, 1, streaming))
    factors = [[projected[n][axis] / baseline[n][axis] for n in loads] for axis in (1, 0)]
    return estimate(*factors)


def needed_factor(score, target):
    low, high = 1, 1.05
    # Find the first crossing. At extreme high-load-only multipliers the order
    # of the user-speed anchors can change, so a distant endpoint is unreliable.
    while score(high) < target:
        low, high = high, high * 1.05
        if high > 1e6:
            return None
    for _ in range(60):
        mid = (low + high) / 2
        if score(mid) < target:
            low = mid
        else:
            high = mid
    return high


def rung_score(multiplier, active):
    factors = [multiplier if n in active else 1 for n in loads]
    return estimate(factors, factors)


checks = [read(p)["result"] for p in check_paths]
assert all(r["concurrency"] == [16] and r["passed"] and r["rungs"][0]["reps"] == 1 for r in checks)
assert all(checks[0][k] == checks[1][k] for k in ("gpus", "weights", "image", "shape"))
def normalized(r, field):
    return r["rungs"][0][field] / r["vs_stock"]["rungs"][0]["stock"][field]
x, y = (normalized(checks[1], field) / normalized(checks[0], field)
        for field in ("tok_s_per_user_e2e", "tok_s_per_gpu"))
def transfer(active):
    return estimate([x if n in active else 1 for n in loads], [y if n in active else 1 for n in loads])

thresholds = {}
for target in (2, 3):
    thresholds[target] = dict(
        speed_and_throughput_all_loads=needed_factor(lambda m: rung_score(m, loads), target),
        speed_and_throughput_16_32_only=needed_factor(lambda m: rung_score(m, (16,32)), target),
        cold_and_streaming_equal=needed_factor(lambda m: phase_score(m,m), target),
        cold_only=needed_factor(lambda m: phase_score(m,1), target),
        streaming_only=needed_factor(lambda m: phase_score(1,m), target))
examples = [{"cold_elapsed_factor": cold, "streaming_elapsed_factor": stream, "modeled_score": phase_score(1/cold,1/stream)}
            for cold,stream in ((1,1),(.8,2/3),(1/1.4,1/1.3),(1,0.5),(.5,1),(.5,.5))]
assert round(examples[0]["modeled_score"],4) == calibration["official_score"]
assert round(thresholds[2]["speed_and_throughput_16_32_only"],3) == 1.665
assert round(thresholds[3]["speed_and_throughput_16_32_only"],3) == 2.447
files = [trace_path,scored_path,*check_paths,phase_source,curve_source,root/"tools/okc_node_loadgen.py"]
report = dict(official_V64=calibration["official_score"],break_even=thresholds,examples=examples,
    V204_single_rung=dict(stock_normalized_user_speed_factor=x,stock_normalized_gpu_throughput_factor=y,
        modeled_if_only_16_moves=transfer((16,)),modeled_if_16_and_32_move=transfer((16,32)),
        modeled_if_all_loads_move=transfer(loads)),
    assumptions="Apply relative wall-time and user-time deltas from the 767-request V41 trace to V64's hardware-scored curve. Initial TTFT includes queueing. Preserve later-turn TTFT, client gaps, requests and schedule. Separate private V204/V64 checks used different nodes with one repetition. No result predicts how V204 actually scales.",
    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
print(json.dumps(report, indent=2, allow_nan=False))
