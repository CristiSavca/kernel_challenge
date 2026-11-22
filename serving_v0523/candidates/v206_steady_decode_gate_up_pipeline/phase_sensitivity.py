"""Sensitivity of the 4x crossover to actual low-load request chronologies.

Reuses V195's measured-interval and score functions. The single archived runs at
16 and 32 users are held fixed, so this range is not a statistical interval.
"""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from statistics import geometric_mean


HERE = Path(__file__).resolve().parent
model = runpy.run_path(str(HERE.parent / "v195_fused_expert_affine_verify/score_headroom.py"))
root = model["ROOT"]
v41_path = model["TRACE"]
v44_path, = (root / "experiments/modal_node/20261113T213641Z-bench").glob("*v44__load.json")
judge = json.loads(model["SCORED"].read_bytes())["result"]
stock = judge["frontier"]["points"]
rungs = judge["rungs"]


def measurements(path):
    trace = json.loads(path.read_bytes())
    rounds = model["measured_intervals"](trace)
    for rung in trace["rungs"]:
        y, x, walls = model["rerate"](rounds, (1, 1, 1))[rung["concurrency"]]
        assert abs(y - rung["tok_s_per_gpu"]) < .06
        assert abs(x - rung["tok_s_per_user_e2e"]) < .006
        assert len(walls) == len(rung["wall_s"])
        assert all(abs(w - ref) < .006 for w, ref in zip(walls, rung["wall_s"]))
    return rounds


def ratios(curve):
    return [
        model["frontier"](curve, anchor["tok_s_per_user"]) / anchor["stock_tok_s_per_gpu"]
        for anchor in stock
    ]


calibration = [point["ratio"] / value for point, value in zip(stock, ratios(rungs))]


def scenario(rounds):
    base = model["rerate"](rounds, (1, 1, 1))

    def score(cold, after):
        new = model["rerate"](rounds, (cold, 1, after))
        curve = []
        for row in rungs:
            load = row["concurrency"]
            curve.append({
                "tok_s_per_gpu": row["tok_s_per_gpu"] * new[load][0] / base[load][0],
                "tok_s_per_user_e2e": row["tok_s_per_user_e2e"] * new[load][1] / base[load][1],
            })
        return geometric_mean(value * adjustment for value, adjustment in zip(ratios(curve), calibration))

    assert abs(score(1, 1) - judge["frontier"]["score"]) < .0001
    return {
        "equal_phase_factor_for_4x": model["crossover"](lambda f: score(f, f)),
        "stock_score_with_instant_post_first_and_fixed_cold": score(1, float("inf")),
    }


if __name__ == "__main__":
    v41 = measurements(v41_path)
    v44 = measurements(v44_path)
    cases = {}
    for label, one_user in (("V41", v41), ("V44", v44)):
        for rep1 in range(3):
            for rep4 in range(2):
                selected = {(load, 0): users for (load, _), users in v41.items() if load >= 16}
                selected[1, 0] = one_user[1, rep1]
                selected[4, 0] = v41[4, rep4]
                cases[f"{label}-1u-rep{rep1}_V41-4u-rep{rep4}"] = scenario(selected)
    reference = scenario(v41)
    frozen = json.loads((HERE.parent / "v195_fused_expert_affine_verify/score_headroom.json").read_bytes())
    assert abs(reference["equal_phase_factor_for_4x"] -
               frozen["model"]["phase_crossovers_to_4"]["equal_first_ttft_and_post_first"]) < 1e-8
    print(json.dumps({
        "official_V64_score": judge["frontier"]["score"],
        "equal_phase_factor_all_archived_V41_reps": reference["equal_phase_factor_for_4x"],
        "range_among_12_low_load_combinations": {
            key: [min(result[key] for result in cases.values()), max(result[key] for result in cases.values())]
            for key in reference
        },
        "fixed_16_and_32_user_reps": 1,
        "V41_trace_sha256": sha256(v41_path.read_bytes()).hexdigest(),
        "V44_partial_trace_sha256": sha256(v44_path.read_bytes()).hexdigest(),
        "cases": cases,
        "physical_candidate_measurement": False,
    }, indent=2))

