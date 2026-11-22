"""Read the final checks and compare their measured loads with the scored incumbent."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess

root = Path(__file__).resolve().parents[2]
runs = root / "serving_v0523/runs"
candidates = root / "serving_v0523/candidates"
loads = (1, 4, 16, 32)
incumbent_id = "ad275bf67d9a4a0a"
previous_id = "abd217baf0ed49fd"
private_16_id = "7fcdf9abf39b486a"
jobs = (("v129", 1, "022544241ab14973"), ("v141", 32, "226d046c12bf4dbc"),
        ("v154c", 16, "d4545dbfa1ad4e24"))
metrics = ("tok_s_per_gpu", "tok_s_per_user_e2e", "ttft_first_p90_s", "ttft_later_p90_s")


def api(command, *args):
    return json.loads(subprocess.check_output(["okc", "serve", command, *args, "--json"], timeout=40))


def saved(job):
    record = json.loads((runs / (job + ".json")).read_bytes())
    assert record["id"] == job and record["team"] == "CristiSavca" and record["board"] == "serving"
    return record


def measured(rung):
    return {key: rung[key] for key in metrics}


def compare(rung, control):
    return {key: round(rung[key] / control[key], 6) for key in metrics}


def first_crossing(estimate, target):
    low, high = 1.0, 1.05
    while estimate(high) < target:
        low, high = high, high * 1.05
        if high > 10:
            return None
    for _ in range(55):
        mid = (low + high) / 2
        if estimate(mid) >= target:
            high = mid
        else:
            low = mid
    return high


def normalized_axes(rung, stock, reference, point):
    """Map both observed rates to the scored parent's stock rates at this load."""
    return (
        rung["tok_s_per_user_e2e"] / stock["tok_s_per_user_e2e"]
        * point["tok_s_per_user"] / reference["tok_s_per_user_e2e"],
        rung["tok_s_per_gpu"] / stock["tok_s_per_gpu"]
        * point["stock_tok_s_per_gpu"] / reference["tok_s_per_gpu"],
    )


def lineage_scenarios(name, previous, estimate, incumbent_score, check=None):
    # All three branches descend from V64. V141 also inherits V139's small-row path.
    reference = {r["concurrency"]: r for r in previous["result"]["rungs"]}
    points = {r["concurrency"]: r for r in previous["result"]["frontier"]["points"]}
    user = [1.0] * len(loads)
    throughput = user.copy()
    proxy = None
    if name == "v141":
        parent = saved("b013f83007d544d3")
        observed, = parent["result"]["rungs"]
        stock, = parent["result"]["vs_stock"]["rungs"]
        assert observed["concurrency"] == stock["concurrency"] == 4
        user[1], throughput[1] = normalized_axes(observed, stock["stock"], reference[4], points[4])
        proxy = parent["id"]
    scenarios = {
        "v129": ((1,), (1, 4)),
        "v141": ((32,), (16, 32)),
        "v154c": ((16,), (16, 32), loads),
    }
    result = {}
    for transfer in scenarios[name]:
        def apply_axes(user_factor, throughput_factor):
            return estimate(
                [user_factor if n in transfer else v for n, v in zip(loads, user)],
                [throughput_factor if n in transfer else v for n, v in zip(loads, throughput)],
            )
        entry = dict(coupled_factor_over_V64_to_match_V204=first_crossing(
            lambda factor: apply_axes(factor, factor), incumbent_score))
        if check is not None:
            row, = check["rungs"]
            stock, = check["vs_stock"]["rungs"]
            n = row["concurrency"]
            assert n in transfer and stock["concurrency"] == n
            x, y = normalized_axes(row, stock["stock"], reference[n], points[n])
            entry.update(normalized_user_factor=x, normalized_throughput_factor=y,
                         hypothetical_score=apply_axes(x, y))
        result[",".join(map(str, transfer))] = entry
    return dict(scored_parent=previous["id"], four_user_proxy=proxy, by_assumed_affected_loads=result)


def build(listing, quota):
    current = saved(incumbent_id)
    previous = saved(previous_id)
    passing16 = saved(private_16_id)
    assert current["result"]["passed"] and passing16["content_hash"] == current["content_hash"]
    best = max(row["score"] for row in listing if row["mode"] == "run" and row.get("passed") and row.get("score"))
    assert current["result"]["frontier"]["score"] == best, "Rebase the comparison on the newly scored incumbent"
    rungs = {row["concurrency"]: row for row in current["result"]["rungs"]}
    calibration = runpy.run_path(str(next(candidates.glob("v265_*/audit.py"))))["calibrate"]
    previous_model, _ = calibration(previous)
    judge = {row["job_id"]: row for row in listing}
    records = {}
    for name, load, job in jobs:
        directory = next(candidates.glob(name + "_*"))
        patch = directory / "patch.diff"
        validation = json.loads((directory / "validation.json").read_bytes())
        receipt = (runs / f"{name}_nov20_check{load}.submit.txt").read_text().splitlines()
        meta = json.loads(receipt[0])
        assert meta["sha256"] == validation["patch_sha256"] == sha256(patch.read_bytes()).hexdigest()
        assert meta["flags"] == validation["launch_flags"] and meta["users"] == load
        assert meta["candidate"].lower() == name and receipt[-1] == "CLI exit code: 0"
        assert re.search(r"\bjob " + job + "$", receipt[1])
        job_row = judge[job]
        assert job_row["mode"] == "check" and job_row.get("users") in (None, load)
        control = passing16["result"]["rungs"][0] if load == 16 else rungs[load]
        row = dict(job=job, users=load, status=job_row["status"], patch_sha256=meta["sha256"],
                   scored_V204=measured(rungs[load]), comparison=measured(control),
                   comparison_run=private_16_id if load==16 else incumbent_id)
        records[name] = row
        passing_check = None
        if job_row["status"] in ("done", "failed", "cancelled", "canceled", "withdrawn"):
            record = saved(job) if (runs / (job+".json")).exists() else api("status",job)
            assert record["id"] == job and record["mode"] == "check" and record["team"] == current["team"]
            result = record.get("result") or {}
            row["passed"] = result.get("passed") is True
            row["result_status"] = result.get("status")
            row["memory"] = result.get("gpu_memory")
            if row["passed"]:
                assert record["serving_flags"] == meta["flags"] and result["patch"]["sha256"] == meta["sha256"]
                assert result["correctness"]["passed"] and result["concurrency"] == [load]
                rung, = result["rungs"]
                assert rung["concurrency"] == load
                row.update(measured=measured(rung), versus_comparison=compare(rung,control),
                           judge_vs_own_stock=result.get("vs_stock"), repetitions=rung["reps"])
                passing_check = result
        row["lineage_scenarios"] = lineage_scenarios(name, previous, previous_model, best, passing_check)
    model, _ = calibration(current)
    def coupled(factor, changed):
        vector = [factor if users in changed else 1.0 for users in loads]
        return model(vector,vector)
    assert abs(coupled(1.0,loads)-best) < 0.0001
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(), cutoff_utc="2026-11-20T08:00:00Z",
                incumbent=dict(job=incumbent_id, score=best, prior=previous["result"]["frontier"]["score"],
                    gpu_memory=current["result"]["gpu_memory"], official_points=current["result"]["frontier"]["points"]),
                limits=quota["limits"], jobs=records,
                V204_sensitivity={str(goal):{"all_four_multiplier":first_crossing(lambda x:coupled(x,loads),goal),
                                               "only_16_32_multiplier":first_crossing(lambda x:coupled(x,(16,32)),goal)}
                                  for goal in (2,3)},
                note="Read-only. Raw comparisons with V204 come from separate jobs. For each lineage scenario, the indicated loads take the same relative change on each rate axis versus V64. Other loads hold their V64 rates, with the measured V139 private four-user rung as a proxy for V141. The transfer assumptions and private-to-scored variation make these decision scenarios, not measured scores.")


if __name__ == "__main__":
    print(json.dumps(build(api("jobs"), api("status")), indent=2))
