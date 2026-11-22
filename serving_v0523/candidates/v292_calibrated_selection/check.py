"""Read archived judge results to calibrate private evidence and time to a decision."""
import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import median


SERVING = Path(__file__).resolve().parents[2]
RUNS = SERVING / "runs"
CANDIDATES = SERVING / "candidates"
GEOMETRY = ("shape", "gpus", "image", "weights", "expected_cache_hit")


def read(path):
    return json.loads(path.read_bytes())


def exact(job):
    result = job["result"]
    assert (result.get("serving_flags") or {}) == (job.get("serving_flags") or {})
    return result["patch"]["sha256"], tuple(sorted((job.get("serving_flags") or {}).items()))


def passed(job):
    result = job["result"]
    return (job["status"] == "done" and result.get("passed") is True
            and result["correctness"]["passed"] and result["later_turns"]["passed"]
            and not result["short_outputs"])


def gains(ratio):
    return round(100 * (ratio - 1), 4)


def hours(job):
    return (job["finished_at"] - job["created_at"]) / 3600


def compare_rung(check, score):
    a, b = check["result"], score["result"]
    assert exact(check) == exact(score) and all(a[k] == b[k] for k in GEOMETRY)
    private, = a["rungs"]
    official, = [r for r in b["rungs"] if r["concurrency"] == private["concurrency"]]
    point, = [p for p in b["frontier"]["points"] if p["concurrency"] == private["concurrency"]]
    return dict(check=check["id"], score=score["id"], users=private["concurrency"],
                private_over_scored_raw_percent=gains(private["tok_s_per_gpu"] / official["tok_s_per_gpu"]),
                private_stock_point=a["vs_stock"]["rungs"][0]["at_stock_point"],
                scored_stock_point=point["ratio"], same_node=a["node"] == b["node"],
                private_memory=a["gpu_memory"]["over_stock_gb"], scored_memory=b["gpu_memory"]["over_stock_gb"])


def compare_checks(first, second):
    a, b = first["result"], second["result"]
    assert all(a[k] == b[k] for k in GEOMETRY + ("concurrency", "reps"))
    assert first["cluster"] == second["cluster"] and a["concurrency"] == [16]
    before, after = a["rungs"][0], b["rungs"][0]
    n_before = a["vs_stock"]["rungs"][0]["at_stock_point"]
    n_after = b["vs_stock"]["rungs"][0]["at_stock_point"]
    attrs = ("tok_s_per_gpu", "tok_s_per_user_e2e", "tok_s_per_user_p50",
             "ttft_first_p90_s", "ttft_later_p90_s")
    return dict(first=first["id"], second=second["id"], same_node=a["node"] == b["node"],
                flags_equal=exact(first)[1] == exact(second)[1],
                percent={k:gains(after[k]/before[k]) for k in attrs},
                stock_point_percent=gains(n_after/n_before), stock_point=[n_before,n_after])


def main():
    observation = read(CANDIDATES / "v291_mixed_ranked_attention/judge_observation.json")
    asof = datetime.strptime(observation["observed_utc"], "%Y-%m-%d %H:%M:%S UTC")
    cutoff = asof.replace(tzinfo=timezone.utc).timestamp()
    jobs = {}
    for path in sorted(RUNS.glob("*.json")):
        if re.fullmatch(r"[0-9a-f]{16}", path.stem):
            job = read(path)
            if job.get("finished_at") and job["finished_at"] <= cutoff:
                jobs[path.stem] = job
    assert all(j["id"] == key and j["team"] == "CristiSavca" and j["board"] == "serving"
               and j["duplicate_of"] is None for key,j in jobs.items())
    assert len(jobs) + len(observation["jobs"]) == observation["visible_run_count"]
    checks = [j for j in jobs.values() if j["mode"] == "check"]
    runs = [j for j in jobs.values() if j["mode"] == "run"]
    scores = [j for j in runs if passed(j)]
    good = [j for j in checks if passed(j)]
    pairs = [compare_rung(c,s) for c in good for s in scores if exact(c) == exact(s)]
    unscored = [j for j in good if all(exact(j) != exact(s) for s in scores)]
    files = {sha256(p.read_bytes()).hexdigest():str(p.relative_to(CANDIDATES))
             for name in ("*/patch.diff", "*/patch.checked*.diff") for p in sorted(CANDIDATES.glob(name))}
    known = {sha256((CANDIDATES / d / "patch.diff").read_bytes()).hexdigest(): d
             for d in ("v68_grouped_decode_verify","v64_humming_grouped_sequence_parallel")}
    assert all(exact(jobs[i])[0] in known for i in ("ac89fddc4ee147ab","04b01f958aa5487c"))
    frontier = jobs["abd217baf0ed49fd"]["result"]["frontier"]
    v68 = compare_checks(jobs["ac89fddc4ee147ab"],jobs["04b01f958aa5487c"])
    factor = v68["stock_point"][1] / v68["stock_point"][0]
    records = []
    for j in unscored:
        r = j["result"]
        chk = r["vs_stock"]["rungs"][0]
        ref = next(p for p in frontier["points"] if p["concurrency"] == chk["concurrency"])
        records.append(dict(job=j["id"], exact_sha256=exact(j)[0], flags=dict(exact(j)[1]),
                            matching_local_patch=files[exact(j)[0]],
                            users=chk["concurrency"], private_ratio=chk["at_stock_point"],
                            V64_scored_frontier_ratio=ref["ratio"]))
    delays = sorted(hours(j) for j in checks)
    scored_delays = sorted(hours(j) for j in runs)
    v68_rung = jobs["04b01f958aa5487c"]["result"]["rungs"][0]
    margin = next(r["spread"] for r in jobs["abd217baf0ed49fd"]["result"]["rungs"]
                  if r["concurrency"] == 16)
    print(json.dumps(dict(
        timestamp=observation["observed_utc"], source=observation,
        completed=dict(passing_private=len(good), private=len(checks), passing_scored=len(scores),
                       scored_attempts=len(runs)), exact_private_scored_pairs=pairs,
        unscored_private_passes=records, V64_to_V68=v68,
        same_node_private={
            "V62_to_V68":compare_checks(jobs["220305c15ded4f6b"],jobs["04b01f958aa5487c"]),
            "V53_to_V56":compare_checks(jobs["54654743532e4e50"],jobs["2eea52ae2db64189"])},
        V68_score_if_factor_transfers_to_one_or_all_four_points=[
            round(frontier["score"]*factor**.25,6),round(frontier["score"]*factor,6)],
        private_standby=dict(margin=margin, must_exceed_raw=v68_rung["tok_s_per_gpu"]*(1+margin),
                             must_exceed_stock_point=v68["stock_point"][1]*(1+margin)),
        historical_completion_hours=dict(private=[round(x,3) for x in delays],
            median_private=round(median(delays),3),private_within_eight=sum(h<=8 for h in delays),
            private_within_four=sum(h<=4 for h in delays),scored_including_one_refund=[round(h,3) for h in scored_delays],
            scored_successful=[round(h,3) for h in sorted(hours(j) for j in scores)])),
        indent=2))


if __name__ == "__main__":
    main()
