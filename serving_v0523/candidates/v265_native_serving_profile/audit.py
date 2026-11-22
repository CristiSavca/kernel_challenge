"""Cross-check the archived serving frontier and compare timed, hardware-matched controls."""
import hashlib
import json
import math
import re
import runpy
from pathlib import Path
from statistics import geometric_mean, median

ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "serving_v0523/candidates"
RUNS = ROOT / "serving_v0523/runs"
NODE = ROOT / "experiments/modal_node"
LOADGEN = ROOT / "tools/okc_node_loadgen.py"
frontier = runpy.run_path(str(LOADGEN))["frontier"]
HASHES = {}


def hashed(path):
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def json_file(path):
    return json.loads(hashed(path))


def scored_controls():
    all_runs = {}
    for path in sorted(RUNS.glob("????????????????.json")):
        run = json.loads(path.read_bytes())
        result = run["result"]
        if result.get("passed") and run["mode"] == "run":
            all_runs[path.stem] = json_file(path)
    assert len(all_runs) == 5
    return all_runs


def calibrate(scored):
    raw_rungs = scored["result"]["rungs"]
    official = scored["result"]["frontier"]
    assert not official["excluded"] and [r["concurrency"] for r in raw_rungs] == [1, 4, 16, 32]
    points = official["points"]

    def raw(fx, fy):
        curve = [dict(tok_s_per_gpu=r["tok_s_per_gpu"] * y, tok_s_per_user_e2e=r["tok_s_per_user_e2e"] * x)
                 for r, x, y in zip(raw_rungs, fx, fy)]
        return [frontier(curve, p["tok_s_per_user"]) / p["stock_tok_s_per_gpu"] for p in points]

    start = raw([1] * 4, [1] * 4)
    calibration = [p["ratio"] / v for p, v in zip(points, start)]

    def estimate(fx, fy):
        return geometric_mean(v * c for v, c in zip(raw(fx, fy), calibration))

    return estimate, dict(local_score=geometric_mean(start), official_score=official["score"],
                          calibration=calibration, per_point_local=start)


def thresholds(scored):
    model, _ = calibrate(scored)
    out = {}
    for goal in (3, 4):
        cutoffs = {}
        for x, y, name in [(True, True, "both_axes"), (False, True, "throughput_only"), (True, False, "interactivity_only")]:
            def score(multiplier):
                return model([multiplier if x else 1] * 4, [multiplier if y else 1] * 4)
            low, high = 1, 1000
            if score(high) >= goal:
                for _ in range(64):
                    mid = (low + high) / 2
                    if score(mid) >= goal:
                        high = mid
                    else:
                        low = mid
                cutoffs[name] = high
            else:
                cutoffs[name] = None
        out[goal] = cutoffs
    h = 0.00001
    base, sensitivities = model([1] * 4, [1] * 4), []
    for i, n in enumerate((1, 4, 16, 32)):
        bumped = [1 + h if j == i else 1 for j in range(4)]
        derivative = lambda v: math.log(v / base) / math.log1p(h)
        sensitivities.append(dict(users=n, throughput=derivative(model([1] * 4, bumped)),
                                  interactivity=derivative(model(bumped, [1] * 4)), both=derivative(model(bumped, bumped))))
    return dict(crossings=out, user_rate_only_plateau=model([1e3] * 4, [1] * 4),
                local_elasticities=sensitivities, log_difference_step=math.log1p(h))


def comparison(control, variant, loads):
    a, b = control["result"], variant["result"]
    assert (a["image"], a["weights"], a["gpus"], a["shape"]) == (b["image"], b["weights"], b["gpus"], b["shape"]) and b["gpus"] == 8
    ar, br = [{r["concurrency"]: r for r in j["rungs"]} for j in (a, b)]
    result = dict(control=control["id"], variant=variant["id"], same_node=a["node"] == b["node"],
                  same_launch_flags=control.get("serving_flags") == variant.get("serving_flags"), loads=[])
    for n in loads:
        p, q = ar[n], br[n]
        rec = dict(users=n, repetitions=[p["reps"], q["reps"]], reported_spread=[p["spread"], q["spread"]])
        labels = ("tok_s_per_gpu", "tok_s_per_user_e2e", "tok_s_per_user_p50", "ttft_first_p90_s", "ttft_later_p90_s")
        for k in labels:
            rec[k] = dict(control=p[k], variant=q[k], variant_over_control=q[k] / p[k])
        if all("vs_stock" in j for j in (a, b)):
            baselines = [next(r for r in j["vs_stock"]["rungs"] if r["concurrency"] == n) for j in (a, b)]
            rec["paired_stock_throughput_ratio_of_ratios"] = (
                (q["tok_s_per_gpu"] / baselines[1]["stock"]["tok_s_per_gpu"]) /
                (p["tok_s_per_gpu"] / baselines[0]["stock"]["tok_s_per_gpu"]))
            rec["indicative_frontier_ratio_of_ratios"] = baselines[1]["at_stock_point"] / baselines[0]["at_stock_point"]
        result["loads"].append(rec)
    return result


def profile_evidence():
    files = sorted(NODE.glob("*/*__server.log"))
    names = ("nsys-rep", "sqlite", "qdrep", "nsys", "pftrace")
    binary_traces = [p.relative_to(NODE).as_posix() for p in NODE.rglob("*") if p.is_file() and p.suffix.lstrip(".") in names]
    rx = re.compile(r"TP(\d+) EP(\d+)\] Prefill stage profile: (\d+) tokens, (\d+) layers, experts=(.*?), total ([\d.]+) ms; (.*)")
    profiles = []
    for p in files:
        lines = hashed(p).decode().splitlines()
        for line in lines:
            if (m := rx.search(line)):
                rank, ep, tokens, layers, backend, total, body = m.groups()
                assert rank == ep
                stages = dict((k, float(v)) for k, v in (s.rsplit(" ", 1) for s in body.split(", ")))
                profiles.append(dict(log=p.name, rank=int(rank), tokens=int(tokens), layers=int(layers),
                                     backend=backend, total_ms=float(total), stages_ms=stages))
    assert len(profiles) == 8 and {r["rank"] for r in profiles} == set(range(8))
    assert all(r["backend"] == "marlin (forced)" and r["tokens"] == 16384 for r in profiles)
    diagnostic = hashed(CAND / "v43_dense_prefill_profile/patch.diff").decode()
    assert "torch.cuda.is_current_stream_capturing()" in diagnostic and "torch.cuda.synchronize()" in diagnostic
    # Log line order is rank completion order. Rank sorting gives a stable table.
    return dict(local_node_server_logs=len(files), node_binary_traces=binary_traces, historical_profile=sorted(profiles, key=lambda r: r["rank"]),
                historical_experts_ms_range=[min(r["stages_ms"]["moe_experts"] for r in profiles), max(r["stages_ms"]["moe_experts"] for r in profiles)],
                scope="Instrumented first Marlin prefill on eight ranks. Branch excludes graph capture and synchronizes CUDA on every lap; elapsed times include first-use overhead. No eight-rank Humming stage measurement in these logs.")


def client_variation():
    sources = [json_file(next(NODE.glob("*/*" + v + "__load.json"))) for v in ("v41", "v44")]
    datasets = [{(t["rep"], t["turn"]): t for t in s["requests"] if t["load"] == 1} for s in sources]
    p, q = datasets
    assert len(p) == len(q) == 39 and p.keys() == q.keys()
    assert all(p[k]["asked"] == p[k]["completion"] == q[k]["asked"] == q[k]["completion"] for k in p)
    after = lambda row: row["dur"] - row["ttft"]
    exact = [k for k in sorted(p) if (p[k]["prompt"], p[k]["cached"]) == (q[k]["prompt"], q[k]["cached"])]
    paired = [after(p[k]) / after(q[k]) for k in p]
    reps = []
    for rep in range(3):
        keys = [(rep, t) for t in range(13)]
        reps.append(dict(rep=rep, old_over_new_cold_ttft=p[keys[0]]["ttft"] / q[keys[0]]["ttft"],
                         old_over_new_post_first=sum(after(p[k]) for k in keys) / sum(after(q[k]) for k in keys),
                         old_over_new_wall=max(p[k]["t0"] + p[k]["dur"] for k in keys) / max(q[k]["t0"] + q[k]["dur"] for k in keys)))
    return dict(paired_count=39, post_first_ratio_median=median(paired), matching_prompt_cache_count=len(exact),
                matching_prompt_cache_median=median(after(p[k]) / after(q[k]) for k in exact),
                repetitions=reps, scope="V41 and V44 private sessions on different machines. Parent-target decode code is outside V44's dense prefill geometry. Matching lengths do not establish equal generated text or acceptance.")


def main():
    hashed(LOADGEN)
    all_scored = scored_controls()
    reference = all_scored["abd217baf0ed49fd"]
    keys = ("ac89fddc4ee147ab", "220305c15ded4f6b", "04b01f958aa5487c", "6c023d17f18d477e", "cca1f9f724604a94")
    private = {k: json_file(RUNS / (k + ".json")) for k in keys}
    refs = {**all_scored, **private}
    def pair(a, b, n):
        return comparison(refs[a], refs[b], n)
    contrasts = {
        "V48b_V64_scored": pair("8ebed2bced8942ff", "abd217baf0ed49fd", (1, 4, 16, 32)),
        "V62_V68_private": pair("220305c15ded4f6b", "04b01f958aa5487c", (16,)),
        "V64_V68_private": pair("ac89fddc4ee147ab", "04b01f958aa5487c", (16,)),
        "V40_same_patch_different_SWA_flag": pair("6c023d17f18d477e", "cca1f9f724604a94", (16,)),
    }
    assert contrasts["V48b_V64_scored"]["same_node"] and contrasts["V62_V68_private"]["same_node"]
    assert not contrasts["V64_V68_private"]["same_node"] and contrasts["V64_V68_private"]["same_launch_flags"]
    a, b = private["6c023d17f18d477e"], private["cca1f9f724604a94"]
    assert a["result"]["patch"]["sha256"] == b["result"]["patch"]["sha256"] and a.get("serving_flags") != b.get("serving_flags")
    return dict(study="V265", public_incumbent=reference["result"]["frontier"]["score"],
                frontier_replays={k: calibrate(v)[1] for k, v in all_scored.items()}, score_scenarios=thresholds(reference),
                contrasts=contrasts, clients=client_variation(), native_profile=profile_evidence(), source_sha256=HASHES,
                hardware_added=False, new_submission=False)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
