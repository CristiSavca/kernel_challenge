"""Aggregate public attention microbench profiles and the official V68 rung. No private request data."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
archive = ROOT / "experiments/candidates/muse_attention_decode"
src = next(archive.glob("e97e9e5*.py"))
assert hashlib.sha256(src.read_bytes()).hexdigest() == src.stem


def calls(path):
    fn = next(x for x in ast.parse(path.read_text()).body if getattr(x, "name", None) == "kernel_fn")
    named = {ast.unparse(n.func): n for n in ast.walk(fn) if isinstance(n, ast.Call)}
    attention = {k.arg: ast.unparse(k.value) for k in named["flash_attn_with_kvcache"].keywords}
    query = attention.pop("q")
    return attention, query, ast.dump(named["rms_norm_gate"])


old, old_q, old_tail = calls(ROOT / "muse_attention_decode/baseline.py")
new, new_q, new_tail = calls(src)
assert old == new and old_tail == new_tail
assert (old_q, new_q) == ("q.view(tokens, hq, d).to(FP8)", "q")
rung = json.loads((ROOT / "serving_v0523/runs/04b01f958aa5487c.json").read_text())["result"]["rungs"][0]
assert rung["concurrency"] == 16 and rung["truncated"] is False
pass_ms = 1000 * rung["acceptance"] / rung["tok_s_per_user_p50"]
required = {}
for speedup in (1.05, 2.833):
    ms = pass_ms * (1-1/speedup)
    required[str(speedup)] = {"net_ms_per_verify": ms, "net_us_per_eligible_layer": ms * 1000/62,
                              "net_us_per_removed_operation": ms * 1000/248}
results = []
for f in (next((archive/"profiles").glob(src.stem+".8727d7d1d2da427c.json")),
          ROOT/"muse_attention_decode/profiles/submission.702a17ec1ad94e03.json"):
    payload = json.loads(f.read_text())
    assert payload["summary"]["passed"]
    cases = {}
    weighted = 0.0
    for p in payload["profiles"]:
        c = p["case"]
        if c.get("load") != "verify 16":
            continue
        assert c["batch"] == 16 and c["q_len"] == 8 and c["page"] == 1 and c["heads"] == "tp8"
        m = re.search(r'^_prepare_qkv\s+([\d.]+) µs.*?1\s*$',p["report"],re.M)
        assert m
        b = payload["benchmarks"][p["index"]]
        assert b["event"] == "benchmark" and b["case"] == c
        delta = (b["baseline_ns"]-b["median_ns"])/1000
        cases[c["layer"]] = dict(weight=c["weight"], fused_producer_us=float(m[1]), baseline_whole_us=b["baseline_ns"]/1000,
                                 candidate_whole_us=b["median_ns"]/1000, whole_delta_us=delta)
        weighted += c["weight"] * delta / 1000
    assert set(cases)=={"sliding","full"} and cases["sliding"]["weight"]==46 and cases["full"]["weight"]==16
    results.append(dict(profile=str(f.relative_to(ROOT)),job=payload["job_id"],by_layer=cases,
                        weighted_whole_attention_delta_ms=weighted,
                        implied_speedup_if_critical_path_transfer_is_exact=pass_ms/(pass_ms-weighted)))
print(json.dumps(dict(
    reference_job="04b01f958aa5487c", actual_v182_h200=False, matched_public_kernel_source=str(src.relative_to(ROOT)),
    archive_evidence=results, v68_acceptance=rung["acceptance"], v68_user_p50_tokens_per_s=rung["tok_s_per_user_p50"],
    conditional_verify_period_ms=pass_ms,max_operations_per_pass=4*62,required_at_fixed_acceptance=required,
    limits=[
        "The profiled GPU region is another fused producer. The per-layer difference is a complete timed attention-call comparison; separate stock casts and physical scatter have no individual measurement in the matched reports.",
        "The 8727... source has the same FA3 arguments and norm-gate tail except for its ready FP8 query. Its fused producer uses an FP8 destination cast and allows live slot zero. V182 uses a BF16-to-FP8 lookup, checks current physical capacity and reserves zero.",
        "The matched attention microbench uses native FA3 and 128 query rows on one H200. V182 may dispatch to V68 paged verify. Scheduling, graph execution, rank imbalance, and concurrent prefill can change the serving delta.",
        "Mean acceptance divided by median per-user speed is an illustrative fixed-cycle proxy, not a measured GPU iteration. Serving and overall score changes require an actual private rung."
    ]),indent=2))
