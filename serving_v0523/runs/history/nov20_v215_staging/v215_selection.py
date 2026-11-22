"""Optional V215/16 selection from the exact Nov19 private result and measured same-load noise."""
import json
import math
from pathlib import Path
import subprocess

V204_JOB = "7fcdf9abf39b486a"
V204_SHA = "3a091e426aa4a867e2a66eefdcf190cd52f11a68fd874684514681b80bbae88e"
V204_RECEIPT = "v168_or_v204_nov19_check16"
V215_SHA = "bbf69a796f2e2ae2edb2b5d913f0e0dc799037d555c10dfe58d00eb71f7cd7b6"
V215_TREE = "e3803fe065ff36c7bdde740ae5b748ddb3d395a0"
V215_PARENT = "9c8eaeddf344558aec76e19f801e897e9f39438b9ddc3f900e7f88a05a03840b"
V215_DONOR = "6223b3cf65a3ac64ef46c45adb574621a8568e358b02c21d0815aa869f6c233f"
NOISE_JOB = "abd217baf0ed49fd"
RAW_SPREAD = 0.0163


def positive(x):
    return type(x) in (float, int) and math.isfinite(x) and x > 0


def frozen_inputs(root, candidate, reference_job):
    root = Path(root) / "serving_v0523"
    manifest = json.loads((root / "candidates" / candidate.directory / "validation.json").read_text())
    expected = dict(candidate="V215", patch_sha256=V215_SHA, tree=V215_TREE, parent="V214",
                    parent_sha256=V215_PARENT, expert_donor="V213", donor_sha256=V215_DONOR,
                    strict_both_donor_orders=True, strict_apply_reverse_full_delta=True, launch_flags=candidate.flags)
    if (any(manifest.get(k) != v for k,v in expected.items())
            or any(manifest.get(k) is not True for k in ("strict_both_donor_orders", "strict_apply_reverse_full_delta"))):
        raise ValueError("V215 frozen export or donor mismatch")
    score = json.loads((root / "runs" / (NOISE_JOB + ".json")).read_text())
    rung = [r for r in score["result"]["rungs"] if r.get("concurrency") == 16]
    if (score.get("id") != NOISE_JOB or score.get("board") != "serving" or score.get("mode") != "run"
            or score.get("status") != "done" or score["result"].get("passed") is not True or len(rung) != 1
            or rung[0].get("reps") != 3 or rung[0].get("spread") != RAW_SPREAD):
        raise ValueError("V64 measured /16 spread changed")
    ref = json.loads((root / "runs" / (reference_job + ".json")).read_text())
    return ref, dict(job=NOISE_JOB, concurrency=16, reps=3, spread=RAW_SPREAD,
                     comparison="strict raw and at_stock_point gains above the measured V64 /16 spread")


def assessed(data, reference, proof, passing_rung):
    result = data.get("result")
    if not isinstance(result, dict):
        return None
    fixed = reference["result"]
    if (data.get("id") != V204_JOB or data.get("board") != "serving" or data.get("mode") != "check"
            or data.get("status") != "done" or data.get("team") != reference.get("team")
            or data.get("duplicate_of") is not None or data.get("refunded", False) is not False
            or result.get("status") != "ok" or result.get("passed") is not True or result.get("concurrency") != [16]
            or result.get("patch", {}).get("sha256") != V204_SHA
            or result.get("serving_flags") != data.get("serving_flags")
            or not isinstance(result.get("node"), str) or not result["node"].strip()
            or data.get("cluster") != proof["comparison_cluster"] or data.get("dedicated") is not True
            or type(result.get("reps")) is not int or type(result.get("gpus")) is not int
            or any(result.get(k) != fixed.get(k) for k in ("shape", "image", "weights", "gpus", "reps", "expected_cache_hit"))
            or result.get("short_outputs") != 0 or (result.get("correctness") or {}).get("passed") is not True
            or (result.get("later_turns") or {}).get("ran") is not True
            or (result.get("later_turns") or {}).get("passed") is not True):
        return None
    rung = passing_rung(result)
    stock = result.get("vs_stock")
    ref_rung = proof["comparison_rung"]
    ref_stock = proof["comparison_vs_stock"]
    if (rung is None or rung.get("reps") != ref_rung["reps"] or not isinstance(stock, dict)
            or stock.get("indicative") is not True or not isinstance(stock.get("rungs"), list)
            or len(stock["rungs"]) != 1 or not isinstance(stock["rungs"][0], dict)
            or stock.get("ttft_ratio_limit") != ref_stock["ttft_ratio_limit"]):
        return None
    norm = stock["rungs"][0]
    matched = norm.get("stock")
    mem = result.get("gpu_memory")
    if (type(norm.get("concurrency")) is not int or norm["concurrency"] != 16 or not isinstance(matched, dict)
            or not all(positive(matched.get(k)) for k in ("tok_s_per_gpu", "tok_s_per_user_e2e",
                                                           "ttft_first_p90_s", "ttft_later_p90_s"))
            or not all(positive(norm.get(k)) for k in ("at_stock_point", "tok_s_per_gpu", "tok_s_per_user_e2e",
                                                       "ttft_first_p90", "ttft_later_p90"))
            or not isinstance(mem, dict) or mem.get("limit_gb") != fixed["gpu_memory"]["limit_gb"]
            or type(mem.get("over_stock_gb")) not in (float, int) or not math.isfinite(mem["over_stock_gb"])
            or mem["over_stock_gb"] > mem["limit_gb"]
            or any(not positive(rung.get(k)) for k in ("ttft_first_p90_s", "ttft_later_p90_s", "ttft_p90_s"))
            or any(norm[k] > stock["ttft_ratio_limit"] for k in ("ttft_first_p90", "ttft_later_p90"))
            or any(rung[k] > stock["ttft_ratio_limit"] * matched[k] for k in ("ttft_first_p90_s", "ttft_later_p90_s"))):
        return None
    raw_gain = rung["tok_s_per_gpu"] / ref_rung["tok_s_per_gpu"] - 1
    normalized_gain = norm["at_stock_point"] / ref_stock["rungs"][0]["at_stock_point"] - 1
    if min(raw_gain, normalized_gain) <= RAW_SPREAD:
        return None
    return dict(parent_job=V204_JOB, parent_sha256=V204_SHA, parent_node=result["node"], parent_cluster=data["cluster"],
                parent_rung=rung, parent_stock_rung=norm, parent_memory=mem, raw_gain=raw_gain,
                normalized_gain=normalized_gain, noise_control_job=NOISE_JOB, gain_floor=RAW_SPREAD)


def parent_request(runner):
    return runner.V204, 16, V204_RECEIPT, runner.v180.nov19.START, runner.v180.nov19.END


def select(runner, runs, root, proof, okc, bounded, now):
    """Missing or insufficient positive V204 evidence leaves the original V198 candidate selected."""
    req = parent_request(runner)
    try:
        if runner.slots.bound_receipt(runs, req) != V204_JOB:
            return runner.V198, None
        response = bounded([okc, "serve", "status", V204_JOB, "--json"], capture_output=True, text=True, timeout=60)
        if response.returncode:
            return runner.V198, None
        data = json.loads(response.stdout)
        identity = runner.v180.fallback.official(req, V204_JOB, okc, lambda *a, **k: response, runs)
        finished = data.get("finished_at")
        if (not identity or identity[0] != "passed" or not positive(finished)
                or not identity[2] <= finished <= now().timestamp()):
            return runner.V198, None
        reference, _ = frozen_inputs(root, runner.V215, runner.v180.nov19.V68_JOB)
        checked = assessed(data, reference, proof, runner.earlier.passing_rung)
        if checked is not None:
            return runner.V215, checked
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        pass
    return runner.V198, None
