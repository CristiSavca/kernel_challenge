"""Optional V229/16 selection from the exact Nov19 private result and measured same-load noise."""
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

V204_JOB = "7fcdf9abf39b486a"
V204_SHA = "3a091e426aa4a867e2a66eefdcf190cd52f11a68fd874684514681b80bbae88e"
V204_RECEIPT = "v168_or_v204_nov19_check16"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
# (directory, full patch hash, delta hash, candidate tree)
LINEAGE = {
    "V224": ("v224_packed_prefill_serving",
             "2de1b0195710e395b2d4c5e2666a8df7a42516c4bb3fc87ae313e0788f1e1a9a",
             "fd24066edbd7476589c813c788d15ec2dacc35f91cb3749518a20e7736f0feb7",
             "8d8af59c59922678e09adf6f0806cdf3c1ab8710"),
    "V225": ("v225_packed_draft_candidates",
             "734cc00c345e97f9b43e33487fc580ba79951a7cd0935c9ad9285b634f82969c",
             "4653fc0e33ed169bb68f73ccd1fc4c7cc39ff5a56d07b31dd704d8156328b285",
             "a328927d38a145ed9d9c627cdced2987be9fc67f"),
    "V228": ("v228_sharded_verify_projections",
             "9a4076c880c43d059660251e050fb64429595c67c657c60a61e607630307f901",
             "ae0bf6728ae363bd1e29f4076adbf502bc8c7339778580d8be63a9b15edbdc37",
             "0a66d4a6b1af96d87f58501388520b695582bbd0"),
    "V229": ("v229_sharded_decode_serving",
             "d81634df5c43230867d314540c7ae9ebe5ff8ba40c8fdc59482104ee3d75adcd",
             "cd09dc82e2bf4f8f79c4b64d4cddea69f387a25ced3c00462390e64835e98a98",
             "fab843fba01f208bed745d72b99c718f137f5785"),
}
V229_SHA, V229_TREE = LINEAGE["V229"][1], LINEAGE["V229"][3]
V219_SHA = "a053e233a6dc32e42258525e66e386414e5675fdbf0032a9cad56e4753c7bc64"
V219_TREE = "2cc74717bd297a9f0c22275d3b3f6e3af97f0623"
NOISE_JOB = "abd217baf0ed49fd"
RAW_SPREAD = 0.0163


def positive(x):
    return type(x) in (float, int) and math.isfinite(x) and x > 0


def plan_lineage():
    return {name: dict(directory=folder, patch_sha256=full, delta_sha256=delta, tree=tree)
            for name, (folder, full, delta, tree) in LINEAGE.items()}


def verify_lineage(base, candidate):
    if (candidate.directory, candidate.sha256) != LINEAGE["V229"][:2]:
        raise ValueError("V229 candidate identity changed")
    for name, (folder, full, delta, tree) in LINEAGE.items():
        directory = base / folder
        info = json.loads((directory / "validation.json").read_bytes())
        is_parent = name == "V224"
        parent = "V219" if is_parent else "V224"
        expected = dict(candidate=name, base=BASE, parent=parent, tree=tree, patch_sha256=full, delta_sha256=delta,
                        parent_sha256=V219_SHA if is_parent else LINEAGE["V224"][1],
                        parent_tree=V219_TREE if is_parent else LINEAGE["V224"][3], launch_flags=candidate.flags)
        if (any(info.get(k) != v for k, v in expected.items())
                or info.get("strict_apply_reverse_full_delta") is not True
                or sha256((directory / "patch.diff").read_bytes()).hexdigest() != full
                or sha256((directory / "delta.diff").read_bytes()).hexdigest() != delta):
            raise ValueError(f"{name} frozen export or lineage changed")
        if is_parent and info.get("all_six_orders_forward_and_reverse") is not True:
            raise ValueError("V224 donor composition changed")
        if name == "V229" and (info.get("both_orders_forward_and_reverse") is not True
                              or info.get("donors") != {n: LINEAGE[n][1] for n in ("V225", "V228")}):
            raise ValueError("V229 joint composition changed")


def frozen_inputs(root, candidate, reference_job):
    root = Path(root) / "serving_v0523"
    verify_lineage(root / "candidates", candidate)
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
        reference, _ = frozen_inputs(root, runner.V229, runner.v180.nov19.V68_JOB)
        checked = assessed(data, reference, proof, runner.earlier.passing_rung)
        if checked is not None:
            return runner.V229, checked
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        pass
    return runner.V198, None
