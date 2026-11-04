#!/usr/bin/env python3
"""Offline source/profile coverage map; no imports or execution of a submission."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
JOB = "7a8c4941741645ee"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    source = ROOT / "experiments/candidates/muse_decode_layer" / (SOURCE_SHA + ".py")
    profile = ROOT / "experiments/candidates/muse_decode_layer/profiles" / (SOURCE_SHA + "." + JOB + ".json")
    assert digest(source) == SOURCE_SHA
    data = json.loads(profile.read_text())
    parsed = ast.parse(source.read_text())
    names = {"bfp_SK2_CONFIGS", "bfp_SK2_PICK", "bfp_SK2_K", "bfp__sk2_geom"}
    selected = [n for n in parsed.body if
                isinstance(n, ast.FunctionDef) and n.name in names or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    env = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<pure-geometry>", "exec"), env)
    geometry = {}
    for cfg in env["bfp_SK2_CONFIGS"]:
        name, layer, rows, r, nt, nwr, wsk, upw, nacc, cl = cfg
        for tokr in (8, 16, 32):
            if env["bfp_SK2_PICK"].get((layer, tokr)) != name and name != "pooler":
                continue
            hl, ns = env["bfp__sk2_geom"](cfg, tokr)
            mt = 2 if tokr == 32 else 1
            actual_nacc = 1 if mt == 2 and not hl and ns >= 4 and nt == 4 else nacc
            key = (mt, tokr, nt, nwr, wsk, upw, ns, not hl, hl, actual_nacc, cl)
            nw = nwr * wsk
            nesc = min(4, ns) if not hl else 1
            toks = 16 * mt
            sst = ((nw * upw * nt * (2048 if hl else 768) + wsk * upw * tokr * 128) + 1023) & ~1023
            rp = r + ((40 - r % 32) % 32)
            fixed = (1024 + ns * sst + 16 * ns + 32 + nesc * ((toks * (r + 1) + 3) & ~3) * 4
                     + ((cl - 1) * wsk * toks * rp * 4 if cl > 1 else 0) + 128)
            item = {"source_config": name, "layer": layer, "N": rows, "K": env["bfp_SK2_K"][layer],
                    "rows_per_CTA": r, "threads": (nw + 1 + nesc) * 32, "grid": rows // r * cl,
                    "dynamic_shared_without_escape_table": fixed,
                    "actual_dynamic_formula": str(fixed) + " + 8*pmax",
                    "ring_bytes": ns * sst,
                    "two_CTAs_excluded_by_dynamic_alone_at_233472B_per_SM": fixed * 2 > 233472}
            geometry.setdefault(key, []).append(item)
    captures = []
    unique = {}
    assert len(data["benchmarks"]) == len(data["profiles"]) == 12
    for bench, capture in zip(data["benchmarks"], data["profiles"]):
        assert bench["index"] == capture["index"]
        rows = []
        total = 0.0
        for line in capture["report"].splitlines():
            match = re.match(r"^(.*?)\s+(\d+\.\d+) µs\s+[\d.]+%\s+(\d+)\s*$", line)
            if not match:
                continue
            name, duration, calls = match.groups()
            total += float(duration)
            if "sk2_kernel<" not in name:
                continue
            tokens = name.split("sk2_kernel<", 1)[1].split(",")[:11]
            key = tuple(True if t.strip() == "true" else False if t.strip() == "false" else int(t.strip()) for t in tokens)
            assert len(key) == 11 and key in geometry
            candidates = geometry[key]
            assert len({g["layer"] for g in candidates}) == 1
            layer = candidates[0]["layer"]
            row = {"visible_prefix": list(key), "layer_from_source_prefix": layer,
                   "duration_us": float(duration), "calls": int(calls)}
            rows.append(row)
            entry = unique.setdefault(key, {"visible_prefix": list(key), "geometry": candidates, "captures": []})
            entry["captures"].append({"index": capture["index"], "duration_us": float(duration)})
        captures.append({"index": bench["index"], "case": bench["case"], "median_us": bench["median_ns"] / 1000,
                         "profile_row_sum_us": total, "SK2_rows": rows})

    def scenario(layers, reduction):
        loads = {}
        for c in captures:
            fraction = sum(r["duration_us"] for r in c["SK2_rows"] if r["layer_from_source_prefix"] in layers) / c["profile_row_sum_us"]
            base = c["median_us"] * c["case"]["weight"]
            pair = loads.setdefault(c["case"]["load"], [0.0, 0.0])
            pair[0] += base
            pair[1] += base * (1 - reduction * fraction)
        return 1 - math.exp(sum(math.log(after / before) for before, after in loads.values()) / len(loads))

    scenarios = {}
    for label, layers in (("all_SK2", {"o", "pooler", "qkv_gate", "unpooler", "router"}),
                          ("QKV_and_pooler", {"qkv_gate", "pooler"}), ("pooler", {"pooler"}),
                          ("QKV", {"qkv_gate"}), ("output_projection", {"o"})):
        low, high = 0.0, 1.0
        for _ in range(60):
            middle = (low + high) / 2
            if scenario(layers, middle) < .05:
                low = middle
            else:
                high = middle
        scenarios[label] = {"hypothetical_20pct_row_cut_decoder_reduction": scenario(layers, .2),
                            "hypothetical_50pct_row_cut_decoder_reduction": scenario(layers, .5),
                            "row_cut_needed_for_5pct_decoder": high if scenario(layers, 1) >= .05 else None}
    return {
        "source_sha256": SOURCE_SHA, "public_job": JOB, "profile_sha256": digest(profile),
        "verifier_sha256": digest(Path(__file__)), "SK2_profile_rows": sum(len(c["SK2_rows"]) for c in captures),
        "distinct_visible_prefixes": len(unique), "visible_prefix_fields": "MT,TOKR,NT,NWR,WSK,UPW,NS,COMP,HL,NACC,CL",
        "identities": list(unique.values()), "captures": captures, "scenarios_not_bounds_or_forecasts": scenarios,
        "matching_cautions": [
            "Truncated names hide OutT, PK, FUSE and BND; scalar runtime M, pmax, UV mode, PDL and generation are absent.",
            "The same pooler TOKR16 prefix and physical M16 occur in verify1 via UV2 and decode16 via ordinary fused mode0. Physical M alone does not resolve this collision.",
            "A record from a correctness prepare can describe the same compiled template but cannot establish the benchmark's dynamic allocation or phase identity.",
            "Static register/local/shared facts transfer only across an unambiguous exact compiled template/device; occupancy additionally requires the same dynamic shared allocation and threads.",
            "Profiles for T128 and T256 contain no SK2 rows. This diagnostic does not inspect their cuBLAS kernels or the UB2 small unpooler.",
            "Profile rows include PDL waits/overlap; separately captured benchmark medians and rounded row sums do not isolate computation."
        ],
        "highest_ROI_question": "Do the broadly used QKV and pooler specializations have unexpected local storage or unusually restrictive compiled resource use that supports a specific new compiler-state/lifetime hypothesis? Their shared rings already exclude two CTAs without a major geometry change, so low registers alone do not establish a residency opportunity.",
        "action_gate": "Use exact observed resources to rule out or constrain a separate falsifiable plan; do not infer a bottleneck, approve a quota sweep, revive failed escape/directory/codec rewrites, or promote the diagnostic score.",
        "gpu_used": False,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "experiments/iterations/20261104-selected-resource-coverage-critic.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("identities", "captures")}, indent=2))
