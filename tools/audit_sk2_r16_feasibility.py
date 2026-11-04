#!/usr/bin/env python3
"""Bounded source-derived arithmetic for one SK2 R16 family; no candidate/GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
DIAG = "e444f46f945e03bb42dab8d80ed4869f8a7da6464f845ac09ac3cf8ea19473fd"


def geometry(*, tokr, nt, nwr, wsk, upw, ns, n, k=8192, pmax, fused=False, parts=0, static=0):
    mt, nw, r, ups = (2 if tokr == 32 else 1), nwr * wsk, nwr * nt * 8, wsk * upw
    nesc, toks = min(ns, 4), 16 * mt
    xby, wby = ups * tokr * 128, nw * upw * nt * 768
    wbyr = nw * 2048 if fused else 0
    sst = (xby + max(wby, wbyr) + 1023) & ~1023
    corr = nesc * ((toks * (r + 1) + 3) & ~3) * 4
    dynamic = 1024 + ns * sst + 16 * ns + 32 + corr + pmax * 8 + 128
    stages, grid = k // (ups * 64), n // r
    router_stages = 16 // parts if fused else 0
    return dict(TOKR=tokr, R=r, NT=nt, NWR=nwr, WSK=wsk, UPW=upw, NS=ns, MT=mt,
        consumer_warps=nw, producer_warps=1, escape_warps=nesc, threads=(nw + 1 + nesc) * 32,
        grid=grid, main_stages_per_CTA=stages, router_stages_per_CTA=router_stages,
        stage_bytes=sst, main_stage_weight_bytes=wby, router_stage_weight_bytes=wbyr,
        pmax=pmax, dynamic_shared_bytes=dynamic, static_planning_bytes=static,
        allocated_shared_with_1KiB_driver_reserve=(dynamic + static + 1024 + 255) // 256 * 256,
        main_weight_bytes=n * k * 3 // 2,
        main_activation_requested_bytes_per_physical_token=grid * k * 2,
        router_weight_bytes=grid * router_stages * wbyr,
        router_activation_requested_bytes_per_physical_token=grid * router_stages * ups * 64 * 2,
        main_stage_copy_pairs=grid * stages, router_stage_copy_pairs=grid * router_stages,
        empty_barrier_arrivals=grid * (stages + router_stages) * (nw + nesc),
        launch_weight_ring_lookahead_bytes=grid * ns * wby,
        router_partial_allocation_bytes=32 * parts * 32 * 8 * 4,
        router_partial_store_bytes=grid * toks * 8 * 4 if fused else 0,
        router_partial_read_bytes_per_physical_token=32 * parts * 8 * 4,
        router_counter_atomics=grid if fused else 0)


def run():
    parent = ROOT / f"experiments/candidates/muse_decode_layer/{SOURCE}.py"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == SOURCE
    resources = json.loads((ROOT / "experiments/iterations/20261104-decoder-selected-resource-analysis.json").read_text())
    assert resources["source_sha256"] == DIAG and resources["run_audit_gate"]["passed"]
    reg_path = ROOT / "experiments/iterations/20261104-sk2-r16-register-admission-critic.json"
    reg = json.loads(reg_path.read_text())
    ceilings = {r["threads"]: r["maximum_reported_registers_for_at_least_two_blocks"] for r in reg["thresholds"]}
    # Bounds from this diagnostic's actual tables only. Pool R16 divides oldR32;
    # QKV R16 can intersect two oldR24 blocks, so oldpmax44 implies newpmax<=88.
    specs = [
        ("A_fused8", dict(tokr=8, nt=2, nwr=1, wsk=8, upw=1, ns=4, n=4096, pmax=56, fused=True, parts=8), "SK2:4"),
        ("A_fused16", dict(tokr=16, nt=2, nwr=1, wsk=8, upw=1, ns=3, n=4096, pmax=56, fused=True, parts=8), "SK2:20"),
        ("A_fused32_compatibility", dict(tokr=32, nt=2, nwr=1, wsk=8, upw=1, ns=2, n=4096, pmax=56, fused=True, parts=8), None),
        ("B_separate_pooler32", dict(tokr=32, nt=2, nwr=1, wsk=4, upw=2, ns=2, n=4096, pmax=56), "SK2:24"),
        ("C_qkv8", dict(tokr=8, nt=2, nwr=1, wsk=4, upw=2, ns=5, n=2304, pmax=88), "SK2:2"),
        ("C_qkv16", dict(tokr=16, nt=2, nwr=1, wsk=4, upw=2, ns=3, n=2304, pmax=88), "SK2:18"),
        ("C_qkv32", dict(tokr=32, nt=2, nwr=1, wsk=4, upw=2, ns=2, n=2304, pmax=88), "SK2:22"),
    ]
    rows = []
    observed = {r["id"]: r for r in resources["resources"]["records"] if r["kind"] == "SK2"}
    for label, spec, old_id in specs:
        # Conservative new static reservation, not a measured new function attribute.
        new = geometry(**spec, static=1024)
        next_stage = geometry(**dict(spec, ns=spec["ns"] + 1), static=1024)
        assert 2 * new["allocated_shared_with_1KiB_driver_reserve"] <= 233472
        assert 2 * next_stage["allocated_shared_with_1KiB_driver_reserve"] > 233472
        new["reported_register_ceiling_for_two_CTAs"] = ceilings[new["threads"]]
        new["new_compiled_registers_and_actual_occupancy"] = "UNKNOWN"
        item = {"family": label, "proposed": new, "one_more_stage_exceeds_half_SM": True}
        if old_id:
            record = observed[old_id]
            t, x, a = record["template"], record["launch"], record["attributes"]
            old = geometry(tokr=t["TOKR"], nt=t["NT"], nwr=t["NWR"], wsk=t["WSK"], upw=t["UPW"], ns=t["NS"],
                n=x["N"], k=x["K"], pmax=x["pmax"], fused=bool(t["FUSE"]), parts=4 if t["FUSE"] else 0,
                static=a["static_shared_bytes"])
            assert old["dynamic_shared_bytes"] == x["dynamic_shared_bytes"] and old["threads"] == x["threads"]
            old.update(observed_registers=a["registers_per_thread"], observed_local_bytes=a["local_bytes_per_thread"],
                       observed_blocks_per_SM=record["blocks_per_SM"], observed_record_id=old_id)
            item["parent"] = old
            item["ratios_not_latency_forecasts"] = {key: new[key] / old[key] for key in (
                "grid", "main_weight_bytes", "main_activation_requested_bytes_per_physical_token", "main_stage_copy_pairs",
                "empty_barrier_arrivals", "launch_weight_ring_lookahead_bytes")}
        rows.append(item)
    # Every router tile/stage/unit must still appear once, including all hi/lo payloads.
    domains = []
    for parts in (4, 8):
        visits = [(cta // parts, (cta % parts) * (16 // parts) + j, wk)
                  for cta in range(32 * parts) for j in range(16 // parts) for wk in range(8)]
        assert len(visits) == len(set(visits)) == 32 * 16 * 8
        assert set(visits) == {(rt, st, wk) for rt in range(32) for st in range(16) for wk in range(8)}
        domains.append({"parts": parts, "tile_stage_unit_visits": len(visits), "one_visit_each": True})
    # Mixed4→8 counter schemas demonstrably fail; uniform8 preserves last-arrival across wrap.
    mixed = [(4 + arrival) & 7 for arrival in range(8)]
    assert mixed.index(7) == 3
    for start in (0, 8, 16, 0xFFFFFFF8):
        assert [j for j in range(8) if ((start + j) & 7) == 7] == [7]
    roi = json.loads((ROOT / "experiments/iterations/20261104-selected-resource-coverage-critic.json").read_text())
    return {"source_sha256": SOURCE, "resource_source_sha256": DIAG, "families": rows,
        "register_admission_primary_evidence": reg["primary_url"], "register_admission_header_sha256": reg["header_sha256"],
        "router_coordinate_coverage": domains,
        "counter_policy": "Uniform8way for ALL FUSE TOKR8/16/32 calls, including tuner alternatives; no mixed4way users or per-call reset",
        "mixed_counter_first_wrong_last_arrival": 4,
        "ROI_conditional_profile_scenarios_not_forecasts": roi["scenarios_not_bounds_or_forecasts"]["QKV_and_pooler"],
        "scope": "CPU arithmetic/coordinate audit only; no future candidate, numerical proof, compiler-resource measurement or GPU experiment",
        "decision": "Feasible resource hypothesis with substantial traffic/lookahead tradeoffs; conditional plan-review gate only, no implementation approval"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "experiments/iterations/20261104-sk2-r16-feasibility.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"saved": str(out.relative_to(ROOT)), "decision": result["decision"], "families": [
        {"name": r["family"], "threads": r["proposed"]["threads"], "dynamic": r["proposed"]["dynamic_shared_bytes"],
         "shared_with_reserves": r["proposed"]["allocated_shared_with_1KiB_driver_reserve"],
         "register_ceiling": r["proposed"]["reported_register_ceiling_for_two_CTAs"]} for r in result["families"]]}, indent=2))
