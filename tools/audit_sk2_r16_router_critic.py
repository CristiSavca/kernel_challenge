#!/usr/bin/env python3
"""Independent plan arithmetic only; no candidate, CUDA or performance model."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run():
    # The unchanged router stream is [16 K512 stages][32 row tiles][8 K64 warps][2048 hi+lo bytes].
    for pieces, stages_per_piece in ((4, 4), (8, 2)):
        seen = set()
        for cta in range(32 * pieces):
            tile, piece = divmod(cta, pieces)
            for stage in range(stages_per_piece):
                global_stage = piece * stages_per_piece + stage
                for warp in range(8):
                    unit = global_stage * 8 + warp
                    key = (tile, unit)
                    assert key not in seen
                    seen.add(key)
        assert seen == set(itertools.product(range(32), range(128)))
    permutation_count = 0
    for order in itertools.permutations(range(8)):
        ready = set()
        counter = 0
        for arrival, piece in enumerate(order):
            ready.add(piece)  # Models a published partial before the release increment.
            last = counter & 7 == 7
            counter += 1
            assert last == (arrival == 7)
            if last:
                assert len(ready) == 8
        permutation_count += 1
    wrap_old_values = [(0xFFFFFFF8 + i) & 0xFFFFFFFF for i in range(8)]
    assert [i for i, old in enumerate(wrap_old_values) if old & 7 == 7] == [7]
    # A prior four-way launch leaves residue4; the next eight-way launch publishes too early.
    mixed_last = [i for i in range(8) if (4 + i) & 7 == 7]
    assert mixed_last == [3]
    max_partial = (((31 * 8 + 7) * 32 + 31) * 8 + 7) * 4 + 4
    assert max_partial == 262144
    assert 32 * 8 * 2 * 8 * 2048 == 8388608

    # Fixed proposed choices, not a parameter sweep. Actual pmax and static allocation are runtime facts.
    shapes = []
    for name, tokr, nt, nwr, wsk, upw, ns, fused in (
        ("A_fused_pooler_8", 8, 2, 1, 8, 1, 4, True),
        ("A_fused_pooler_16", 16, 2, 1, 8, 1, 3, True),
        ("A_fused_pooler_32_compatibility", 32, 2, 1, 8, 1, 2, True),
        ("B_separate_pooler_32", 32, 2, 1, 4, 2, 2, False),
        ("C_QKV_8", 8, 2, 1, 4, 2, 5, False),
        ("C_QKV_16", 16, 2, 1, 4, 2, 3, False),
        ("C_QKV_32", 32, 2, 1, 4, 2, 2, False),
    ):
        nw, ups, rows = nwr * wsk, wsk * upw, nwr * nt * 8
        mt = 2 if tokr == 32 else 1
        toks, nesc = mt * 16, min(ns, 4)
        wby, xby, wby_router = nw * upw * nt * 768, ups * tokr * 128, nw * 2048
        weight_extent = max(wby, wby_router) if fused else wby
        sst = (xby + weight_extent + 1023) & ~1023
        cp = rows + 1
        dynamic_fixed = 1024 + ns * sst + 16 * ns + 32 + nesc * ((toks * cp + 3) & ~3) * 4 + 128
        rp = rows + ((40 - rows % 32) % 32)
        red_extent = wsk * toks * rp * 4
        assert red_extent <= ns * sst
        shapes.append(dict(name=name, TOKR=tokr, NT=nt, NWR=nwr, WSK=wsk, UPW=upw,
                           NS=ns, NESC=nesc, threads=(nw + 1 + nesc) * 32,
                           weight_copy_bytes=wby, router_weight_copy_bytes=wby_router if fused else None,
                           activation_reserved_bytes_per_stage=xby, stage_extent_bytes=sst,
                           ring_bytes=ns * sst, reduction_scratch_extent_bytes=red_extent,
                           dynamic_fixed_before_8_pmax=dynamic_fixed))
    return {
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Abstract plan arithmetic and counter schedule; not an actual-source proof or GPU memory-order proof.",
        "router_coverage": {"row_tiles": 32, "rows_per_tile": 8, "K64_units_per_tile": 128,
                            "old_CTA_groups": 4, "new_CTA_groups": 8, "new_stages_per_CTA": 2,
                            "exact_unit_bijection": True, "unchanged_router_weight_bytes": 8388608,
                            "unchanged_router_weight_copies": 512},
        "counter_model": {"all_eight_arrival_permutations": permutation_count,
                          "wraparound_residue_checked": True,
                          "mixed_four_then_eight_negative_control": "Counter starts4mod8; eighth-way last condition fires on arrival4, before all8 partials exist.",
                          "chosen_resolution": "Uniform eight-way routing for every FUSE TOKR8/16/32 specialization, including tuner alternatives; both ordinary and UV paths allocate eight-part buffers. This is a plan requirement, not yet an implemented proof."},
        "partial_buffer": {"old_bytes": 131072, "new_bytes": max_partial,
                           "fixed_token_stride": 32, "changes_required": "Both ordinary and UV allocation sites and all writer/reader rt/rq strides must agree."},
        "fixed_geometry_arithmetic": shapes,
        "lifetime_gates": [
            "Use max(pooler WBY, router WBYR) in device SST, host dynamic allocation and preparation geometry; transaction counts still use the exact payload copied.",
            "Keep all escape warps observing and returning every ring stage, including the router suffix.",
            "Keep cbar and consumer barriers before ring reuse for router partials and pooler output reduction.",
            "Keep existing PDL wait/trigger placement and the dependency that prevents routing from zeroing UV sum slots before all pooler readers finish.",
            "Non-last router contributors do not spin, so eight-way reduction itself does not need all256 CTAs resident; two-CTA capacity is the performance hypothesis gate.",
            "All new exact PK specializations need actual resource records and valid admission. Target A8/16, B32 and C8/16/32 require at least two blocks. Compile-time FUSE TOKR32 compatibility may admit one only if its compiled resources are clean and offline profiles establish it was not selected as a performance target; no phase detection belongs in the candidate.",
            "Preserve every BF16 hi/lo router term and FP32 accumulation. Eight-way reduction and changed escape grouping alter association; exact bitwise equality is not claimed."
        ],
        "performance_tradeoffs": [
            "A: same48MiB pooler weights and8MiB router weights; main weight copies2048to4096, router copies remain512; main activation requests double.",
            "A: grid128to256 and persistent MT1/NACC2 pooler accumulators32to16 words create a concrete possible latency-hiding mechanism, not a register allocation prediction.",
            "A: weight lookahead perSM assuming2CTAs falls144to96KiB atTOKR8 and120to72KiB atTOKR16; more stage slots do not mean more prefetched weight bytes.",
            "B: separateT32 pooler doublesgrid, halvesKstagecount, keeps4096 weight copies; activation requests double and Kpartial warp count increases2to4.",
            "C: QKV grid96to144 and activation/weight-copy requests increase50%; packed weight bytes and MMA work stay constant. Partial second-CTA population and tail costs prevent a gain inference from capacity alone."
        ],
        "decision": "Await the author's fixed source/lifecycle/resource plan. No implementation or official check is approved by this arithmetic record.",
        "gpu_used": False,
    }


if __name__ == "__main__":
    result = run()
    target = ROOT / "experiments/iterations/20261104-sk2-r16-router-feasibility-critic.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(target), "router_coverage": result["router_coverage"],
                      "counter_model": result["counter_model"], "fixed_geometry_arithmetic": result["fixed_geometry_arithmetic"]}, indent=2))
