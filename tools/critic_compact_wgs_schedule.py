#!/usr/bin/env python3
"""Independent abstract compact-WGS partition/progress audit, not CUDA proof.

The model uses the proposed K128/128-row geometry and parent's cost tables.
An exact-source review must still show that the implementation matches it.
"""
import bisect
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = 264
STAGES = 32
COSTS = ([112, 135, 140, 158, 173, 185, 200, 207],
         [100, 104, 115, 123, 132, 144, 156, 174])


def jobs(histogram):
    result = []
    for count in histogram:
        result.extend([64] * (count // 64))
        if count % 64:
            result.append(count % 64)
    return result


def audit(counts, phase, grid=GRID):
    units_per_job = 64 if phase == 0 else 32
    stages_per_job = units_per_job * STAGES
    if not counts:
        # This explicit branch is mandatory; stage_at on nj=0 is undefined.
        return {"jobs": 0, "stages": 0, "split_units": 0, "max_shares": 0, "max_product": 0, "max_fixup_slot": -1}
    per_stage = [COSTS[phase][(count + 7) // 8 - 1] for count in counts]
    prefix = [0]
    for cost in per_stage:
        prefix.append(prefix[-1] + cost * stages_per_job)
    total = prefix[-1]

    def stage_at(cost_position):
        j = bisect.bisect_right(prefix, cost_position) - 1
        assert 0 <= j < len(counts)
        return j * stages_per_job + (cost_position - prefix[j]) // per_stage[j]

    boundaries = [stage_at(total * c // grid) for c in range(grid)] + [len(counts) * stages_per_job]
    assert boundaries[0] == 0
    assert all(a <= b for a, b in zip(boundaries, boundaries[1:]))
    owners = defaultdict(list)
    slot_to_unit = {}
    max_product = total * grid
    for c, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        # Reordering a trailing partial in phase A changes only local order.
        split = ((end - 1) // STAGES) * STAGES
        spans = [(split, end), (start, split)] if phase == 0 and end % STAGES and split > start else [(start, end)]
        visited = 0
        for begin, stop in spans:
            g = begin
            while g < stop:
                u = g // STAGES
                s0 = g % STAGES
                s1 = min(STAGES, s0 + stop - g)
                owners[u].append((c, s0, s1))
                visited += s1 - s0
                g += s1 - s0
        assert visited == end - start
    split_units = 0
    max_shares = 0
    max_slot = -1
    for u, shares in owners.items():
        ordered = sorted(shares, key=lambda x: x[1])
        assert ordered[0][1] == 0 and ordered[-1][2] == STAGES
        assert all(a[2] == b[1] for a, b in zip(ordered, ordered[1:]))
        assert sum(b - a for _, a, b in ordered) == STAGES
        if len(ordered) == 1:
            continue
        split_units += 1
        max_shares = max(max_shares, len(ordered))
        first_stage = u * STAGES
        j, local_stage = divmod(first_stage, stages_per_job)
        xe = prefix[j] + (local_stage + 1) * per_stage[j]
        # Mirrors the intended inverse including its corrections, in exact ints.
        owner = min(grid - 1, ((xe - 1) * grid) // total)
        while owner + 1 < grid and total * (owner + 1) // grid < xe:
            owner += 1
        while owner > 0 and total * owner // grid >= xe:
            owner -= 1
        assert owner == ordered[0][0]
        assert boundaries[owner] <= first_stage < boundaries[owner + 1]
        assert owner not in slot_to_unit or slot_to_unit[owner] == u
        slot_to_unit[owner] = u
        max_slot = max(max_slot, owner)
        if phase == 1:
            # Designated finisher contains stage zero. Its waits point to
            # strictly later CTAs. With all CTAs resident, this is acyclic.
            assert all(c > owner for c, _, _ in ordered[1:])
    assert len(owners) == len(counts) * units_per_job
    assert max_slot < 320
    return {"jobs": len(counts), "stages": boundaries[-1], "split_units": split_units,
            "max_shares": max_shares, "max_product": max_product, "max_fixup_slot": max_slot}


def main():
    rng = random.Random(601104)
    cases = [("zero", [0] * 32),
             ("near-max-jobs-M384", [321] * 9 + [129] + [1] * 22),
             ("near-max-jobs-M384-reversed", [1] * 22 + [129] + [321] * 9)]
    for m in (1, 8, 16, 17, 31, 32, 63, 64, 65, 127, 128, 255, 256, 383, 384):
        cases += [(f"one-expert-{m}", [m] + [0] * 31),
                  (f"eight-experts-{m}", [m] * 8 + [0] * 24),
                  (f"uniform32-{m}", [(8 * m) // 32 + (i < (8 * m) % 32) for i in range(32)])]
        for trial in range(24):
            # Every token has at most eight distinct local experts. Random
            # local counts include empty tokens; histograms obey count <= M.
            histogram = [0] * 32
            for _ in range(m):
                for e in rng.sample(range(32), rng.randrange(9)):
                    histogram[e] += 1
            cases.append((f"random-M{m}-{trial}", histogram))
    results = []
    maximum = {"max_product": 0, "max_shares": 0, "max_fixup_slot": -1, "jobs": 0}
    for name, histogram in cases:
        counts = jobs(histogram)
        assert len(counts) <= 88
        for phase in (0, 1):
            result = audit(counts, phase)
            for key in maximum:
                maximum[key] = max(maximum[key], result[key])
            results.append({"case": name, "histogram": histogram, "phase": phase, **result})
    # Conservative declared-array bound, beyond physically possible 384-token
    # histograms, catches integer overflow without calling it realistic input.
    bound_results = [audit([64] * 88, phase) for phase in (0, 1)]
    report = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": "K128, 128 output rows, 32 stages/unit, two compute WGs, FT_T64, 264 resident CTAs, parent cost tables.",
        "status": "pass",
        "histogram_cases": len(cases),
        "phase_audits": len(results),
        "maxima_over_physically_valid_cases": maximum,
        "conservative_88job_bound": bound_results,
        "invariants": ["Every logical stage assigned exactly once; empty partitions are valid.", "Each split unit's partial counts sum to 32.", "Inverse fixup owner is the CTA actually containing stage zero, even with rounded/empty partitions.", "Distinct split units do not alias one CTA boundary slot within a phase.", "Phase-B designated-finisher waits point only to strictly later CTA owners, hence have no cycle.", "Phase A uses nonwaiting last-contributor completion; all phase-B input dependencies therefore eventually become available if all producers can progress."],
        "limits": ["Abstract schedule proof, not a test of the candidate's actual CUDA source.", "Assumes every grid CTA can become resident; actual occupancy and PDL admission still require separate verification.", "Does not establish mbarrier byte counts, asynchronous shared-memory lifetimes, numerical correctness or speed.", "GPU ordering and progress depend on the exact release/acquire and deferred-publication implementation."],
    }
    path = ROOT / "experiments/iterations/20261104-compact-wgs-independent-schedule.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "pass", "histograms": len(cases), "phase_audits": len(results), "maxima": maximum, "bound": bound_results}))


if __name__ == "__main__":
    main()
