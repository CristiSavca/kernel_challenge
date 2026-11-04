#!/usr/bin/env python3
"""Bounded manifest-driven WGS queue search; no CUDA generation or GPU use.

The independent arithmetic oracle retains every unsafe-lifetime reason. The
existing conservative event checker verifies its decisions and compares each
surviving schedule's mathematical trace with the pinned parent. Logical AF
storage is reported, never translated into physical registers or performance.
"""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path

import audit_wgs_wait2_lifetimes as checker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tools/wgs_queue_search_manifest.json"
DEFAULT_OUTPUT = ROOT / "experiments/iterations/20261104-wgs-queue-cpu-search.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_manifest(path):
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == 1
    for value in manifest["pinned_inputs"].values():
        assert sha256(ROOT / value["path"]) == value["sha256"], value["path"]
    assert (ROOT / manifest["pinned_inputs"]["lifetime_checker"]["path"]).resolve() == Path(checker.__file__).resolve()
    fixed = manifest["fixed"]
    assert {k: fixed[k] for k in (
        "mma_steps_per_group", "commit_groups_per_stage", "af_words_per_group",
        "weight_ring_slots", "activation_ring_slots", "threads",
        "dynamic_shared_bytes", "segment_end_wait_depth",
    )} == dict(mma_steps_per_group=2, commit_groups_per_stage=8,
               af_words_per_group=8, weight_ring_slots=3,
               activation_ring_slots=3, threads=576,
               dynamic_shared_bytes=231168, segment_end_wait_depth=0)
    # This version deliberately supports one bounded domain, not arbitrary CUDA.
    assert manifest["search_space"] == dict(wait_depth=list(range(8)),
        af_banks=[1, 2, 4, 8, 16], release_ci=list(range(8)))
    assert manifest["parent_schedule"] == dict(wait_depth=1, af_banks=2, release_ci=0)
    assert manifest["device_gate"]["max_reported_registers_per_thread"] == 96
    assert manifest["device_gate"]["min_actual_occupancy_ctas"] == 1
    assert manifest["coverage"]["start_sequences"] == list(range(6))
    assert manifest["coverage"]["ntt"] == list(range(1, 9))
    return manifest


def oracle_reasons(depth, banks, release_ci):
    """Closed-form necessary/sufficient lifetime conditions in this model.

    Power-of-two banks <=8 divide the eight-group stage; banks16 uses only
    bank0..7 and resets after eight. The minimum bank reuse distance is thus
    min(banks,8). Before decoding, the last `depth` committed groups may live.
    After ci's commit/wait, the old stage is retired iff ci+1 >= depth.
    These checks are independent; report both even when the event checker stops
    at the first hazard. They assume ordered group retirement and segment wait0.
    """
    assert banks in (1, 2, 4, 8, 16) and 0 <= depth <= 7 and 0 <= release_ci < 8
    reasons = []
    reuse = min(banks, 8)
    if depth >= reuse:
        reasons.append(dict(code="af-overwrite", wait_depth=depth,
            bank_reuse_distance=reuse,
            detail="A bank is decoded again while its previous committed group may still read it."))
    if release_ci + 1 < depth:
        reasons.append(dict(code="x-early-release", wait_depth=depth,
            new_stage_groups_committed=release_ci + 1,
            remaining_previous_stage_groups=depth - release_ci - 1,
            detail="Activation-slot credit returns before all previous-stage groups must retire."))
    return reasons


def call_checker(schedule, lengths, start, ntt):
    return checker.run(lengths, start, ntt,
        banks=schedule["af_banks"], depth=schedule["wait_depth"],
        release_ci=schedule["release_ci"])


def checker_failure(schedule, lengths=(2,), start=0, ntt=1):
    try:
        call_checker(schedule, lengths, start, ntt)
    except AssertionError as exc:
        details = exc.args[0] if exc.args else "unclassified assertion"
        code = details[0] if isinstance(details, tuple) else str(details)
        return dict(code=code, details=details)
    return None


def independent_expected_trace(lengths, start, ntt):
    # Flat logical K order and accumulator parity, independent of queue logic.
    return [(seq, step, ntt, (step // 2) % 2 if ntt == 1 else 0)
            for seq in range(start, start + sum(lengths)) for step in range(16)]


def register_evidence_checks(manifest):
    evidence = json.loads((ROOT / manifest["pinned_inputs"]["register_limit_evidence"]["path"]).read_text())
    rows = {x["regs_per_thread"]: x for x in evidence["rows"] if x["threads"] == 576}
    assert rows[96]["register_admissible_ctas"] == 1
    assert rows[97]["register_admissible_ctas"] == 0
    assert rows[112]["register_admissible_ctas"] == 0
    assert all(x["cuda_occ_error"] == 0 for x in rows.values())
    return dict(max_reported_registers_per_thread=96,
        evidence_at_96=rows[96], negative_at_97=rows[97], negative_at_112=rows[112],
        scope="Unmodified NVIDIA occupancy-header register-limit calculation, not an actual device query.")


def negative_controls():
    controls = []
    for name, schedule, expected in (
        ("non-power-of-two-bank-stage-reset", dict(af_banks=3, wait_depth=2, release_ci=1), "af-overwrite"),
        ("two-banks-depth-two", dict(af_banks=2, wait_depth=2, release_ci=1), "af-overwrite"),
        ("release-old-X-after-ci-zero", dict(af_banks=4, wait_depth=2, release_ci=0), "x-early-release"),
    ):
        failure = checker_failure(schedule)
        assert failure and failure["code"] == expected, (name, failure)
        controls.append(dict(name=name, schedule=schedule, detected=failure))
    # Independent trace oracle catches numerical-order mutations, not only
    # operand lifetime problems. NTT1 parity stays independent of AF bank index.
    trace = independent_expected_trace((2,), 0, 1)
    swapped = trace.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    wrong_parity = [(seq, step, ntt, (step // 2) & 3) for seq, step, ntt, _ in trace]
    assert swapped != trace and wrong_parity != trace
    controls += [dict(name="swap-first-two-K-steps", detected="trace differs"),
                 dict(name="couple-NTT1-accumulator-to-four-AF-banks", detected="trace differs")]
    return controls


def search(manifest, manifest_path):
    register_checks = register_evidence_checks(manifest)
    controls = negative_controls()
    coverage = manifest["coverage"]
    cases = list(itertools.product(coverage["segment_lengths"],
                                  coverage["start_sequences"], coverage["ntt"]))
    assert len({(tuple(a), b, c) for a, b, c in cases}) == len(cases)
    parent_traces = []
    for lengths, start, ntt in cases:
        parent, peak, flushes = call_checker(manifest["parent_schedule"], lengths, start, ntt)
        assert parent == independent_expected_trace(lengths, start, ntt)
        assert peak == (16 if sum(lengths) else 0)
        assert flushes == sum(bool(n) for n in lengths)
        parent_traces.append(parent)

    outcomes = []
    reason_counts = collections.Counter()
    grouped = collections.defaultdict(list)
    total_trace_cases = total_commit_groups = 0
    total_event_checks = len(cases) + 3  # parent cases and event-model negative controls
    domain = manifest["search_space"]
    for depth, banks, release in itertools.product(domain["wait_depth"], domain["af_banks"], domain["release_ci"]):
        schedule = dict(wait_depth=depth, af_banks=banks, release_ci=release)
        reasons = oracle_reasons(depth, banks, release)
        witness = checker_failure(schedule)
        total_event_checks += 1
        assert bool(witness) == bool(reasons), (schedule, witness, reasons)
        if witness:
            assert witness["code"] in {r["code"] for r in reasons}
        # Give a checker witness for every independent reason, including those
        # hidden behind an earlier error in the original configuration.
        for reason in reasons:
            isolated = dict(schedule)
            if reason["code"] == "af-overwrite":
                isolated["release_ci"] = 7
            else:
                isolated["af_banks"] = 8
            isolated_failure = checker_failure(isolated)
            total_event_checks += 1
            assert isolated_failure and isolated_failure["code"] == reason["code"]
            reason["isolated_witness_schedule"] = isolated
            reason["isolated_checker_failure"] = isolated_failure
            reason_counts[reason["code"]] += 1

        row = dict(schedule=schedule,
            decision="REJECT_UNSAFE_LIFETIME" if reasons else "LIFETIME_SAFE_DEVICE_UNKNOWN",
            rejection_reasons=reasons,
            first_checker_failure=witness,
            within_deeper_queue_hypothesis=depth >= 2,
            declared_af_words=8 * banks,
            addressable_af_banks_per_stage=min(banks, 8),
            maximum_logical_live_af_words=None,
            checked_trace_cases=0,
            parent_trace_equal=None,
            actual_compiled_registers="UNKNOWN",
            actual_occupancy="UNKNOWN",
            performance="UNKNOWN",
            ready_for_gpu=False)
        if not reasons:
            peak_max = groups = 0
            for case, parent in zip(cases, parent_traces):
                lengths, start, ntt = case
                trace, peak, flushes = call_checker(schedule, lengths, start, ntt)
                assert trace == parent
                assert peak == (8 * (depth + 1) if sum(lengths) else 0)
                assert flushes == sum(bool(n) for n in lengths)
                peak_max = max(peak_max, peak)
                groups += len(trace) // 2
            row.update(maximum_logical_live_af_words=peak_max,
                       checked_trace_cases=len(cases), parent_trace_equal=True,
                       checked_commit_groups=groups)
            total_trace_cases += len(cases)
            total_event_checks += len(cases)
            total_commit_groups += groups
            # Banks16 resets after 8 groups: abstract bank indices equal banks8.
            # Declared storage may compile differently; not device equivalence.
            abstract_key = (depth, min(banks, 8), release)
            grouped[abstract_key].append(schedule)
        outcomes.append(row)

    lookup = {(r["schedule"]["wait_depth"], r["schedule"]["af_banks"], r["schedule"]["release_ci"]): r for r in outcomes}
    selected = manifest["independently_chosen_experiment"]
    selected_row = lookup[(selected["wait_depth"], selected["af_banks"], selected["release_ci"])]
    assert selected_row["decision"] == "LIFETIME_SAFE_DEVICE_UNKNOWN"
    assert selected_row["maximum_logical_live_af_words"] == 24
    accepted = [r for r in outcomes if not r["rejection_reasons"]]
    return dict(schema_version=1, tool_sha256=sha256(Path(__file__)),
        manifest_path=str(manifest_path.relative_to(ROOT)), manifest_sha256=sha256(manifest_path),
        pinned_inputs=manifest["pinned_inputs"], fixed=manifest["fixed"],
        coverage=dict(**coverage, cases_per_surviving_schedule=len(cases)),
        summary=dict(configurations=len(outcomes), rejected=len(outcomes)-len(accepted),
            lifetime_safe_device_unknown=len(accepted),
            deeper_queue_lifetime_survivors=sum(r["within_deeper_queue_hypothesis"] for r in accepted),
            reasons=dict(reason_counts), trace_comparisons=total_trace_cases,
            commit_groups_compared=total_commit_groups, event_checker_calls=total_event_checks,
            abstract_behavior_classes=len(grouped), gpu_ready=0),
        independently_chosen_experiment=selected_row,
        selection_note="The four-bank/wait2/ci1 experiment was chosen and implemented independently before this wrapper. This search does not rank predicted speed or authorize further variants.",
        register_gate=register_checks, device_gate=manifest["device_gate"],
        negative_controls=controls,
        abstract_aliases=[dict(key=list(key), schedules=value) for key, value in grouped.items() if len(value)>1],
        limits=["Finite abstract event model plus an independent arithmetic lifetime oracle; not a Lean proof or actual-source validation.",
                "Conservative ordered retirement assumes unchanged producer/barrier participants, segment wait0, FCI3 and ring protocol.",
                "Trace equality covers modeled K order/NTT1 accumulation parity, not all floating-point operands, CUDA numerical results or whole-grid interleavings; starting sequences span two logical ring generations but barrier phases themselves are not simulated.",
                "A safe lifetime does not imply useful overlap, freedom from serialization, resource admission or speedup.",
                "Logical live/declaration words do not predict physical registers; bank16 has 128 declared words but only banks0..7 are referenced in this model.",
                "The register limit is necessary only. Exact-source actual attributes and occupancy queries remain required for both specializations.",
                "No external repository code, CUDA candidate, GPU job, model provider or trained performance predictor was executed."],
        gpu_used=False, candidate_written=False, outcomes=outcomes)


def main():
    if not __debug__:
        raise RuntimeError("Run without -O: the pinned checker uses assertions for safety validation.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = checked_manifest(manifest_path)
    result = search(manifest, manifest_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), **result["summary"]), indent=2))


if __name__ == "__main__":
    main()
