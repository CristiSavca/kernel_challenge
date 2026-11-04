#!/usr/bin/env python3
"""Plan-only conservative WGMMA lifetime model; no CUDA or candidate edits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py"
OUT = ROOT / "experiments/iterations/20261104-wgs-wait2-lifetimes.json"


def run(lengths, start_seq, ntt, banks=4, depth=2, release_ci=1):
    pending = []
    live_x = {}
    seq = start_seq
    releases = []
    trace = []
    peak = 0
    flushes = 0
    for nstage in lengths:
        if not nstage:
            continue  # empty ranges do not enter wgs_seg
        assert not pending  # preceding segment's wait<0>
        prev = None
        for stage in range(nstage):
            slot = seq % 3
            assert slot not in live_x, ("x-overwrite", seq, live_x)
            live_x[slot] = seq
            for ci in range(8):
                bank = ci % banks
                assert not any(g[0] == bank for g in pending), ("af-overwrite", seq, ci, pending)
                # Decode occurs before commit; count the current two-MMA bank
                # alongside every still-live committed bank.
                peak = max(peak, 8 * (len(pending) + 1))
                for jj in range(2):
                    trace.append((seq, 2 * ci + jj, ntt, ci & 1 if ntt == 1 else 0))
                pending.append((bank, seq, ci))
                # Worst legal completion: wait completes only the oldest groups
                # necessary to retain `depth`. Earlier completions cannot worsen
                # an input-lifetime hazard.
                pending = pending[-depth:] if depth else []
                if ci == release_ci and prev is not None:
                    assert not any(g[1] == prev for g in pending), ("x-early-release", seq, ci, pending)
                    del live_x[prev % 3]
                    releases.append(prev)
                    prev = None
                if stage == 0 and ci == 3:
                    # In source, pd refers only to the preceding fully drained
                    # segment; flush does not access current d/af/x registers.
                    flushes += 1
            prev = seq
            seq += 1
        pending.clear()  # source segment-end wait<0>, before generic d access
        assert prev is not None
        del live_x[prev % 3]
        releases.append(prev)
    assert not live_x and not pending
    assert releases == list(range(start_seq, seq))
    return trace, peak, flushes


def main():
    cases = groups = flushes = 0
    for start in range(6):  # all ring slots and both barrier generations
        for ntt in range(1, 9):
            for first in range(17):
                for second in range(17):
                    lengths = (first, second, 1, 16)  # partial/full/phase-tail segments
                    got, peak, nf = run(lengths, start, ntt)
                    parent, parent_peak, _ = run(lengths, start, ntt, banks=2, depth=1, release_ci=0)
                    assert got == parent  # K order and NTT1 accumulator selection
                    assert peak == 24 and parent_peak == 16
                    cases += 1
                    groups += len(got) // 2
                    flushes += nf
    failures = []
    for name, args in (
        ("three-banks-reset-every-stage", dict(banks=3, depth=2, release_ci=1)),
        ("two-banks-with-depth-two", dict(banks=2, depth=2, release_ci=1)),
        ("old-x-release-ci0", dict(banks=4, depth=2, release_ci=0)),
    ):
        try:
            run((2,), 0, 1, **args)
        except AssertionError as exc:
            failures.append(dict(case=name, detected=str(exc)))
        else:
            raise AssertionError(f"negative control did not fail: {name}")
    result = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        candidate_created=False, gpu_used=False, cases=cases, commit_groups=groups,
        deferred_flush_boundaries=flushes, peak_live_af_words=24,
        declared_af_words=32, parent_peak_live_af_words=16,
        instruction_trace_equal=True, negative_controls=failures,
        assumptions=["Conservative ordered group retirement allowed by wait_group N.",
                     "Every segment retains wait_group0 before generic accumulator accesses.",
                     "Actual source producer/barrier participants and all pending-work semantics remain inherited.",
                     "This abstract model is not candidate-source validation, CUDA compilation, full-grid simulation, or performance evidence.",
                     "Peak logical AF words do not predict physical register allocation or compiler serialization."])
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
