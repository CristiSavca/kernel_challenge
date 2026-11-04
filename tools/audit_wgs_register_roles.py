#!/usr/bin/env python3
"""CPU-only plan audit: register conservation, roles and join progress.

This models the proposed roles, not CUDA allocation or a candidate source.
"""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/iterations/20261104-wgs-register-roles-plan.json"


def audit(initial=96, producer_threads=128, idle_join=True):
    assert producer_threads == 128, "setmaxnreg producer role needs a full WG"
    quotas = [112] * 512 + [32] * producer_threads
    assert len(quotas) == 640
    donated = producer_threads * (initial - 32)
    requested = 512 * (112 - initial)
    assert donated >= requested, "CTA-local register pool is insufficient"
    assert sum(quotas) <= 640 * initial
    roles = {"consumer": list(range(512)), "weight": list(range(512, 544)),
             "activation": list(range(544, 576)), "idle": list(range(576, 640))}
    assert sorted(sum(roles.values(), [])) == list(range(640))
    for wg in range(5):
        assert len(set(quotas[128*wg:128*(wg+1)])) == 1
    bar2 = roles["consumer"] + roles["weight"] + roles["activation"]
    if idle_join:
        bar2 += roles["idle"]
    assert len(set(bar2)) == len(bar2) == 640, "barrier2 participant mismatch"
    assert len(set(roles["consumer"] + roles["activation"])) == 544

    # Explore every ordering of initial WG requests. A blocked consumer inc
    # cannot prevent the independently ready producer donation. Once donated,
    # exactly four acquisitions fit. Route/gather waits occur only afterwards.
    schedules = 0
    for order in itertools.permutations(range(5)):
        pool = 0
        acquired = set()
        blocked = set()
        donated_once = False
        for wg in order:
            if wg == 4:
                assert not donated_once
                pool += donated
                donated_once = True
            else:
                blocked.add(wg)
            for c in sorted(blocked):
                need = 128 * (112 - initial)
                if pool >= need:
                    pool -= need
                    acquired.add(c)
            blocked -= acquired
        assert acquired == set(range(4)) and not blocked and pool == donated-requested
        # All consumers may now complete their inherited consumer-only routing
        # barriers and arrive at7. The x warp then computes ranges and joins2.
        assert 4*128+32 == 544
        assert 4*128+32+32+64 == 640
        schedules += 1
    return dict(initial_registers=initial, pool=640*initial,
                final_registers=sum(quotas), donation=donated,
                acquisition=requested, request_orders=schedules,
                role_thread_counts={k:len(v) for k,v in roles.items()},
                barrier2=640, barrier7=544, consumer_barrier=512)


def main():
    result = audit()
    negative = {}
    for name, args in (("partial_producer_wg", {"producer_threads":64}),
                       ("initial88_pool", {"initial":88}),
                       ("missing_idle_join", {"idle_join":False})):
        try:
            audit(**args)
        except AssertionError as exc:
            negative[name] = str(exc)
        else:
            raise AssertionError(f"negative control passed: {name}")
    evidence = dict(verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    result=result, rejected_controls=negative, gpu_used=False,
                    candidate_source_tested=False,
                    assumptions=[
                        "Actual initial96 allocation and retained setmaxnreg instructions must be established separately.",
                        "Whole-WG role assignment is uniform; donation precedes every producer role split and wait.",
                        "A ready producer can eventually execute; the model is not a CUDA fairness/timing simulator.",
                        "The separate pinned NVIDIA-header proof establishes total allocation/admission, not actual role-to-register-partition mapping.",
                        "Inherited routing/gather/TMA lifetimes are unchanged; this model proves only the new role/join dependencies."])
    OUT.write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
