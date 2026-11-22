"""Compare synchronized token ranges to V135's rank-local compact windows on synthetic completions."""
import json

import numpy as np

from check import BASE, BUDGET, EXPERTS, OWNERS, T, native_slots
from planner import plan_splits


def ranges(route_counts, cap):
    # Rows have first been compacted only for the native rank-local input.
    start = 0
    live = np.zeros(route_counts.shape[1], dtype=np.int64)
    for i in range(0, len(route_counts), 128):
        block = route_counts[i:i+128].sum(axis=0)
        if (block > cap).any():
            raise ValueError("a physical row tile exceeds the arena")
        if (live + block > cap).any():
            yield start, i
            start, live = i, np.zeros_like(live)
        live += block
    if start < len(route_counts):
        yield start, len(route_counts)


def partial_evidence(rng, grouped, extra_hot_peer):
    # Keep the *observed* rank total in every case, supply new synthetic IDs
    # for the unknown experts/owners/order. No archived original IDs are needed.
    n = T
    p = rng.random((n, 224), dtype=np.float32)
    if extra_hot_peer:
        p[:,:32] *= 0.55
    others = 32 + np.argpartition(p,6,axis=1)[:,:7]
    local = 4 + np.argpartition(rng.random((n,28),dtype=np.float32),6,axis=1)[:,:7]
    mask = np.zeros(n*7, dtype=bool)
    selected = np.arange(30697-n) if grouped else rng.choice(n*7,30697-n,replace=False)
    mask[selected] = True
    ids = np.column_stack((np.arange(n)%4,np.where(mask.reshape(n,7),local,others)))
    assert ((ids//32)==0).sum() == 30697
    return ids


def compare(ids):
    n, k = ids.shape
    ranks = len(BUDGET)
    cap = n + max(128*k,(n+7)//8)  # E_local/E = 1/8 here, explicitly target EP8 top-eight
    ownership = ids // 32
    counts = np.stack([(ownership==r).sum(axis=1) for r in range(ranks)], axis=1)
    native_parts, native_compute = [], []
    for r in range(ranks):
        compact = ids[counts[:,r]!=0]
        c = ((compact//32) == r).sum(axis=1)[:,None]
        pieces = list(ranges(c,cap))
        work = 0
        for lo,hi in pieces:
            bank = np.bincount(compact[lo:hi].ravel(), minlength=EXPERTS)[r*32:(r+1)*32]
            work += sum(BASE.issued(int(x)) for x in bank)
        native_parts.append(len(pieces))
        native_compute.append(work*BASE.owner_ns_per_issued_row + len(pieces)*BASE.launch_ns_per_move)
    native_ns = max(native_compute)
    pieces = list(ranges(counts,cap))
    combined_ns, moved = 0.0, 0
    window_rows = []
    for lo,hi in pieces:
        histogram = np.bincount(ids[lo:hi].ravel(),minlength=EXPERTS).tolist()
        plan = plan_splits(histogram, OWNERS, BUDGET, BASE)
        assert max(counts[lo:hi].sum(axis=0)) <= cap
        # Even a window which declines its own transfers participates in this
        # global schedule and its histogram/admission. The common baseline uses
        # one native pair per rank in each window, including the parent fences.
        combined_ns += plan.projected_ns + BASE.launch_ns_per_move
        if not plan.moves:
            combined_ns += BASE.global_planning_ns
        moved += sum(m.rows for m in plan.moves)
        window_rows.append(dict(tokens=hi-lo,owner0_live=int(counts[lo:hi,0].sum()),
                                helped=sum(m.rows for m in plan.moves)))
    return dict(cap=cap, rank_rows=counts.sum(axis=0).tolist(),native_windows=native_parts,
                common_windows=window_rows, native_expert_ms=round(native_ns/1e6,4),
                proposed_expert_ms=round(combined_ns/1e6,4), moved_rows=moved,
                reference_ratio=round(native_ns/combined_ns,5),
                would_admit=bool(moved and native_ns-combined_ns>BASE.minimum_saving_ns))


if __name__ == '__main__':
    rng = np.random.default_rng(186)
    cases = {name:compare(native_slots(rng,T,name)) for name in ('uniform','single_hot','four_local')}
    for clustered in (False, True):
        for other in (False, True):
            cases[f'known_total_grouped_{clustered}_hot_peer_{other}'] = compare(partial_evidence(rng,clustered,other))
    assert not cases['uniform']['would_admit']
    assert all(max(item['native_windows']) > 1 for k,item in cases.items() if k.startswith('known_total'))
    print(json.dumps(cases,indent=2))
