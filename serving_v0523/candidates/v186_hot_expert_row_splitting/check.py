"""Run the planner, distinguish archived observations from synthetic completions, and check slot return."""

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys
import time

import numpy as np

from planner import Costs, plan_splits

ROOT = Path(__file__).resolve().parents[3]
MIB = 1 << 20
H = 4096
T = 16384
RANKS = 8
EXPERTS = 256
OWNERS = [e // 32 for e in range(EXPERTS)]
# Two scalar FP32 scales accompany all the packed E2M1 bytes and E4M3 group scales.
BANK = 3 * H * H // 2 + 3 * H * H // 16 + 8
# These are explicit assumptions for the reference model, never inferred from a user or a request.
BASE = Costs(176, 165, 165, 225, 30_000, 100_000, 50_000, BANK, H, H, 32 * MIB)
BUDGET = [256 * MIB] * RANKS


def archives():
    rows, timed, stress = [], [], []
    for path in sorted((ROOT / 'experiments/modal_serving').glob('*/result.json')):
        data = json.loads(path.read_text())
        if data.get('gpu') != 'NVIDIA H200':
            continue
        for case in data.get('cases', []):
            if 'local_routes' not in case:
                continue
            rows.append((case['T'], case['local_routes']))
            if case['T'] == T:
                timed.append(dict(source=str(path.relative_to(ROOT)), routes=case['local_routes'],
                                  stock_us=case['stock_us'], dense384_us=case.get('dense384_us'),
                                  isolated_bf16_us=case['dense_ceiling']['us']))
        bcg = data.get('bcg_dense384', {})
        if 'trial2' in bcg:
            stress.append(dict(source=str(path.relative_to(ROOT)), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               trials=[bcg['trial' + str(i)] for i in range(3)]))
    assert len(stress) == 1 and len(timed) == 4
    hot = stress[0]['trials'][2]['local_routes']
    assert hot == 30697 and all(row['routes'] == 16224 for row in timed)
    return dict(h200_timed_rank_counts=sorted(set(rows)), h200_16k=timed, h200_graph_replay=stress,
                hot_fraction_of_mean=hot/T, some_other_rank_at_most=(T*8-hot)//7,
                forced_hot_expert_at_least=math.ceil(T/4), ordinary_checked_all_rank_histograms=0)


def even(total, length):
    q, r = divmod(total, length)
    return [q + (i < r) for i in range(length)]


def completion(peak_other):
    # Two possible full count vectors extending the single archived owner observation.
    known_owner = even(T, 4) + [0] * 28
    known_owner = [a + b for a, b in zip(known_owner, even(30697 - T, 32))]
    rest = T * 8 - sum(known_owner)
    totals = [peak_other] + even(rest-peak_other, 6) if peak_other else even(rest, 7)
    return known_owner + [v for n in totals for v in even(n, 32)]


def native_slots(rng, n, kind):
    # Inputs are fresh synthetic top-k IDs, never serving conversations.
    p = rng.random((n, EXPERTS), dtype=np.float32)
    ids = np.argpartition(p, 7, axis=1)[:, :8]
    if kind == 'four_local':
        hot = np.argpartition(p[:, :32], 3, axis=1)[:, :4]
        cold = np.argpartition(p[:, 32:], 3, axis=1)[:, :4] + 32
        ids = np.concatenate((hot, cold), axis=1)
    elif kind == 'single_hot':
        ids[:, 1:] = np.argpartition(p[:, 1:], 6, axis=1)[:, :7] + 1
        ids[:, 0] = 0
    elif kind == 'duplicate_slots':
        ids[:, 0:2] = 3
        ids[::3, 4] = -1
    return ids


def report_plan(counts, owners, budget, costs):
    plan = plan_splits(counts, owners, budget, costs)
    return dict(moves=[asdict(m) for m in plan.moves], moved_rows=sum(m.rows for m in plan.moves),
                max_routes=max(sum(n for n, o in zip(counts, owners) if o == r) for r in range(len(budget))),
                projected_expert_component_ms=round(plan.projected_ns/1e6, 5),
                original_expert_component_ms=round(plan.original_ns/1e6, 5),
                ratio=round(plan.original_ns / plan.projected_ns, 5) if plan.projected_ns else 1.0,
                max_reserved_mib=round(max(plan.extra_bytes_by_rank)/MIB, 3),
                total_wire_mib=round(sum(costs.transfer_bytes(m.rows) for m in plan.moves)/MIB, 3))


def round_bf16(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def check_slot_return(rng, ids, owners, costs, budget):
    flat = ids.ravel()
    valid = np.flatnonzero(flat >= 0)
    order = valid[np.argsort(flat[valid], kind='stable')]
    counts = np.bincount(flat[valid], minlength=len(owners)).tolist()
    starts = np.cumsum([0] + counts[:-1])
    plan = plan_splits(counts, owners, budget, costs)
    assert plan == plan_splits(counts, owners, budget, costs)
    assigned = np.full(flat.size, -1)
    for m in plan.moves:
        assert owners[m.expert] == m.owner and m.owner != m.helper
        slots = order[starts[m.expert]+m.begin:starts[m.expert]+m.end]
        assert len(slots) == m.rows and np.all(assigned[slots] == -1)
        assigned[slots] = m.helper
    is_local = assigned[order] == -1
    owner_ids = np.array(owners)[flat[order]]
    assigned[order[is_local]] = owner_ids[is_local]
    assert np.all(assigned[valid] >= 0) and np.all(assigned[flat < 0] == -1)
    # Synthetic rowwise BF16 W2 result. Identical immutable bank and complete K
    # are the calling contract. Rows are returned by original slot, even for duplicates.
    x = round_bf16(rng.normal(size=(ids.shape[0], 4)))
    w = round_bf16(rng.normal(size=(len(owners), 4, 4)))
    def projected(slots):
        return round_bf16(np.einsum('bi,bij->bj', x[slots // ids.shape[1]], w[flat[slots]], optimize=False))
    y = np.zeros((flat.size, 4), dtype=np.float32)
    y[valid] = projected(valid)
    received = np.zeros_like(y)
    writes = np.zeros(flat.size, dtype=np.uint8)
    ranks = list(reversed(range(len(budget))))
    for worker in ranks:
        slots = valid[assigned[valid] == worker][::-1]
        received[slots] = projected(slots)
        writes[slots] += 1
    assert np.all(writes[valid] == 1) and np.all(writes[flat < 0] == 0)
    assert np.array_equal(y.view('u4'), received.view('u4'))
    scores = rng.random(ids.shape, dtype=np.float32)
    scales = rng.uniform(0.25, 1.75, len(owners)).astype('f4')
    def finalize(buf):
        result = np.zeros_like(x)
        for rank in range(len(budget)):
            local = np.zeros_like(x)
            for k in range(ids.shape[1]):
                keep = (ids[:, k] >= 0) & (np.array(owners)[ids[:, k]] == rank)
                slots = np.flatnonzero(keep)*ids.shape[1] + k
                z = buf[slots] * scales[flat[slots], None]
                z = z / np.sqrt(np.mean(z*z, axis=1, keepdims=True) + np.float32(1e-5))
                local[keep] += z * scores[keep, k, None]
            result += round_bf16(local)
        return round_bf16(result)
    assert np.array_equal(finalize(y).view('u4'), finalize(received).view('u4'))
    assert all(used <= cap for used, cap in zip(plan.extra_bytes_by_rank, budget))
    return len(valid), sum(m.rows for m in plan.moves), bool(plan.moves)


def main():
    observed = archives()
    rng = np.random.default_rng(186)
    scenarios = {'observed_rank_even_peers': completion(0), 'observed_rank_hidden_hot_peer': completion(38000)}
    for kind in ('uniform', 'four_local', 'single_hot'):
        ids = native_slots(rng, T, kind)
        scenarios[kind] = np.bincount(ids.ravel(), minlength=EXPERTS).tolist()
    scenarios['same_four_striped'] = np.array(scenarios['four_local']).reshape(32, 8).T.ravel().tolist()
    scenarios['all_disjoint_equal'] = [512] * EXPERTS
    tables = {k: report_plan(v, OWNERS, BUDGET, BASE) for k, v in scenarios.items()}
    assert not tables['all_disjoint_equal']['moves']
    assert not tables['uniform']['moves'] and not tables['same_four_striped']['moves']
    assert tables['single_hot']['moves'] and tables['four_local']['moves']
    assert sum(c for c in scenarios['observed_rank_even_peers'][:32]) == 30697
    sensitivity = {}
    for bw in (100, 225, 450):
        for ns in (102, 165, 359, 443):
            costs = replace(BASE, link_bytes_per_ns=bw, owner_ns_per_issued_row=ns, helper_ns_per_issued_row=ns)
            data = report_plan(scenarios['observed_rank_even_peers'], OWNERS, BUDGET, costs)
            marginal = ns - (4*H + 16)/bw
            threshold = math.ceil((BANK/bw + costs.launch_ns_per_move) / marginal) if marginal > 0 else None
            sensitivity[f'{bw}GBps_{ns}ns'] = dict(rows=data['moved_rows'], ratio=data['ratio'],
                                                   single_bank_net_saving_row_threshold=threshold)
    checks = Counter()
    for _ in range(60):
        ranks = int(rng.choice([2, 3, 4, 6, 8]))
        experts = ranks * 32
        n = int(rng.choice([0, 1, 31, 176, 881, 2049, 4096]))
        ids = native_slots(rng, n, 'duplicate_slots' if rng.integers(2) else 'single_hot') % experts
        ids[::7, 3] = -1
        owners = (np.arange(experts) % ranks).tolist()
        shuffled = rng.permutation(experts)
        owners = [owners[i] for i in shuffled]
        # Small synthetic tiles and transport cost allow forced paths at every
        # legal width. The modeled target-sized probes use true bytes separately.
        costs = Costs(int(rng.choice([1, 7, 32, 176])), 200, 220, 200, 30, 10, 1, 1024, 4, 4, 16)
        budgets = [int(rng.choice([0, 1<<20])) for _ in range(ranks)]
        entries, moved, active = check_slot_return(rng, ids, owners, costs, budgets)
        checks.update(cases=1, valid_slots=entries, moved_slots=moved, positive=int(active))
    assert checks['positive'] > 3 and checks['moved_slots'] > 1000
    assert not plan_splits(scenarios['four_local'], OWNERS, [0]*8, BASE).moves
    for arguments in (([1], [8], BUDGET), ([-1], [0], BUDGET), ([1], [], BUDGET), ([0], [0], [])):
        try:
            plan_splits(*arguments, BASE)
        except ValueError:
            continue
        raise AssertionError(arguments)
    nonfinite = 0
    for field in ('link_bytes_per_ns','global_planning_ns','owner_ns_per_issued_row'):
        for value in (float('inf'), float('nan')):
            try:
                replace(BASE, **{field:value})
            except ValueError:
                nonfinite += 1
    assert nonfinite == 6
    result = dict(archived_physical=observed, cost_assumptions=asdict(BASE), budgets_bytes=BUDGET,
                  scenarios=tables, synthetic_completion_sensitivity=sensitivity, slot_return_cpu=checks,
                  floating_boundaries={'native_BF16_rows_returned_to_original_owner': True,
                                       'hardware_kernel_reassociation_tested': False})
    if '--time' in sys.argv:
        host = {}
        for name in ('all_disjoint_equal','observed_rank_even_peers','single_hot','four_local'):
            samples = []
            for _ in range(9):
                start = time.perf_counter_ns()
                plan_splits(scenarios[name], OWNERS, BUDGET, BASE)
                samples.append((time.perf_counter_ns()-start)/1000)
            host[name] = round(statistics.median(samples),1)
        result['host_python_planning_median_us'] = host
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
