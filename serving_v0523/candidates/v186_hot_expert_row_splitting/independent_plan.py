"""Reconstruct the row intervals, compute time, send volume and per-rank allocation independently."""
import hashlib
import json
from pathlib import Path
import random
from planner import Costs, plan_splits

r = random.Random(18655)
positive = moved = 0
for _ in range(2100):
    nr = r.choice([2, 4, 8])
    ne = nr * r.randint(1, 5)
    owners = [i % nr for i in range(ne)]
    counts = [r.randrange(50) for _ in owners]
    if r.randrange(4):
        counts[r.randrange(ne)] += r.randrange(30, 180)
    c = Costs(r.choice([1,4,8,16]), r.uniform(2,7),r.uniform(.5,7),r.uniform(20,1100),
              r.uniform(1,11),r.uniform(1,5),r.uniform(1,13),r.randrange(64,1024),
              r.randrange(2,12),r.randrange(2,12),r.randrange(100))
    caps = [r.randrange(300,6000) for i in range(nr)]
    plan = plan_splits(counts, owners, caps, c)
    positive += bool(plan.moves)
    moved += len(plan.moves)
    senders, receivers = set(), set()
    retained, mem, wire = counts[:], [0]*nr, [0.]*nr
    ceil_rows = lambda n: (n+c.rows_per_tile-1)//c.rows_per_tile*c.rows_per_tile
    compute = [0.]*nr
    for m in plan.moves:
        e, a, b = m.expert, m.owner, m.helper
        assert a == owners[e] and a != b and b not in receivers and m.end == retained[e] and 0 <= m.begin < m.end
        n=m.end-m.begin
        retained[e]=m.begin
        senders.add(a); receivers.add(b)
        mem[a] += n*(4*c.hidden+32)
        mem[b] += n*(4*c.hidden+32+6*c.intermediate)+c.helper_workspace_bytes+c.weight_bytes
        comm=(c.weight_bytes+n*(4*c.hidden+16))/c.link_bytes_per_ns+c.launch_ns_per_move
        wire[a]+=comm; wire[b]+=comm
        compute[b]+=ceil_rows(n)*c.helper_ns_per_issued_row
    assert senders.isdisjoint(receivers) and tuple(mem) == plan.extra_bytes_by_rank
    assert all(m<=available for m,available in zip(mem,caps))
    for e,n in enumerate(retained):
        compute[owners[e]]+=ceil_rows(n)*c.owner_ns_per_issued_row
    estimate=max(compute)+max(wire)+(c.global_planning_ns if plan.moves else 0)
    assert abs(estimate-plan.projected_ns)<1e-8
    assert not plan.moves or estimate+c.minimum_saving_ns<plan.original_ns
print(json.dumps(dict(cases=2100,nonempty=positive,moves=moved,seed=18655,source_sha256=hashlib.sha256(
    Path(__file__).with_name('planner.py').read_bytes()).hexdigest()),indent=2))
