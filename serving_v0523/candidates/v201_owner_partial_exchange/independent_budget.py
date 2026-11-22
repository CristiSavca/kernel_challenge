"""Bound owner-only BF16 exchange traffic and exhibit distinct floating-point reduction results."""
import json
import math
import random
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORLD = 8
CHANNELS = (4096, 8192)
LAYER_COUNT = 62


def f32(x):
    return struct.unpack("!f", struct.pack("!f", x))[0]


def bf16(x):
    word = struct.unpack("!I", struct.pack("!f", x))[0]
    word = (word + 32767 + ((word >> 16) & 1)) & 0xffff0000
    return struct.unpack("!f", struct.pack("!I", word))[0]


def ordered(xs, round_to):
    acc = xs[0]
    for x in xs[1:]:
        acc = round_to(acc + x)
    return bf16(acc)


def balanced(xs, round_to):
    while len(xs) > 1:
        xs = [round_to(xs[i] + xs[i+1]) for i in range(0,len(xs),2)]
    return bf16(xs[0])


def numerical():
    rng = random.Random(883)
    counts = {}
    for spread in (0, 8, 13, 14, 18, 28, 60):
        n = 10000
        tree_changes = 0
        bf16_changes = 0
        for _ in range(n):
            xs = [rng.choice((-1,1)) * math.ldexp(rng.randint(128,255), rng.randint(0,spread)) for _ in range(8)]
            rank = ordered(xs,f32)
            tree_changes += rank != balanced(xs,f32)
            bf16_changes += rank != balanced(xs,bf16)
            if spread <= 13:
                assert rank == bf16(math.fsum(xs))
        counts[spread] = dict(count=n, compared_to_fp32_tree=tree_changes, compared_to_bf16_tree=bf16_changes)
    inputs=[2**30,1.,float(-(2**30)),1.,0.,0.,0.,0.]
    counter = dict(inputs=inputs, rank_fp32=ordered(inputs,f32), tree_fp32=balanced(inputs,f32), exact=bf16(math.fsum(inputs)))
    assert (counter["rank_fp32"],counter["tree_fp32"],counter["exact"])==(1.,0.,2.)
    return dict(exponent_spread=counts, cancellation=counter)


def path_cost():
    data=ROOT/"experiments/modal_node/20261113T204343Z-bench"
    trace=json.loads(next(data.glob("*v41__load.json")).read_bytes())
    groups=[]
    for line in next(data.glob("*v41__server.log")).read_text().splitlines():
        if "Cache flushed successfully!" in line:
            groups.append([])
        elif groups and "TP0 EP0" in line and (match:=re.search(r"Prefill batch, #new-seq: \d+, #new-token: (\d+)", line)):
            groups[-1].append(int(match[1]))
    labels=[(r["concurrency"],wall) for r in trace["rungs"] for wall in r["wall_s"]]
    assert len(groups)==len(labels)==7
    examples=[]
    for q, h in [(16384,c) for c in CHANNELS]:
        total=q*h*2
        owner=total*(WORLD-1)//WORLD
        examples.append(dict(rows=q,width=h,full_bytes=total,ring_all_reduce_send_bytes=2*owner,
                             owner_exchange_send_bytes=owner,owner_exchange_receive_bytes=owner))
    trace_rows=[]
    for rows,(load,wall) in zip(groups,labels):
        # Count just unambiguous full model chunks. Ignore any partial/bucket geometry.
        count=rows.count(16384)
        saved=sum(e["owner_exchange_send_bytes"] for e in examples)*count*LAYER_COUNT
        seconds=saved/450e9
        trace_rows.append(dict(users=load,graph_full_chunks=count,wall_s=wall,net_ms_per_2_reductions_for_5pct_raw=wall*(1-1/1.05)*1000/(count*LAYER_COUNT),
                               ring_savings_in_peak_link_seconds=seconds,equivalent_fraction=seconds/wall))
    return dict(full_examples=examples, traces=trace_rows, transport="Ring allreduce only; simultaneous send/receive. NVLS incumbent and sustained bandwidth unknown.")


if __name__=="__main__":
    print(json.dumps(dict(numerical=numerical(),traffic=path_cost()),indent=2))
