"""Run the pinned page-one full allocator and V192 segment rule on real client length sequences."""
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REPO=ROOT/"serving_v0523/sglang"
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
FULL_CAPACITY=3420199
K_TILE, PARTS, WIDTH, COMMIT=128,132,8,4
torch.set_num_threads(1)
src="python/sglang/srt/mem_cache/allocator/token.py"
body=subprocess.check_output(["git","-C",str(REPO),"show",PIN+":"+src])
node=next(n for n in ast.parse(body).body if getattr(n,"name",None)=="TokenToKVPoolAllocator")
methods=[n for n in node.body if getattr(n,"name",None) in ("alloc","clear","available_size","free")]
ns=dict(torch=torch)
exec("from __future__ import annotations\n"+ast.unparse(ast.Module(methods,[])),ns)


class Bank:
    alloc=ns["alloc"]
    free=ns["free"]
    clear=ns["clear"]
    available_size=ns["available_size"]

    def __init__(self, run):
        self.size,self.device,self.need_sort=FULL_CAPACITY,"cpu",False
        self.clear()
        if run:
            ids=self.alloc(self.size)
            complete=(self.size//run)*run
            order=torch.randperm(complete//run,generator=torch.Generator().manual_seed(511))
            # A hypothetical free list made by returning complete contiguous fragments in shuffled order.
            self.free(torch.cat((ids[:complete].view(-1,run)[order].flatten(),ids[complete:])))


def coverage(inputs):
    measures=Counter()
    counts=[(len(x)+K_TILE-1)//K_TILE for x in inputs]
    per=(sum(counts)+PARTS-1)//PARTS
    base=0
    for t,count in zip(inputs,counts):
        x=t.numpy()
        assert x.min()>0 and x.max()<=FULL_CAPACITY
        full=len(x)//K_TILE
        blocks=x[:full*K_TILE].reshape(full,K_TILE)
        certified=np.zeros(count,dtype=np.bool_)
        certified[:full]=np.all(blocks[:,1:]-blocks[:,:-1]==1,axis=1)
        measures["tiles"]+=count
        measures["certified_tiles"]+=int(certified.sum())
        for left in range(base//per,(base+count-1)//per+1):
            lo=max(base,left*per)-base
            hi=min(base+count,(left+1)*per)-base
            measures["partition_segments"]+=1
            if certified[lo:hi].all():
                measures["direct_segments"]+=1
                measures["tiles_in_direct_segments"]+=hi-lo
        base+=count
    return measures


def run(records,fragment):
    a=Bank(fragment)
    user_ids=sorted({x["user"] for x in records})
    rows={u:torch.empty(0,dtype=torch.int64) for u in user_ids}
    shared_count=max(x["cached"] for x in records if x["turn"]==0)
    shared=a.alloc(shared_count)
    totals=Counter()
    by_turn=defaultdict(Counter)
    for turn in sorted({x["turn"] for x in records}):
        jobs=sorted((r for r in records if r["turn"]==turn),key=lambda r:r["t0"])
        assert len(jobs)==len(rows)
        additions={}
        for job in jobs:
            u,cached=job["user"],job["cached"]
            prior=rows[u] if turn else shared
            assert 0<=cached<=len(prior)
            additions[u]=[prior[:cached]]
        need={r["user"]:r["prompt"]-r["cached"] for r in jobs}
        while any(need.values()):
            for job in jobs:
                u=job["user"]
                if need[u]:
                    n=min(need[u],16384)
                    additions[u].append(a.alloc(n))
                    need[u]-=n
        for job in jobs:
            additions[job["user"]].append(a.alloc(WIDTH))
        first=[torch.cat(additions[job["user"]]) for job in jobs]
        measure=coverage(first)
        totals.update(measure)
        by_turn["first_turn" if turn==0 else "later_turns"].update(measure)
        # Recompute policy on fresh physical mappings each turn; never time attention or infer judge acceptance.
        need={r["user"]:max(0,r["completion"]-WIDTH) for r in jobs}
        while any(need.values()):
            for u in user_ids:
                if need[u]:
                    n=min(COMMIT,need[u])
                    additions[u].append(a.alloc(n))
                    need[u]-=n
        for job in jobs:
            u=job["user"]
            rows[u]=torch.cat(additions[u])[:job["prompt"]+job["completion"]]
        assert a.available_size()>=0
    totals["request_first_blocks"]=len(records)
    return totals,by_turn


client=next((ROOT/"experiments/modal_node/20261113T204343Z-bench").glob("*v41__load.json"))
history=defaultdict(list)
for q in json.loads(client.read_bytes())["requests"]:
    history[q["load"],q["rep"]].append(q)
report={}
for size in (0,4096,256,32):
    item=defaultdict(Counter); separate=defaultdict(lambda:defaultdict(Counter))
    for (concurrency,rep),jobs in sorted(history.items()):
        measure,by_turn=run(jobs,size)
        item[concurrency].update(measure)
        for key,t in by_turn.items():separate[concurrency][key].update(t)
    for concurrency,counts in item.items():
        counts["tma_tile_share"]=counts["tiles_in_direct_segments"]/counts["tiles"]
        counts["by_turn"]=separate[concurrency]
    report["fresh" if size==0 else "free_runs_"+str(size)]=item

print(json.dumps(dict(allocator=src,source_sha256=sha256(body).hexdigest(),pin=PIN,capacity=FULL_CAPACITY,
    client=str(client.relative_to(ROOT)),client_sha256=sha256(client.read_bytes()).hexdigest(),
    ordinary_page_size=1,tile_width=K_TILE,partitions=PARTS,representative_commit=COMMIT,
    results=report,physical_h200=False,
    scope="Actual V41 prompt/cached/output lengths and native page-one alloc/free order. Same-turn cohorts are co-scheduled hypothetically; one representative first verify per request. Additional appends model four new tokens per round, retaining prior evictable rows. Returned-fragment orders are sensitivities. The judge has no physical address trace; these are conditional eligibility figures, not time or live coverage."),indent=2))
