"""Replay page-one DFlash reservation and retirement over archived client lengths."""
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
SGLANG = ROOT / "serving_v0523/sglang"
ALLOCATION = "mem_cache/allocation.py"
SIZING = "mem_cache/allocation_sizing.py"
TOKEN = "mem_cache/allocator/token.py"
RESERVE = "speculative/dflash_info_v2.py"
TILE, WIDTH, PARTITIONS, CAPACITY = 128, 8, 132, 3420199
torch.set_num_threads(1)
source_hashes = {}


def source(rel):
    path = "python/sglang/srt/" + rel
    raw = subprocess.check_output(["git", "-C", SGLANG, "show", PIN + ":" + path])
    source_hashes[rel] = sha256(raw).hexdigest()
    return ast.parse(raw)


def native(rel, name):
    node = next(n for n in ast.walk(source(rel)) if isinstance(n, ast.FunctionDef) and n.name == name)
    compiled = compile("from __future__ import annotations\n" + ast.unparse(node), rel, "exec")
    exec(compiled, globals())


for method in ["clear", "alloc", "free", "available_size"]:
    native(TOKEN, method)
native(ALLOCATION, "alloc_for_spec_decode")
native(SIZING, "page_aligned_decode_alloc_lens")
prep = next(n for n in ast.walk(source(RESERVE)) if isinstance(n, ast.FunctionDef) and n.name == "prepare_for_decode")
assert "reserve = 2 * block_size" in ast.unparse(prep)


class Pool:
    clear = clear
    alloc = alloc
    free = free
    available_size = available_size
    page_size = 1

    def __init__(self, run):
        self.size, self.device, self.need_sort = CAPACITY, "cpu", False
        self.clear()
        if run:
            ids = self.alloc(self.size)
            end = self.size // run * run
            shuffle = torch.randperm(end // run, generator=torch.Generator().manual_seed(511))
            self.free(torch.cat([ids[:end].view(-1, run)[shuffle].flatten(), ids[end:]]))


def alloc_token_slots(cache, count):
    slots = cache.token_to_kv_pool_allocator.alloc(count)
    assert slots is not None and len(slots) == count
    return slots


def assign_req_to_token_pool_func(indices, table, before, after, slots, count):
    offset = 0
    for index, lo, hi in zip(indices.tolist(), before.tolist(), after.tolist(), strict=True):
        table[index, lo:hi] = slots[offset:offset + hi-lo]
        offset += hi-lo
    assert offset == len(slots) and count == len(indices)


def certify(rows):
    measures = Counter()
    counts = [(len(t) + TILE-1) // TILE for t in rows]
    part = (sum(counts) + PARTITIONS-1) // PARTITIONS
    base = 0
    for t, count in zip(rows, counts, strict=True):
        x = t.numpy()
        assert x.min() > 0 and x.max() <= CAPACITY
        complete = len(x) // TILE
        tiles = x[:complete * TILE].reshape(complete, TILE)
        valid = np.zeros(count, dtype=bool)
        valid[:complete] = (np.diff(tiles, axis=1) == 1).all(axis=1)
        measures["tiles"] += count
        for j in range(base // part, (base + count-1) // part + 1):
            lo, hi = max(0, j * part-base), min(count, (j+1)*part-base)
            segment = valid[lo:hi]
            measures["segments"] += 1
            if segment.all():
                measures["whole"] += len(segment)
            edges = np.flatnonzero(np.diff(np.r_[False, segment, False]))
            spans = edges[1::2] - edges[::2]
            long_runs = spans >= 4
            measures["run4"] += int(spans[long_runs].sum())
            if len(segment) <= 32:
                measures["selected"] += len(segment) if segment.all() else 0
                measures["accumulation_runs"] += 1
            else:
                direct_rows = int(spans[long_runs].sum())
                starts = edges[::2][long_runs]
                ends = edges[1::2][long_runs]
                calls = 1 if not len(starts) else 2*len(starts)-1+int(starts[0]>0)+int(ends[-1]<len(segment))
                measures["selected"] += direct_rows
                measures["accumulation_runs"] += calls
            for n in (4, 8):
                end = len(segment) // n * n
                if end:
                    measures["group" + str(n)] += n * int(segment[:end].reshape(-1,n).all(axis=1).sum())
                if end < len(segment) and segment[end:].all():
                    measures["group" + str(n)] += len(segment) - end
        base += count
    return measures


def run(records, fragment):
    pool = Pool(fragment)
    cache = NS(token_to_kv_pool_allocator=pool)
    prev = {}
    users = max(r["user"] for r in records) + 1
    rows = torch.empty((users, max(r["prompt"]+r["completion"] for r in records)+2*WIDTH+1),dtype=torch.int64)
    shared = pool.alloc(max(r["cached"] for r in records if not r["turn"]))
    assert shared is not None
    by_phase = defaultdict(Counter)
    stats = Counter()
    for turn in sorted({r["turn"] for r in records}):
        jobs = sorted((r for r in records if r["turn"] == turn), key=lambda r:r["t0"])
        assert len(jobs) == users
        for job in jobs:
            u, n = job["user"], job["cached"]
            prior = shared if turn == 0 else prev[u]
            assert n <= len(prior)
            rows[u,:n] = prior[:n]
        progress = {r["user"]:r["cached"] for r in jobs}
        while any(progress[r["user"]] < r["prompt"] for r in jobs):
            for r in jobs:
                u = r["user"]
                lo, hi = progress[u], min(r["prompt"], progress[u]+16384)
                if hi > lo:
                    rows[u,lo:hi] = alloc_token_slots(cache,hi-lo)
                    progress[u] = hi
        req = {r["user"]: NS(kv=NS(kv_committed_len=r["prompt"], kv_allocated_len=r["prompt"])) for r in jobs}
        seen = progress.copy()
        stop = {r["user"]:r["prompt"]+r["completion"]-1 for r in jobs}
        wave = 0
        while any(progress[u] < stop[u] for u in progress):
            active = [r["user"] for r in jobs if progress[r["user"]] < stop[r["user"]]]
            live = [req[u] for u in active]
            # Simulate overlap's one-iteration CPU lag. Each request's latest GPU length is in progress.
            for u,r in zip(active,live,strict=True):
                r.kv.kv_committed_len = seen[u]
            before, after, total = page_aligned_decode_alloc_lens(live, reserve=2*WIDTH, page_size=1)
            old, new = torch.tensor(before), torch.tensor(after)
            alloc_for_spec_decode(cache, NS(req_to_token=rows), reqs=live,req_pool_indices=torch.tensor(active),
                cur_kv_lens=old,cur_kv_lens_cpu=old,nxt_kv_lens=new,nxt_kv_lens_cpu=new,num_needed_tokens=total)
            for u in active:
                assert progress[u]+WIDTH <= req[u].kv.kv_allocated_len
            if wave in (0, 4, 16, 32, 48):
                phase = "first" if wave == 0 else "later"
                by_phase[phase].update(certify([rows[u,:progress[u]+WIDTH] for u in active]))
                by_phase[phase]["sampled_batches"] += 1
                by_phase[phase]["live_requests_sampled"] += len(active)
            for u in active:
                seen[u] = progress[u]
                # A non-constant speculative commit sequence with mean four and support 1..7.
                accepted = 1 + (3*wave + 5*u + turn) % 7
                progress[u] = min(stop[u], progress[u]+accepted)
            stats["reservation_rounds"] += 1
            wave += 1
        retired = []
        for u in progress:
            prev[u] = rows[u,:stop[u]].clone()
            retired.append(rows[u,stop[u]:req[u].kv.kv_allocated_len])
        tails=torch.cat(retired)
        stats["retired_surplus_slots"] += len(tails)
        pool.free(tails)
        assert pool.available_size() >= 0
    return by_phase, stats


client = next((ROOT/"experiments/modal_node/20261113T204343Z-bench").glob("*v41__load.json"))
history=defaultdict(list)
for r in json.loads(client.read_bytes())["requests"]:
    history[r["load"],r["rep"]].append(r)
results = {}
for fragment in (0,4096,256,32):
    name = "fresh" if fragment == 0 else "free_runs_" + str(fragment)
    data = defaultdict(lambda: defaultdict(Counter))
    for (load,rep),records in sorted(history.items()):
        phases, stats = run(records,fragment)
        for phase, measures in phases.items():
            data[load][phase].update(measures)
        data[load]["reservation"].update(stats)
    for load, phases in data.items():
        for phase in ("first","later"):
            m=phases[phase]
            phases[phase] = dict(m,coverage={k:m[k]/m["tiles"] for k in ("whole","group4","group8","run4","selected")})
    results[name] = data
print(json.dumps(dict(base=PIN,source_sha256=source_hashes, client_sha256=sha256(client.read_bytes()).hexdigest(), results=results,
    physical_h200=False, model="Observed request lengths and turn order with hypothetical same-turn cohorts. Native allocator, reserve sizing and spec allocation with a CPU scatter, staged prompt allocations, one iteration of CPU lag and varying synthetic 1..7 commits. Excess reservations returned at the end of each turn. Shared and reusable prefixes are modeled. Counts sample waves 0/4/16/32/48 from each turn; retired requests leave the active batch. Coverage describes candidate tile eligibility, not speed or live H200 allocation."),indent=2))
