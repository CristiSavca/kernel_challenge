"""Replay the cache chronologically, starting empty at every archived repetition."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
prior = runpy.run_path(str(next(HERE.parent.glob("v202_*/phase_budget.py"))))
archive = prior["ARCHIVE"]
client, = archive.glob("*v41__load.json")
server, = archive.glob("*v41__server.log")
order = [r["concurrency"] for r in json.loads(client.read_bytes())["rungs"] for _ in range(r["reps"])]
setup = runpy.run_path(str(prior["GEOMETRY"].with_suffix(".py")))
choose, _ = setup["pinned_selector"]()
grid = json.loads(prior["GEOMETRY"].with_suffix(".json").read_bytes())
prompt_grid, verify_grid = grid["target_prefill_graph_rows"], grid["target_verify_graph_requests"]
before = json.loads(next(HERE.parent.glob("v212_*/geometry.json")).read_bytes())["by_load"]
TP, HIDDEN, CHANNELS = 8, 8192, 1280
BANK = 2 * HIDDEN * CHANNELS


def wire(tokens, capacity, cached, allow_padding):
    rows = (capacity + TP - 1) // TP
    ordinary = 2 * (TP-1) * rows * HIDDEN
    def sent(width, rows_per_rank):
        peers = TP // width-1
        return 2*((width-1)*rows_per_rank*HIDDEN + peers*width*rows_per_rank*CHANNELS) + (peers-int(cached))*BANK
    width = min((2,4),key=lambda n: sent(n,rows))
    use = (tokens >= 128*TP and (allow_padding or rows == (tokens+TP-1)//TP)
           and 8*sent(width,rows) <= 7*ordinary)
    # New P2P actually moves only the valid per-owner/per-cohort prefixes.
    # Summing over the mesh gives this exact mean payload per rank.
    actual = sent(width,tokens/TP if allow_padding else rows) if use else ordinary
    return use, actual, width


def simulate(padded):
    totals = defaultdict(Counter)
    slot = -1
    for line in server.read_text().splitlines():
        if "Cache flushed successfully!" in line:
            slot += 1
            cached, previous = False, None
        m = prior["PATTERN"].search(line)
        if not m or slot < 0:
            continue
        p, q, t, queue, pending = map(int, m.groups())
        stats = totals[str(order[slot])]
        body, role = q, "prompt"
        if t:
            combined = q+t*8
            if combined<=16384 and choose(combined,prompt_grid)<=choose(q,prompt_grid)+8*choose(t,verify_grid):
                body,role = combined,"mixed"
            elif (q==16384 and p==1 and queue==0 and pending>0 and previous is not None and
                  previous[0]==1 and previous[3]==0 and previous[4]==q+pending):
                last=(q+pending-1)%16384+1
                if (last+t*8<=16384 and choose(last+t*8,prompt_grid)==choose(last,prompt_grid)
                    and choose(q-t*8,prompt_grid)+8*choose(t,verify_grid)>=16384):
                    body,role = 16384,"mixed"
        previous = p,q,t,queue,pending
        stats[role+"_opportunities"] += 1
        if body < 128*TP:
            continue
        capacity = choose(body,prompt_grid) if padded else body
        cold, old_bytes, _ = wire(body,capacity,False,False)
        fast, new_bytes, width = wire(body,capacity,cached,True)
        stats[role+"_v212_calls"] += int(cold)
        stats[role+"_v214_calls"] += int(fast)
        stats[role+"_v212_sent_bytes"] += old_bytes
        stats[role+"_v214_sent_bytes"] += new_bytes
        if fast:
            stats[role+"_width_"+str(width)] += 1
            stats["cold_primes"] += int(not cached)
            cached = True
    assert slot+1 == len(order)
    for load,counts in totals.items():
        shape = "padded_capture_shards" if padded else "live_shards"
        assert counts["mixed_opportunities"] == before[load]["mixed_opportunities"]
        assert counts["mixed_v212_calls"] == before[load][shape]["calls"]
    return totals


print(json.dumps(dict(tp=TP, prepared_peer_bytes=BANK, source=str(server.relative_to(prior["ROOT"])),
    cumulative_per_rank_sent_bytes=True, live_shards=simulate(False), padded_capture_shards=simulate(True),
    assumptions="V41 chronology. Cold immutable weights at every repetition, unit-scale fused helper, eligible TP8/NCCL path. "
    "Before the first successful exchange both candidates use V212. Counts include original one-role prompts and "
    "possible mixed targets in chronological order. Payload is ideal per-rank transmission for the modeled endpoint only. "
    "V212 declines oversized shards. V214 retains the physical rank pitch and transfers only the live prefixes. Bytes are mesh totals divided by eight.",
    H200_candidate=False),indent=2))
