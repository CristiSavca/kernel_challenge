"""Estimate the prefix-sensitive part of cold prefill from two original eight-H200 runs."""
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import re
from statistics import mean, median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHUNK = 16384
RANK_Q_HEADS, D, FULL_LAYERS = 16, 64, 16
REPORT = re.compile(
    r"\[([^]]+) TP0 EP0\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), "
    r"#cached-token: (\d+),.*#running-req: (\d+), #queue-req: (\d+), "
    r"#pending-token: (\d+), cuda graph: (True|False), input throughput \(token/s\): ([\d.]+)"
)
ARCHIVES = [
    ("V41", "20261113T204343Z-bench", "results__20261113T204343Z-bench__v41__"),
    ("V44", "20261113T213641Z-bench", "results__partial__v44__"),
]


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def get_groups(lines, prompts):
    records = []
    for text in lines:
        match = REPORT.search(text)
        if match:
            stamp, seqs, tokens, cached, running, queue, pending, graph, rate = match.groups()
            records.append(dict(time=stamp, seqs=int(seqs), q=int(tokens), cached=int(cached),
                running=int(running), queue=int(queue), pending=int(pending), graph=graph=="True",
                rate=float(rate), replay="#replay-token:" in text))
    cursor = 0
    groups = []
    for p in prompts:
        start = next(ix for ix in range(cursor, len(records)) if records[ix]["q"]==CHUNK
                     and records[ix]["q"]+records[ix]["pending"]==p)
        chain = []
        remaining = p
        for row in records[start:]:
            assert row["seqs"]==1 and row["running"]==row["queue"]==row["cached"]==0 and not row["replay"]
            assert row["q"]==min(remaining,CHUNK) and row["pending"]==remaining-row["q"]
            chain.append({**row, "prefix":p-remaining, "gap_s":row["q"]/row["rate"]})
            remaining-=row["q"]
            if remaining==0:
                break
        assert remaining==0
        groups.append(chain)
        cursor=start+len(chain)
    return groups


def slope(groups):
    # Fixed effects for each request remove its constant timing offset.
    numer = denom = 0.
    for rows in groups:
        xs, ys = zip(*rows)
        xbar, ybar = mean(xs), mean(ys)
        numer += sum((x-xbar)*(y-ybar) for x,y in rows)
        denom += sum((x-xbar)**2 for x in xs)
    return numer/denom


def work(chunk_rows):
    return sum(q*(2*p+q+1)/2 for p,q in chunk_rows)


def run(label, dirname, stem, v64):
    source=ROOT/"experiments/modal_node"/dirname
    log,load=source/(stem+"server.log"),source/(stem+"load.json")
    raw=json.loads(load.read_bytes())
    cold=[r for r in raw["requests"] if r["load"]==1 and r["turn"]==0]
    cold.sort(key=lambda r:r["rep"])
    assert len(cold)==3 and all(not r["cached"] for r in cold)
    rung=next(r for r in raw["rungs"] if r["concurrency"]==1)
    grid=json.loads((HERE.parent/"v232_mixed_prompt_overflow/geometry.json").read_bytes())["captured_prompt_grid"]
    groups=get_groups(log.read_text().splitlines(),[r["prompt"] for r in cold])
    interiors=[]
    for chain in groups:
        assert all(x["graph"] for x in chain[:-1])
        # Incoming and outgoing records must ALL be full, so a one-batch overlap
        # cannot substitute the tiny terminal residue. Entry gap includes frontend.
        interiors.append([(row["prefix"]/CHUNK,row["gap_s"]) for i,row in enumerate(chain)
            if i>0 and i+1<len(chain) and all(ch["q"]==CHUNK for ch in chain[i-1:i+2])])
    fit=slope(interiors)
    # Reassign each fixed-size gap to its previous or next chunk. This changes
    # the intercept only. Both fits must recover the same history slope.
    for shift in (-1,1):
        assert math.isclose(fit,slope([[(x+shift,y) for x,y in rows] for rows in interiors]),rel_tol=1e-10)
    leave=[slope([x for i,x in enumerate(interiors) if i!=skip]) for skip in range(3)]
    pair_slopes=[(b[1]-a[1])/(b[0]-a[0]) for rows in interiors for a,b in itertools.combinations(rows,2)]
    reps=[]
    for rep,(chain,request,interior) in enumerate(zip(groups,cold,interiors)):
        p=request["prompt"]
        pairs=work([(row["prefix"],row["q"]) for row in chain])
        assert pairs==p*(p+1)//2
        extent=pairs/CHUNK**2
        estimated=extent*fit
        padded=sum(next(k for k in grid if k>=row["q"])-row["q"] for row in chain)
        wall=rung["wall_s"][rep]
        reps.append(dict(rep=rep,prompt=p,client_ttft_s=request["ttft"],wall_s=wall,
            all_chunk_rows=[x["q"] for x in chain],actual_graph_flags=[x["graph"] for x in chain],
            interior_measurements=[dict(prior_chunks=int(x),interval_s=y) for x,y in interior],
            graph_padding_rows_at_archived_keys=padded,prompt_pairs=pairs,modeled_context_sensitive_s=estimated,
            modeled_ttft_share=estimated/request["ttft"],
            required_remaining_ttft_factor_even_if_context_term_free={
                str(target):target*(1-estimated/request["ttft"]) for target in (2,3)},
            hypothetical_raw_one_user_rung_multiplier_2x_context_term=wall/(wall-.5*estimated),
            hypothetical_raw_one_user_rung_multiplier_instant_context_term=wall/(wall-estimated)))
    units=v64["shape"]["first_turn"]*(v64["shape"]["first_turn"]+1)/(2*CHUNK**2)
    ops=4*RANK_Q_HEADS*D*FULL_LAYERS*CHUNK**2
    return dict(run=label,load_sha256=digest(load),log_sha256=digest(log),wall_s=rung["wall_s"],
        useful_intervals=sum(map(len,interiors)),median_interior_gap_s=median(y for a in interiors for _,y in a),
        slope_s_per_added_16k_prefix_per_16k_query=fit,individual_request_slopes=[slope([rows]) for rows in interiors],
        median_within_request_pair_slope=median(pair_slopes),leave_one_request_out_slopes=leave,
        incremental_full_attention_flops_per_rank=ops,equivalent_tflop_s_if_all_slope_is_full_attention=ops/fit/1e12,
        v64_shape_modeled_context_sensitive_s=fit*units,v64_shape_loo_sensitivity_s=[min(leave)*units,max(leave)*units],
        reps=reps)


def main():
    judge=ROOT/"serving_v0523/runs/abd217baf0ed49fd.json"
    v64=json.loads(judge.read_bytes())["result"]
    runner=ROOT/"serving_v0523/sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
    src=runner.read_text()
    assert "(prefill_stats.log_input_tokens + prefill_stats.log_replay_tokens)" in src and "/ gap_latency" in src
    public=next(x for x in v64["rungs"] if x["concurrency"]==1)
    sweep=next(x["build"] for x in v64["launch_settings"]["derived"] if x["setting"].endswith(".memory_usage.weight"))
    # The reported 'gb' figures are used at both byte conventions to avoid assuming the label is precise.
    diagnostic={}
    for chunk in (16384,32768,65536):
        n=math.ceil(v64["shape"]["first_turn"]/chunk)
        saved=math.ceil(v64["shape"]["first_turn"]/CHUNK)-n
        diagnostic[chunk]=dict(outer_forwards=n,fewer_weight_sweeps=saved,
            full_resident_bytes_if_read_once_per_eliminated_sweep=[saved*sweep*unit for unit in (10**9,2**30)],
            equivalent_seconds_at_4p8TBps=[saved*sweep*unit/4.8e12 for unit in (10**9,2**30)],
            one_extra_BF16_8192_tensor_bytes_per_rank_if_unsharded=(chunk-CHUNK)*8192*2)
    print(json.dumps(dict(cold_turns=[run(*x,v64) for x in ARCHIVES],official_V64=dict(
        sha256=digest(judge),score=v64["frontier"]["score"],first_prompt=v64["shape"]["first_turn"],
        one_user=public,memory=v64["gpu_memory"],resident_weight_gb=sweep),larger_chunk_counterfactual=diagnostic,
        time_source_sha256=digest(runner),physical_V244_timing=False,
        interpretation="Scheduler wall-gap observations, attributed to prefix length by a within-request fit. Context term and larger-chunk costs are counterfactual, not a timestamp for individual attention kernels."),indent=2))


if __name__=="__main__":
    main()
