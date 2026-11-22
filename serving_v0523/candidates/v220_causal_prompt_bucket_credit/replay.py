"""Bound a prompt-local credit policy with V215 gates and V41 recorded request shapes."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent/"v215_decode_reduction_paired_context/patch.diff"
CAP = 16384
WIDTH = 8


def simulate(left, tails, remaining_rounds, initial, policy, prefill, verify, pad):
    balance = initial
    steps = Counter()
    for _ in range(remaining_rounds):
        if left <= CAP:
            break
        moved = WIDTH*tails
        final = (left-1)%CAP+1
        if final + moved > CAP:
            left -= CAP
            continue
        debit = pad(final+moved,prefill)-pad(final,prefill)
        # Positive credit can only be realized after a joined target actually returns.
        gain = pad(CAP-moved,prefill)+WIDTH*pad(tails,verify)-CAP
        assert gain >= 0
        funds = 0 if policy=="incumbent" else balance
        if policy=="prospective":
            funds += gain
        if debit <= funds:
            steps["joins"] += 1
            steps["request_verifications"] += tails
            steps["charged_larger_tail_rows"] += debit
            steps["retired_separate_verify_rows"] += WIDTH*pad(tails,verify)
            steps["earnings"] += gain
            balance += gain-debit
            left -= CAP-moved
        else:
            left -= CAP
        assert balance >= 0
    return dict(steps)


def main():
    patch=PARENT.read_bytes()
    assert sha256(patch).hexdigest()=="bbf69a796f2e2ae2edb2b5d913f0e0dc799037d555c10dfe58d00eb71f7cd7b6"
    for s in ("prefill._pad_to_bucket(last, buckets) != prefill._pad_to_bucket(last + reserved, buckets)",
              "if result is not None:\n+                return result",
              "prompt, decode, prompt_result, tail_result, relay, shared_target=False",
              "return DFlashMixedGraphPlan(prompt_rows, verify_rows, mixed_rows)"):
        assert s.encode() in patch
    cfg=runpy.run_path(str(ROOT/"experiments/iterations/20261118-serving-mixed-graph-capacity.py"))
    log_file=next(cfg["ARCHIVE"].glob("*v41__server.log"))
    load_file=next(cfg["ARCHIVE"].glob("*v41__load.json"))
    text=log_file.read_text()
    load=json.loads(load_file.read_bytes())
    prefill=cfg["capture_grid"](text,r"Capture target prefill CUDA graph begin\. backend=breakable, num_tokens=")
    verify=cfg["capture_grid"](text,r"Capture target verify CUDA graph end\. bs=")
    pad,selector_sha=cfg["pinned_selector"]()
    assert prefill[-1]==CAP and verify[-1]==32
    # First-pass prompt events at exact repeated queue-empty, single-request lengths.
    order=[r["concurrency"] for r in load["rungs"] for _ in range(r["reps"])]
    pattern=re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+)")
    ordinal=-1
    chains=[]
    for line in text.splitlines():
        if "Cache flushed successfully!" in line:
            ordinal+=1
            prev=None
            call=0
            partial_credit=0
        if ordinal<0 or (match:=pattern.search(line)) is None:
            continue
        seq,q,tails,queue,pending=map(int,match.groups())
        call+=1
        identified = (seq==1 and q==CAP and tails>0 and queue==0 and pending>0 and prev and
                      prev[0]==1 and prev[3]==0 and prev[4]==q+pending)
        if identified:
            if chains and chains[-1]["repetition"]==ordinal and chains[-1]["end"]==call-1 and chains[-1]["tails"]==tails:
                chains[-1]["end"]=call
            else:
                chains.append(dict(repetition=ordinal,load=order[ordinal],start=call,end=call,tails=tails,
                                   remaining=q+pending,earlier_possible_partial_credit=partial_credit))
        if tails and q+WIDTH*tails<=CAP:
            credit=pad(q,prefill)+WIDTH*pad(tails,verify)-pad(q+WIDTH*tails,prefill)
            if credit>=0:
                partial_credit+=credit
        prev=(seq,q,tails,queue,pending)
    for c in chains:
        run=lambda initial,mode:simulate(c["remaining"],c["tails"],c["end"]-c["start"]+1,initial,mode,prefill,verify,pad)
        c["incumbent"]=run(0,"incumbent")
        c["prior_completed_only"]=run(0,"prior")
        c["prospective_current_credit"]=run(0,"prospective")
        c["hypothetical_preexisting_512"]=run(512,"prior")
        assert c["incumbent"]==c["prior_completed_only"]
    # One request at a time, one fixed known cohort for every continuation after its first chunk.
    fixed={}
    total=0
    for concurrency in (4,16,32):
        summaries=[]
        for tails in range(1,concurrency):
            aggregate={k:Counter() for k in ("incumbent","prior","prospective")}
            for req in load["requests"]:
                if req["load"]!=concurrency or req["turn"]!=0:
                    continue
                total+=1
                for policy in aggregate:
                    counts=simulate(req["prompt"]-CAP,tails,(req["prompt"]+CAP-1)//CAP,0,policy,prefill,verify,pad)
                    aggregate[policy].update(counts)
            assert aggregate["incumbent"]==aggregate["prior"]
            summaries.append(dict(cohort=tails,incumbent=aggregate["incumbent"]["joins"],causal=aggregate["prior"]["joins"],
                                  prospective=aggregate["prospective"]["joins"]))
        fixed[concurrency]=summaries
    print(json.dumps(dict(parent_sha256=sha256(patch).hexdigest(),archive_log_sha256=sha256(log_file.read_bytes()).hexdigest(),
                          client_sha256=sha256(load_file.read_bytes()).hexdigest(),selector_sha256=selector_sha,
                          identified_chains=chains,request_cohort_pairs=total,fixed_cohort=fixed,
                          counterfactual="Every eligible simulated joint target succeeds. Credits count target-body graph rows, not time."),indent=2))


if __name__=="__main__":
    main()
