"""Count identifiable, budget-neutral continuations in the archived H200 serving trace."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy

ROOT=Path(__file__).resolve().parents[3]
up=runpy.run_path(str(ROOT/"experiments/iterations/20261118-serving-mixed-graph-capacity.py"))
log=next(up["ARCHIVE"].glob("*v41__server.log"))
client=next(up["ARCHIVE"].glob("*v41__load.json"))
content=log.read_text()
order=[r["concurrency"] for r in json.loads(client.read_bytes())["rungs"] for _ in range(r["reps"])]
grid=up["capture_grid"](content,r"Capture target prefill CUDA graph begin\. backend=breakable, num_tokens=")
verify=up["capture_grid"](content,r"Capture target verify CUDA graph end\. bs=")
info=json.loads(next(up["ARCHIVE"].glob("*v41__server_info.json")).read_bytes())["internal_states"][0]
cap=min(info["chunked_prefill_size"],info["max_prefill_tokens"])
width=info["speculative_num_draft_tokens"]
table=defaultdict(Counter)
choose,_=up["pinned_selector"]()
pattern=re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+)")
ordinal=-1
prev=None
for line in content.splitlines():
    if "Cache flushed successfully!" in line:
        ordinal+=1
        prev=None
    if ordinal<0:
        continue
    m=pattern.search(line)
    if not m:
        continue
    seq,rows,running,queue,pending=map(int,m.groups())
    load=order[ordinal]
    out=table[load]
    out["prompt_calls"]+=1
    if running and rows==cap:
        out["full_prompt_with_running"]+=1
        out["full_prompt_running_appearances"]+=running
        if seq==1 and queue==0 and pending>0 and prev is not None and prev[0]==1 and prev[3]==0 and prev[4]==rows+pending:
            out["identifiable_partial_continuations"]+=1
            tail=width*running
            last=(rows+pending-1)%cap+1
            if last+tail<=cap and choose(last+tail,grid)==choose(last,grid) and choose(cap-tail,grid)+width*choose(running,verify)>=cap:
                out["fit_same_current_and_final_buckets"]+=1
                out["running_appearances_fitting"]+=running
                out["shifted_prompt_rows"]+=tail
        elif queue>0 or prev is None or prev[3]>0:
            out["other_full_calls_with_ambiguous_pending_work"]+=1
    prev=(seq,rows,running,queue,pending)
# Run every real first-turn prompt through a sequential, fixed-capacity chunker.
# The hypothetical cohort stays the same size for that request; V41 does not log its counterfactual.
sim=defaultdict(dict)
for concurrency in sorted(set(order)):
    reqs=[r for r in json.loads(client.read_bytes())["requests"] if r["load"]==concurrency and r["turn"]==0]
    for tails in range(1,concurrency):
        cnt=Counter()
        for req in reqs:
            full=req["prompt"]
            rem=full
            native=[]
            for begin in range(0,full,cap):
                native.append(choose(min(cap,full-begin),grid))
            shaped=[]
            while rem:
                chunk=min(rem,cap)
                if rem!=full and rem>cap:
                    shifted=tails*width
                    last=(rem-1)%cap+1
                    if (0<shifted<cap and last+shifted<=cap and choose(last+shifted,grid)==choose(last,grid)
                            and choose(cap-shifted,grid)+width*choose(tails,verify)>=cap):
                        chunk=cap-shifted
                        cnt["merged_calls"]+=1
                        cnt["combined_request_steps"]+=tails
                shaped.append(choose(chunk+(tails*width if chunk<min(rem,cap) else 0),grid))
                rem-=chunk
            assert native==shaped
            cnt["cold_requests"]+=1
            cnt["prompt_forwards"]+=len(native)
        sim[concurrency][tails]=cnt
print(json.dumps(dict(simulated_constant_cohort=sim,source=str(log.relative_to(ROOT)), sha256=sha256(log.read_bytes()).hexdigest(), by_load=table,
    width=width, chunk_rows=cap, capture_grid=grid,
    interpretation="Original V41 scheduler. A live continuation is identifiable when two consecutive prompt calls each have one request, empty waiting queues and exactly matching pending work after subtracting the next full chunk. These are local per-event opportunities, not observations of V194 dispatch. The original mixed admission and GPU availability still apply."),indent=2))
