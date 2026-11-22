"""Replay causal client dependencies under native FCFS, shortest-prefill and longest-output policies."""
import ast
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import runpy
import statistics
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SERVING = ROOT/"serving_v0523"
PARENT = HERE.parent/"v241_split_serving_decode"
PIN = json.loads((PARENT/"validation.json").read_bytes())["base"]
wave = runpy.run_path(str(ROOT/"experiments/iterations/20261119-serving-cold-wave.py"))
archive = wave["ARCHIVE"]
server = next(archive.glob("*v41__server.log"))
clients = next(archive.glob("*v41__load.json"))
text = server.read_text()
data = json.loads(clients.read_bytes())
CAP = 16384
WIDTH = 8
grid = wave["grid"](text, "Capture target prefill CUDA graph begin. backend=breakable, num_tokens=")
verify_grid = wave["grid"](text, "Capture target verify CUDA graph end. bs=")
grid += [CAP+WIDTH*b for b in verify_grid if b&(b-1)==0]
assert grid == sorted(set(grid)) and data["gpus"] == 8


def native_source(rel):
    return subprocess.check_output(["git","-C",str(SERVING/"sglang"),"show",PIN+":python/sglang/"+rel],text=True)


def build_policy():
    source=native_source("srt/managers/schedule_policy.py")
    nodes=next(n for n in ast.parse(source).body if getattr(n,"name",None)=="SchedulePolicy")
    names={"_shortest_prefill_work","_sort_by_shortest_prefill","_sort_by_longest_output","shortest_prefill_chunk_limit"}
    nodes.body=[n for n in nodes.body if getattr(n,"name",None) in names]
    assert len(nodes.body)==len(names)
    env=dict(CacheAwarePolicy=SimpleNamespace(SHORTEST_PREFILL_FIRST="shortest-prefill-first"),
             _ceil_div=lambda x,y:(x+y-1)//y)
    exec("from __future__ import annotations\n"+ast.unparse(nodes),env)
    return env["SchedulePolicy"](),hashlib.sha256(source.encode()).hexdigest()


def fits_binding():
    patch=(PARENT/"patch.diff").read_text()
    start=patch.index("+def mixed_prefill_graph_fits(")
    lines=[]
    for line in patch[start:].splitlines():
        if not line.startswith("+"): break
        lines.append(line[1:])
    env={}
    node=ast.parse("\n".join(lines)).body[0]
    exec(ast.unparse(node),env)
    return env["mixed_prefill_graph_fits"]


policy, policy_sha = build_policy()
fits = fits_binding()
standalone_period = {r["users"]:r["estimated_period_ms"]/1000 for r in json.loads(
    (HERE.parent/"v140_decode_overlap_feasibility/check.json").read_bytes())["model"]["rungs"]}


def period(batch):
    if batch in standalone_period: return standalone_period[batch]
    hi=next(k for k in (4,16,32) if k>batch)
    lo={4:1,16:4,32:16}[hi]
    return standalone_period[lo]+(standalone_period[hi]-standalone_period[lo])*(batch-lo)/(hi-lo)


@dataclass
class Request:
    record: dict
    serial: int
    done: int = 0
    output: int = 0

    def __post_init__(self):
        r=self.record
        self.user,self.turn=r["user"],r["turn"]
        self.rid=(self.user,self.turn)
        self.origin_input_ids=self.full_untruncated_fill_ids=range(r["prompt"])
        self.output_ids=()
        self.num_matched_prefix_tokens=r["cached"]
        self.time_stats=SimpleNamespace(wait_queue_entry_time=self.serial)
        self.sampling_params=SimpleNamespace(max_new_tokens=r["asked"])
        assert 0<=r["cached"]<r["prompt"] and 0<r["completion"]<=r["asked"]

    @property
    def prefix_len(self): return self.num_matched_prefix_tokens+self.done

    @property
    def remaining(self): return len(self.origin_input_ids)-self.prefix_len


def replay(records,policy_name,commits,arrival,join_every,hold_first):
    policy.policy=policy_name
    by={(r["user"],r["turn"]):r for r in records}
    users=len(by)//13
    assert len(by)==users*13
    order=sorted((r for r in records if not r["turn"]),key=lambda r:r["t0"])
    if arrival:
        # Arrival IDs do not identify the true server queue order. Perturb only the
        # first simultaneous wave, retaining its oldest request and causal later turns.
        front,rest=order[0],order[1:]; random.Random(arrival).shuffle(rest);order=[front]+rest
    actual_wait=[Request(r,i) for i,r in enumerate(order)]
    waiting=[] if hold_first else actual_wait.copy()
    inflight=actual_wait.pop(0) if hold_first else None
    late=actual_wait if hold_first else []
    running=[]
    serial=users
    cost=Counter()
    first_ends={}
    finish={}
    ticks=0
    while waiting or running or inflight or late:
        ticks+=1
        assert ticks<50000
        if ticks>1 and late:
            waiting.extend(late);late=[]
        if policy_name=="shortest-prefill-first":
            policy._sort_by_shortest_prefill(waiting,set())
        elif policy_name=="lof":
            policy._sort_by_longest_output(waiting,False,1)
        budget=CAP;tasks=[]
        if inflight:
            limit=policy.shortest_prefill_chunk_limit(inflight,waiting,budget,1)
            q=min(inflight.remaining,limit or budget)
            tasks.append((inflight,q));budget-=q
            if q==inflight.remaining: inflight=None
        while budget and waiting:
            req=waiting[0]
            if inflight is not None and req.remaining>budget:break
            q=min(req.remaining,budget);budget-=q
            tasks.append((waiting.pop(0),q))
            if q<req.remaining:inflight=req
        q_total=CAP-budget
        tails=len(running)
        adopted=False
        if q_total:
            p=grid[bisect_left(grid,q_total)]
            cost["prompt_batches"]+=1
            cost["prompt_graph_rows"]+=p
            cost["prompt_rows"]+=q_total
            if tails and join_every and (cost["prompt_batches"]%join_every==0):
                v=WIDTH*verify_grid[bisect_left(verify_grid,tails)]
                mix=q_total+WIDTH*tails
                m=grid[bisect_left(grid,mix)]
                adopted=fits(p,v,m,mix,CAP,grid[-1])
                if adopted:
                    cost["joint_groups"]+=1
                    cost["joint_request_steps"]+=tails
                    cost["joint_extra_rows"]+=m-p
                    cost["joined_decode_seconds_proxy"]+=period(tails)
            for req,n in tasks:
                # Count actual attention pairs independent of GPU scheduling.
                cost["full_attended_pairs"]+=n*req.prefix_len+n*(n+1)//2
                req.done+=n
        if running and (adopted or not q_total):
            if not adopted:
                cost["standalone_groups"]+=1
                cost["standalone_request_steps"]+=tails
                cost["standalone_seconds_proxy"]+=period(tails)
            for req in running:
                req.output+=min(commits,req.record["completion"]-req.output)
        # Prompt roles and verification roles are disjoint on the current target.
        for req,n in tasks:
            if not req.remaining:
                if req.turn==0:first_ends[req.user]=int(cost["prompt_batches"])
                req.output=1
                running.append(req)
        next_running=[]
        for req in running:
            if req.output!=req.record["completion"]:
                next_running.append(req);continue
            assert req.rid not in finish
            finish[req.rid]=req.output
            if req.turn<12:
                waiting.append(Request(by[req.user,req.turn+1],serial));serial+=1
        running=next_running
    assert len(finish)==len(records) and sum(finish.values())==sum(r["completion"] for r in records)
    assert cost["prompt_rows"]==sum(r["prompt"]-r["cached"] for r in records)
    cost["output_tokens"]=sum(finish.values())
    cost["all_requested_turns"]=len(finish)
    cost["mean_first_prompt_batch"]=sum(first_ends.values())/users
    return cost


def main():
    ordered=[(r["concurrency"],i,r["wall_s"][i]) for r in data["rungs"] for i in range(r["reps"])]
    assert len(ordered)==7
    waves=text.split("Cache flushed successfully!")[1:]
    aggregates=defaultdict(list); primary=[];baseline=[]
    for (users,rep,wall),section in zip(ordered,waves):
        rs=[r for r in data["requests"] if (r["load"],r["rep"])==(users,rep)]
        first=[r for r in rs if r["turn"]==0]
        events=wave["cold_wave"](section,first)
        actual_first=sum(e["new"] for e in events)
        seconds=min(r["t0"]+r["ttft"] for r in first)-min(r["t0"] for r in first)
        # Total cold-wave time per prompt row from that repetition includes frontend and startup gaps.
        prompt_price=seconds/actual_first
        for commits in (2,4,6):
            for arrival in (0,17,71):
                for join_every in (0,1,2):
                    for held in (False,True):
                        stats={name:replay(rs,name,commits,arrival,join_every,held) for name in ("fcfs","shortest-prefill-first","lof")}
                        base=stats["fcfs"]
                        for name,stat in stats.items():
                            assert [base[k] for k in ("prompt_rows","full_attended_pairs","all_requested_turns","output_tokens")] == [stat[k] for k in ("prompt_rows","full_attended_pairs","all_requested_turns","output_tokens")]
                            # Charge the extra body rows to the prompt; non-body join cost is unknown.
                            for joint_charge in (0,.5,1):
                                def work(x):
                                    return prompt_price*(x["prompt_graph_rows"]+x["joint_extra_rows"])+x["standalone_seconds_proxy"]+joint_charge*x["joined_decode_seconds_proxy"]
                                if name!="fcfs" and join_every:
                                    aggregates[users,name].append(100*(work(base)/work(stat)-1))
                                if commits==4 and not arrival and join_every==1 and held and joint_charge==.5:
                                    primary.append(dict(users=users,rep=rep,policy=name,model_seconds=work(stat),**stat))
                                if commits==4 and not arrival and join_every==0 and held and joint_charge==0 and name=="fcfs":
                                    baseline.append(dict(users=users,rep=rep,measured_V41_wall_s=wall,proxy_s=work(stat),relative_error=work(stat)/wall-1))
    ranges=[]
    for (users,name),samples in aggregates.items():
        ranges.append(dict(users=users,policy=name,comparisons=len(samples),min_pct=min(samples),median_pct=statistics.median(samples),
                           max_pct=max(samples),faster=sum(x>0 for x in samples),slower=sum(x<0 for x in samples)))
    print(json.dumps(dict(candidate="V245",parent="V241",parent_sha256=hashlib.sha256((PARENT/"patch.diff").read_bytes()).hexdigest(),
        policy_source_sha256=policy_sha,client_sha256=wave["digest"](clients),server_sha256=wave["digest"](server),
        result_rows=primary,relative_proxy_ranges=ranges,FCFS_control_vs_archive=baseline,
        scope="Closed-customer replay of all 767 archived requests. Native policy comparators, continuation reservation and V241 graph-fit predicate. The simulation preserves every request's recorded input/cache and output length and first-turn arrival set. It models paper-time for compile-free serialized graph work, assumes fixed greedy acceptance, immediate client reissue, and initialized prefix metadata. It excludes host/rank/allocator stalls, per-layer memory traffic, priority clients and GPU precision. Costs derive from the archived V41 cold wave at that rung and the four earlier V64 decode periods. Neither the counterfactual schedule nor the estimated cost is a benchmark."),indent=2))


if __name__=="__main__":
    main()

