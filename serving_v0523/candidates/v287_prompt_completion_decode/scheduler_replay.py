"""Run the production scheduling loop and checkpoint helper on lightweight test requests."""
import ast
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace as S

stage = Path(sys.argv[1])
here = Path(__file__).resolve().parent
tree = json.loads((here/"validation.json").read_text())["tree"]
def source(p):
    return subprocess.check_output(["git","-C",stage,"show",tree+":python/sglang/srt/"+p]).decode()
def extract(text, names, context):
    for f in ast.walk(ast.parse(text)):
        if isinstance(f, ast.FunctionDef) and f.name in names:
            f.decorator_list=[]
            exec("from __future__ import annotations\n"+ast.unparse(f),context)
class Mode(Enum):
    EXTEND=1
    DECODE=2
    def is_extend(self):
        return self==Mode.EXTEND

class Req:
    def __init__(self,n,size,skip):
        self.rid=n
        self.size=size
        self.extend_end=0
        self.prefix_len=0
        self.complete=False
        self.skip_radix_cache_insert=skip
    def finished(self):
        return self.complete

class Batch:
    def __init__(self,reqs,mode,chunked=None):
        self.reqs=list(reqs)
        self.forward_mode=mode
        self.chunked_req=chunked
        self.contains_last_prefill_chunk=chunked is None or len(self.reqs)!=1
        self.batch_is_full=False
        self.is_prefill_only=False
    def filter_batch(self,reqs_to_exclude=()):
        self.reqs=[r for r in self.reqs if not r.finished() and r not in reqs_to_exclude]
    def batch_size(self):
        return len(self.reqs)
    def is_empty(self):
        return not self.reqs
    def merge_batch(self,other):
        assert not set(self.reqs).intersection(other.reqs)
        self.reqs.extend(other.reqs)

ctx=dict(ForwardMode=Mode,NextBatchPlan=S,set_schedule_time_batch=lambda b:None)
sched=source("managers/scheduler.py")
extract(sched,{"get_next_batch_to_run","_should_defer_prefill","_arm_prefill_decode_interval","stash_chunked_request"},ctx)
common=source("mem_cache/common.py")
extract(common,{"checkpoint_kv_cache"},ctx)
ctx["DisaggregationMode"]=S(PREFILL="prefill")
def run(lengths,cap,flag,period,cancel):
    requests=[Req(i,n,i%2==1) for i,n in enumerate(lengths)]
    pending=list(requests)
    prompt_log=[]
    decodes=[]
    checkpoints={r.rid:[] for r in requests}
    sched=S(require_mlp_sync=False,dllm_config=None,disaggregation_mode="null",chunked_req=None,
            enable_hisparse=False,prefill_decode_interval_min_bs=1,prefill_decode_interval=period,
            _prefill_decode_interval_remaining=0,prefill_decode_interval_on_completion=flag,
            process_pending_chunked_abort=lambda:None,_process_hicache_events=lambda:None,
            dp_attn_adapter=S(maybe_prepare_mlp_sync_batch=lambda b,**kw:b,maybe_convert_decode_to_extend=lambda b:b),
            ngram_embedding_manager=S(prepare_for_forward=lambda b,**kw:b))
    def check(r,up_to):
        assert up_to>r.prefix_len and (not checkpoints[r.rid] or up_to>checkpoints[r.rid][-1])
        checkpoints[r.rid].append(up_to)
    sched.tree_cache=S(checkpoint=check)
    for nm in ("stash_chunked_request","_should_defer_prefill","_arm_prefill_decode_interval","get_next_batch_to_run"):
        setattr(sched,nm,MethodType(ctx[nm],sched))
    # Test doubles replace only the GPU-backed batch preparation, allocator
    # and device model. The pinned scheduler controls admission and handoff.
    def prefill(running):
        remaining=cap
        chosen=[]
        while pending and remaining:
            r=pending[0]
            if r.extend_end:
                assert r.prefix_len==r.extend_end
            cnt=min(r.size-r.extend_end,remaining)
            first=r.extend_end
            r.extend_end+=cnt
            remaining-=cnt
            prompt_log.append((r.rid,first,r.extend_end))
            chosen.append(r)
            if r.extend_end==r.size:
                pending.pop(0)
                sched.chunked_req=None
            else:
                sched.chunked_req=r
        b=Batch(chosen,Mode.EXTEND,sched.chunked_req) if chosen else None
        return S(batch_to_run=b,running_batch=running)
    def decode(b):
        assert all(r.extend_end==r.size and not r.finished() for r in b.reqs)
        b.forward_mode=Mode.DECODE
        decodes.append(tuple(r.rid for r in b.reqs))
        return b
    sched.get_new_batch_prefill=prefill
    sched.update_running_batch=decode
    last=None
    run_batch=Batch([],Mode.DECODE)
    seen=[]
    boundaries=[]
    while pending:
        out=sched.get_next_batch_to_run(run_batch,last)
        run_batch=out.running_batch
        last=out.batch_to_run
        assert last is not None
        seen.append(last.forward_mode.name)
        if last.forward_mode==Mode.EXTEND:
            boundaries.append((len(seen)-1,last.contains_last_prefill_chunk))
            if cancel:
                for r in last.reqs:
                    if r.rid%3==0 and r.extend_end==r.size:
                        r.complete=True
        assert len(seen)<1000
    for req in requests:
        spans=[(a,b) for idx,a,b in prompt_log if idx==req.rid]
        assert spans[0][0]==0 and spans[-1][1]==req.size
        assert all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
    return seen,boundaries,prompt_log,decodes

cases=partials=refills=0
for lengths in ([9],[2,10],[5,13,2],[3,3,3,3,3],[30,2,9,15,1],[11,11,11,11,11]):
    for cap in (2,4,8):
        for p in (0,1,2):
            for cancel in (False, True):
                edge=run(lengths,cap,True,p,cancel)
                control=run(lengths,cap,False,p,cancel)
                assert edge[2]==control[2]
                assert len(edge[3])<=len(control[3])
                for i,completed in edge[1][:-1]:
                    if not completed:
                        assert edge[0][i+1]=="EXTEND"
                        partials+=1
                    elif edge[0][i+1]=="DECODE":
                        refills+=1
                cases+=1
print(json.dumps(dict(tree=tree,scheduler_sha256=sha256(sched.encode()).hexdigest(),
    checkpoint_sha256=sha256(common.encode()).hexdigest(),paired_workload_cases=cases,
    intermediate_chunks_immediately_followed_by_prefill=partials,completed_prompt_batches_followed_by_decode=refills,
    scope="Actual get_next_batch_to_run, stash, checkpoint and cadence methods; device forward, allocator and batch scaffolding substituted. Includes skipped radix insert, mixed completion + parked partial, cancelled finished prompt, filtered empty cohort and fresh decode."),indent=2))
