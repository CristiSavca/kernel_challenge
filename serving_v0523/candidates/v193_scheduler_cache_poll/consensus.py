"""Portable CPU-only prototype for combining a scheduler inbox-size broadcast and an idle cache vote."""
import ast
import json
import os
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT=Path(__file__).resolve().parents[3]
PINNED=ROOT/"serving_v0523/sglang/python/sglang/srt/utils/common.py"


def pinned_broadcast():
    source=ast.parse(PINNED.read_bytes())
    body=next(n for n in source.body if getattr(n,"name",None)=="broadcast_pyobj")
    env=dict(torch=torch,np=np,dist=dist,pickle=pickle,List=List,Any=Any,Optional=Optional,is_musa=lambda:False)
    exec(compile(ast.Module(body=[body],type_ignores=[]),str(PINNED),"exec"),env)
    return env["broadcast_pyobj"]


def vote(rank, message_size, cache_clean, digest):
    # Root contributes its real message size; MIN hands that size to every rank.
    values=torch.tensor([-message_size if rank==0 else 0, int(cache_clean), digest, -digest],dtype=torch.int64)
    dist.all_reduce(values, op=dist.ReduceOp.MIN)
    size,clean,minimum,neg_max=values.tolist()
    # Native receipt/payload and HiCache checks stay live on any veto or incoming message.
    return -size, bool(clean and size==0 and minimum==-neg_max)


def bench(rank,world,dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(dir).resolve()/"pg"),rank=rank,world_size=world)
    send=pinned_broadcast()
    scenarios=0
    for size in (0,1,129000):
        for dirty in (-1,*range(world)):
            for skew_digest in (False,True):
                minimum,fast=vote(rank,size,rank!=dirty,17+int(skew_digest and rank==world-1))
                assert minimum==size and fast==(size==0 and dirty==-1 and not skew_digest)
                scenarios+=1
    rank_times=[[],[]]
    for repeat in range(11):
        for choice in ((0,1) if repeat%2==0 else (1,0)):
            dist.barrier()
            t=time.perf_counter_ns()
            for _ in range(40):
                if choice==0:
                    assert send([] if rank==0 else None,rank,dist.group.WORLD)==[]
                    v=torch.tensor([0,0,17,-17],dtype=torch.int64)
                    dist.all_reduce(v,op=dist.ReduceOp.MIN)
                    assert v.tolist()==[0,0,17,-17]
                else:
                    assert vote(rank,0,True,17)==(0,True)
            rank_times[choice].append((time.perf_counter_ns()-t)/40e6)
    packets=[None]*world
    dist.all_gather_object(packets,rank_times)
    if rank==0:
        baselines=[[max(p[c][j] for p in packets) for j in range(1,11)] for c in (0,1)]
        native,combined=map(statistics.median,baselines)
        Path(dir,"results.json").write_text(json.dumps(dict(ranks=world,scenarios=scenarios,
            native_ms=native,one_min_ms=combined,median_paired_difference_ms=statistics.median(a-b for a,b in zip(*baselines)),
            native_trials_ms=baselines[0],one_min_trials_ms=baselines[1])))
    dist.destroy_process_group()


if __name__=="__main__":
    records=[]
    for world in (2,4,8):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(bench,args=(world,directory),nprocs=world,join=True)
            records.append(json.loads(Path(directory,"results.json").read_bytes()))
    print(json.dumps(dict(torch=torch.__version__,machine=os.uname().machine,hardware="local CPU Gloo loopback; no H200 service",records=records),indent=2))
