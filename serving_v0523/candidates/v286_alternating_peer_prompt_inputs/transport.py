"""Replay production allocation and alternating peer publication using real Gloo and mapped BF16 slabs."""
import argparse
import ast
import contextlib
import ctypes
from datetime import timedelta
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SOURCE="python/sglang/srt/layers/muse_spark/peer_prompt_pack.py"


def run_world(rank,world,stage,folder):
    torch.set_num_threads(1)
    root=Path(folder)
    dist.init_process_group("gloo",init_method="file://"+str(root/f"group{world}"),world_size=world,rank=rank,
                            timeout=timedelta(seconds=120))
    try:
        data=exercise(rank,world,Path(stage),root)
        (root/f"{world}_{rank}.json").write_text(json.dumps(data))
    finally:
        dist.destroy_process_group()


def exercise(rank,world,stage,root,grouped_check=None):
    rows,width=16,512
    state=NS(fault=None,mode=None,barriers=0,probing=False,packs=0,waits=0,writes=0,requested=[])
    class Comm:
        device=torch.device("cpu")
        disabled=False; has_multicast=True; pull_enabled=True; _is_graph_mode_supported=True
    comm=Comm(); comm.rank=rank; comm.world_size=world
    group=NS(world_size=world,rank_in_group=rank,device_group=dist.group.WORLD,cpu_group=dist.group.WORLD,ca_comm=comm)
    class Stream:
        def __init__(self,*args,**kwargs): pass
        def wait_stream(self,s): state.waits+=1
        def synchronize(self): pass
    class Handle:
        def __init__(self,local):
            self.rank=rank;self.world_size=world;self.local=local
        def get_buffer(self,r,shape,dtype):
            return self.local if r==rank else storage(r,shape,dtype)
        def barrier(self,channel):
            assert channel==0
            state.barriers+=1
            dist.barrier()
    def storage(r,shape,dtype):
        f=root/f"slab{world}_{state.mode}_{shape[0]}_{r}"
        return torch.from_file(str(f),shared=True,size=shape[0]*shape[1],dtype=dtype).view(shape)
    def alloc(shape,stride,dtype,dev,key):
        state.requested.append(shape[0]//rows)
        if rank==world-1 and (state.fault=="no_allocation" or state.fault=="large_allocation" and shape[0]==2*rows):
            raise RuntimeError("injected allocator failure")
        return storage(rank,shape,dtype)
    def rendezvous(t):
        dist.barrier()
        if rank==world-1 and state.fault=="large_registration" and t.size(0)==2*rows:
            return None
        return Handle(t)
    def pack(addresses, row_map, out, shard_rows):
        if state.probing and state.fault=="large_probe" and rank==world-1 and state.requested[-1]==2:
            raise RuntimeError("injected peer read failure")
        if not state.probing:
            state.packs+=1
            if (state.epoch+rank)%4==0:time.sleep(.003)
        for r,ptr in enumerate(addresses.tolist()):
            mapping=row_map[r*shard_rows:(r+1)*shard_rows]
            if not mapping.numel():continue
            row,slot=torch.where(mapping>=0)
            if not row.numel():continue
            raw=(ctypes.c_uint16*(shard_rows*width)).from_address(ptr)
            block=torch.frombuffer(raw,dtype=torch.int16).view(shard_rows,width)
            out.view(torch.int16)[mapping[row,slot].long()]=block[row]

    mod=ModuleType("sglang.srt.distributed.device_communicators.custom_all_reduce_v2")
    mod.CustomAllReduceV2=Comm
    binding=dict(torch=torch,threading=threading,logger=logging.getLogger("v286/replay"),dist=dist,
                 pack_peer_rows=pack,TritonError=RuntimeError)
    tree=ast.parse((stage/SOURCE).read_text())
    body=[n for n in tree.body if getattr(n,"name",None) in ("PeerPromptInputs","prepare_peer_prompt_inputs","peer_storage_rows")]
    exec("from __future__ import annotations\n"+ast.unparse(ast.Module(body,type_ignores=[])),binding)
    def data(epoch,r,n):
        return ((torch.arange(n*width,dtype=torch.int32).view(n,width)*197 + epoch*8291 + r*1009) & 65535).to(torch.int16)
    result=[]
    with contextlib.ExitStack() as contexts:
        for ctx in (patch.dict(sys.modules,{mod.__name__:mod}),patch.object(torch._C._distributed_c10d,"_SymmetricMemory",
                    NS(empty_strided_p2p=alloc,rendezvous=rendezvous),create=True),
                    patch.object(torch.cuda,"Stream",Stream),patch.object(torch.cuda,"stream",lambda _:contextlib.nullcontext()),
                    patch.object(torch.cuda,"current_stream",lambda *_:Stream()),patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
                    patch.object(torch.Tensor,"record_stream",lambda *_:None)):
            contexts.enter_context(ctx)
        for mode in ("ordinary","large_allocation","large_probe","large_registration","no_allocation"):
            state.fault=state.mode=mode; state.requested=[]; state.probing=True
            workspace=binding["prepare_peer_prompt_inputs"](group,width,2,rows*world,[rows*world],rows*world,rows*world)
            if mode=="no_allocation":
                assert workspace is None and state.requested==[2,1]
                result.append(dict(mode=mode,declined=True));continue
            banks=2 if mode=="ordinary" else 1
            assert len(workspace.banks)==banks and state.requested==([2] if banks==2 else [2,1])
            state.probing=False
            if grouped_check is not None:
                result.append(grouped_check(stage,binding,workspace,mode,rank,world,state,group,data,rows,width))
                continue
            count=0; checked=0; empty=0; before=state.barriers
            for epoch in range(27):
                state.epoch=epoch; n=(1,7,16,3)[epoch%4]; k=2
                # A unanimous eager fallback or capture stub runs no workspace method.
                if epoch in (3,8,19):
                    continue
                one=data(epoch,rank,n).view(torch.bfloat16)
                assert workspace.supports(one,group)
                all_rows=torch.cat([data(epoch,r,n) for r in range(world)])
                wanted=torch.arange(n*world*k).reshape(-1,k)
                routed=(wanted*13+epoch+rank)%5<3
                if (rank+epoch)%7==0:routed.fill_(False)
                if epoch==24:routed.fill_(False)
                indices=torch.full_like(wanted,-1,dtype=torch.int32)
                order=torch.randperm(int(routed.sum()),generator=torch.Generator().manual_seed(epoch+rank))
                indices[routed]=order.to(torch.int32)
                out=torch.empty(len(order),width,dtype=torch.bfloat16)
                gold=torch.empty_like(out,dtype=torch.int16)
                owner,slot=torch.where(routed)
                gold[order]=all_rows[owner]
                expected_bank=count%banks
                assert workspace.next_bank==expected_bank
                workspace.pack(one,indices if len(order) else None,out if len(order) else None)
                assert torch.equal(out.view(torch.int16),gold)
                checked+=out.numel(); count+=1;empty+=int(not len(order))
            assert state.barriers-before==count*(1 if banks==2 else 2)
            dist.barrier()
            result.append(dict(mode=mode,banks=banks,publications=count,empty_readers=empty,words=checked,barriers=state.barriers-before))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path);args=p.parse_args();stage=args.stage.resolve()
    all_cases={}
    with tempfile.TemporaryDirectory(prefix="v286-gloo-") as td:
        for world in (2,4,8):
            mp.spawn(run_world,args=(world,str(stage),td),nprocs=world)
            all_cases[world]=[json.loads((Path(td)/f"{world}_{r}.json").read_text()) for r in range(world)]
    counts={}
    for mode in ("ordinary","large_allocation","large_probe","large_registration","no_allocation"):
        cases=[c for ranks in all_cases.values() for r in ranks for c in r if c["mode"]==mode]
        counts[mode]={k:sum(c.get(k,0) for c in cases) for k in ("publications","words","barriers","empty_readers","declined")}
    print(json.dumps(dict(torch=torch.__version__,worlds=[2,4,8],header_sha256=hashlib.sha256((stage/SOURCE).read_bytes()).hexdigest(),
        counts=counts,words_compared=sum(c["words"] for c in counts.values()),mismatches=0,
        scope="Real Gloo startup votes and inter-process barriers, CPU pointers to shared mapped BF16 slabs, synchronous stream stand-ins. No CUDA time or H200 memory measurement."),indent=2))
if __name__=="__main__":
    main()
