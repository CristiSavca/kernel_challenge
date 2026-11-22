"""Execute V295's public startup with host tensor storage and asymmetric admission fixtures."""
import ast
import contextlib
import hashlib
import json
import logging
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch._C._distributed_c10d import _SymmetricMemory

path=Path(sys.argv[1])/"python/sglang/srt/layers/muse_spark/peer_prompt_pack.py"
tree=ast.parse(path.read_bytes())
names={"PeerPromptInputs","peer_storage_rows","prepare_peer_prompt_inputs"}
nodes=[n for n in tree.body if getattr(n,"name",None) in names]
assert len(nodes)==3
class Stream:
    def __init__(self,*args,**kwargs): pass
    def synchronize(self): pass
class Comm:
    disabled=False
    has_multicast=True
    pull_enabled=True
    _is_graph_mode_supported=True
    device=torch.device("cpu")
module=ModuleType("custom_all_reduce_v2")
module.CustomAllReduceV2=Comm
sys.modules["sglang.srt.distributed.device_communicators.custom_all_reduce_v2"]=module
def check(world, width, topk, rows, faults=(), rank=0, group_ok=True, inspect=True):
    group=SimpleNamespace(world_size=world,rank_in_group=rank,cpu_group="cpu",device_group=SimpleNamespace(group_name="group"))
    group.ca_comm=Comm()
    group.ca_comm.world_size,group.ca_comm.rank=world,rank
    state=SimpleNamespace(try_index=-1,shapes=[],slabs=[],attempt_votes=0,remote_ptrs=[],group=group)
    def agree(out, value, group):
        if isinstance(value,tuple):
            out[:]=[value]*world
            return
        out[:]=[value]*world
        state.attempt_votes+=1
        if (state.try_index,state.attempt_votes) in faults:
            out[-1]=False
    def allocate(shape,stride,dtype,device,groupname):
        state.try_index+=1
        state.attempt_votes=0
        assert groupname=="group" and stride==(width,1) and dtype==torch.bfloat16
        state.shapes.append(shape)
        # Avoid allocating a bulk dummy working set for the over-budget metadata case.
        if not inspect or (state.try_index,"allocation") in faults: raise RuntimeError("modeled allocation veto")
        state.slabs=[torch.full(shape,-77,dtype=torch.bfloat16) for _ in range(world)]
        return state.slabs[rank]
    def rendezvous(storage):
        assert storage.data_ptr()==state.slabs[rank].data_ptr()
        if (state.try_index,"registration") in faults: raise RuntimeError("modeled registration veto")
        return SimpleNamespace(rank=rank,world_size=world,get_buffer=lambda i,shape,dtype:state.slabs[i])
    def probe(addresses,mapping,output,shard_rows):
        assert int(torch.count_nonzero(mapping>=0))==1 and mapping[-1,0]==0 and shard_rows==2
        if (state.try_index,"probe") in faults: raise RuntimeError("modeled probe veto")
        state.remote_ptrs.extend(addresses.tolist())
    dist=SimpleNamespace(get_backend=lambda _:"gloo" if group_ok else "nccl",all_gather_object=agree)
    log=logging.Logger("fixture")
    log.addHandler(logging.NullHandler())
    env=dict(torch=torch,threading=threading,dist=dist,pack_peer_rows=probe,logger=log,TritonError=RuntimeError)
    exec(compile(ast.Module(nodes,[]),str(path),"exec"),env)
    with contextlib.ExitStack() as ctx:
        for cm in (patch.object(_SymmetricMemory,"empty_strided_p2p",side_effect=allocate),
                   patch.object(_SymmetricMemory,"rendezvous",side_effect=rendezvous),
                   patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
                   patch.object(torch.cuda,"current_stream",return_value=Stream()),
                   patch.object(torch.cuda,"Stream",Stream)):
            ctx.enter_context(cm)
        ws=env["prepare_peer_prompt_inputs"](group,width,topk,rows*world,[rows*world],rows*world,rows*world)
    kind="declined"
    if ws is not None:
        kind="published" if ws.route_packets else "legacy"+str(len(ws.banks))
        assert ws.storage.shape==(rows,width)
        pitch=state.shapes[-1][0]//len(ws.banks)
        spans={}
        for i,(bank,addresses) in enumerate(ws.banks):
            assert bank.data_ptr()==ws.storage.data_ptr()+i*pitch*width*2
            for r in range(world):
                h=(state.slabs[r].data_ptr()+2*i*pitch*width,2*rows*width)
                assert addresses[r].item()==h[0]
                spans.setdefault(r,[]).append(h)
                if ws.route_packets:
                    packet=ws.route_packets[i][r]
                    assert packet.data_ptr()==h[0]+h[1] and packet.numel()>=2*rows*topk+4
                    assert packet.dtype==torch.int32 and packet.is_contiguous()
                    spans[r].append((packet.data_ptr(),packet.numel()*4))
        for regions in spans.values():
            assert all(a+n<=b for (a,n),(b,_) in zip(regions,regions[1:]))
        if ws.route_packets:
            assert ws.yes.tolist()==[1]*4 and len(ws.banks)==2
    return dict(world=world,rank=rank,hidden=width,topk=topk,rows=rows,expected_faults=list(faults),
                selection=kind,attempted_shapes=state.shapes)
cases=[]
for world in (2,4,8):
    for rank in (0,world-1):
        for faults,wanted in [((),"published"), (((0,"allocation"),),"legacy2"), (((0,"registration"),),"legacy2"),
                             (((0,"probe"),),"legacy2"), (((0,1),),"legacy2"), (((0,2),),"legacy2"),
                             (((0,1),(1,2)),"legacy1"), (((0,2),(1,2),(2,2)),"declined")]:
            r=check(world,64,8,17,faults,rank=rank)
            assert r["selection"]==wanted,r
            cases.append(r)
r=check(4,63,7,17)
assert r["selection"]=="legacy2" and r["attempted_shapes"]==[(34,63)]
cases.append(r)
r=check(2,4,16,131072,inspect=False)
assert r["selection"]=="declined" and r["attempted_shapes"]==[(262144,4),(131072,4)]
cases.append(r)
r=check(2,64,8,17,group_ok=False)
assert r["selection"]=="declined" and r["attempted_shapes"]==[]
cases.append(r)
print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=cases,
    scope="Host Torch, simulated collective votes and mapped storage. Exact startup, slab constructor and probe control flow. CUDA and IPC not executed."),indent=2))
