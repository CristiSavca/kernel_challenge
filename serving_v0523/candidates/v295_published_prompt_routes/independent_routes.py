"""Run the staged model entry across Gloo ranks and CPU-mapped symmetric slabs."""
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
from unittest.mock import patch, PropertyMock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


P = "python/sglang/srt/"
MODEL, SP = P+"models/muse_spark.py", P+"layers/layernorm_sp.py"
PEER = P+"layers/muse_spark/peer_prompt_pack.py"
HUM = P+"layers/muse_spark/muse_spark_humming_moe.py"
PACKAGE = "sglang.srt.layers.muse_spark."
CAPACITY, WIDTH, K = 270, 32, 2


def bind(stage, path, names, env, owner=None):
    tree = ast.parse((stage/path).read_bytes())
    nodes = tree.body if owner is None else next(n.body for n in tree.body if getattr(n,"name","")==owner)
    pieces = [n for n in nodes if getattr(n,"name","") in names]
    assert len(pieces) == len(names)
    exec("from __future__ import annotations\n"+ast.unparse(ast.Module(pieces,type_ignores=[])), env)


def module(name, **attrs):
    mod = ModuleType(name)
    vars(mod).update(attrs)
    return mod


class Marked(torch.Tensor):
    """A single-rank type veto with the original FP32 collective wire format."""


def fixture(rank, world, stage, root):
    state = NS(mode="", case="", epoch=0, records=[], events=[], counts={})
    parallel = NS()
    class Comm:
        device = torch.device("cpu")
        disabled = False
        has_multicast = pull_enabled = _is_graph_mode_supported = True
    comm = Comm()
    comm.rank, comm.world_size = rank, world
    group = NS(world_size=world, rank_in_group=rank, ca_comm=comm, device_group=dist.group.WORLD,
               cpu_group=dist.group.WORLD)
    parallel.tp_group = group

    class Stream:
        def __init__(self, *args, **kwargs): pass
        def wait_stream(self, other): pass
        def synchronize(self): pass

    def slab(owner, shape, dtype):
        size = shape[0]*shape[1]
        path = root/f"slab-{world}-{state.mode}-{size}-{owner}"
        return torch.from_file(str(path),shared=True,size=size,dtype=dtype).view(shape)

    class Handle:
        def __init__(self, storage):
            self.rank, self.world_size = rank, world
            self.storage = storage
        def get_buffer(self, owner, shape, dtype):
            return self.storage if owner == rank else slab(owner,shape,dtype)
        def barrier(self, channel):
            assert channel == 0
            state.events.append("publication")
            dist.barrier()

    def allocate(shape, strides, dtype, device, key):
        if rank==world-1 and (state.mode=="absent" or
                state.mode=="legacy" and shape[0]>2*CAPACITY):
            raise RuntimeError("injected one-rank allocation decline")
        return slab(rank,shape,dtype)

    def rendezvous(storage):
        dist.barrier()
        return Handle(storage)

    def copy_rows(addresses, mapping, out, rows):
        if not state.probing and (rank+state.epoch)%3==0:
            time.sleep(.002)
        for owner,ptr in enumerate(addresses.tolist()):
            dst = mapping[owner*rows:(owner+1)*rows]
            row,slot = torch.where(dst>=0)
            if row.numel():
                raw = (ctypes.c_int16*(rows*WIDTH)).from_address(ptr)
                words = torch.frombuffer(raw,dtype=torch.int16).view(rows,WIDTH)
                out.view(torch.int16)[dst[row,slot].long()] = words[row]

    def gather(output, value):
        label = ("vote" if value.ndim==1 else "activation" if value.shape[1]==WIDTH else
                 "ids" if not value.is_floating_point() else "scores")
        state.events.append(label)
        wire = value.as_subclass(torch.Tensor)
        buffers = [torch.empty_like(wire) for _ in range(world)]
        dist.all_gather(buffers,wire)
        output.copy_(torch.cat(buffers))
    group.all_gather_into_tensor = gather

    def result(ids):
        local = torch.where((ids>=rank*2)&(ids<rank*2+2),ids-rank*2,-1)
        return (local+1).sum(-1).unsqueeze(-1).expand(-1,WIDTH).bfloat16()

    def scatter(output, checked_group):
        assert checked_group is group and torch.equal(output,result(state.ids))
        # The output reduction is outside the protocol under test. Avoid an extra
        # group rendezvous here so adjacent eligible invocations can race to reuse.
        return output[rank*state.rows:(rank+1)*state.rows]

    sp_env = dict(torch=torch,get_parallel=lambda:parallel)
    bind(stage,SP,{"sp_exit_gather"},sp_env)
    sp = NS(sp_num_tokens=lambda:state.n,sp_exit_gather=sp_env["sp_exit_gather"],reduce_scatter_token_rows=scatter)
    def invert(grid):
        def call(shard,slots,inputs,mapped,first,gather_rows,total,*args,**kwargs):
            assert not gather_rows and total==mapped.numel()
            mapped.fill_(-1)
            mapped.flatten()[slots[first:]] = torch.arange(total-first,dtype=mapped.dtype)
        return call
    class Map:
        def __getitem__(self, grid): return invert(grid)
    hen = dict(torch=torch,_GROUPED_COMPUTE={},_pack_prompt_routes=Map(),
               triton=NS(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length()))
    bind(stage,HUM,{"humming_grouped_expert_outputs","local_humming_expert_ids","prompt_route_order"},hen)

    def verify_routes(scores,ids,local=False):
        target_ids = (torch.where((state.ids>=rank*2)&(state.ids<rank*2+2),state.ids-rank*2,-1).int()
                      if local else state.ids)
        assert ids.dtype==target_ids.dtype and torch.equal(ids,target_ids)
        assert scores.dtype==state.scores.dtype
        assert torch.equal(scores.contiguous().view(torch.uint8),state.scores.contiguous().view(torch.uint8))
        state.counts["route_pairs"] += ids.numel()

    class Experts:
        num_local_experts, intermediate_size_per_partition = 2, WIDTH
        use_w4a16 = use_humming = True
        _post_expert_norm_weight = object()
        post_expert_norm = _post_expert_norm = object()
        _expert_map = object()
        humming_prompt_gate_up = True
        humming_prompt_down = None
        humming_grouped_tunings = {"w13":{},"w2":{}}
        _forward_pre_hooks = {}
        def __call__(self, h, scores, ids, norm):
            state.events.append("dense")
            assert norm is self.post_expert_norm and torch.equal(h.view(torch.int16),state.gold)
            verify_routes(scores,ids)
            state.counts["activation_words"] += h.numel()
            return result(ids)
        def _grouped_humming_and_combine_impl(self, h, ids, scores, out, *,peer_plan):
            state.events.append("grouped")
            verify_routes(scores,ids,local=True)
            actual,mapping = hen["humming_grouped_expert_outputs"](self,h,ids,defer_output_scale=True,peer_plan=peer_plan)
            positions = torch.where(ids.flatten()>=0)[0]
            order = positions[torch.argsort(ids.flatten()[positions],stable=True)]
            assert torch.equal(actual.view(torch.int16),state.gold[order//K])
            if order.numel():
                golden = torch.full_like(ids,-1).flatten()
                golden[order] = torch.arange(order.numel(),dtype=ids.dtype)
                assert torch.equal(mapping.flatten(),golden)
            else:
                state.counts["empty_local"] += 1
            state.counts["activation_words"] += actual.numel()
            out.copy_(result(state.ids))
    experts = Experts()
    modules = [module(PACKAGE+"muse_spark_fp4_moe",MuseSpark1xFp4Experts=Experts,_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256),
        module(PACKAGE+"muse_spark_humming_moe",**{k:hen[k] for k in ("local_humming_expert_ids","prompt_route_order")}),
        module(PACKAGE+"decode_gate_up",run_gate_up=lambda layer,inputs,*args,**kwargs:inputs.clone()),
        module("sglang.srt.layers.moe.token_dispatcher.standard",_map_topk_ids_to_local_experts=
               lambda x,_:torch.where((x>=rank*2)&(x<rank*2+2),x-rank*2,-1)),
        module("sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context",is_in_breakable_cuda_graph=lambda:False),
        module("sglang.srt.distributed.device_communicators.custom_all_reduce_v2",CustomAllReduceV2=Comm),
        module("humming.layer",HummingMethod=NS(forward_layer=lambda layer,*,inputs,outputs,**kw:outputs.copy_(inputs))),
        module("sglang.kernels.ops.activation.activation",silu_and_mul=None)]
    pen = dict(torch=torch,nn=torch.nn,threading=threading,dist=dist,logger=logging.getLogger("fixture"),
               TritonError=RuntimeError,layernorm_sp=sp,pack_peer_rows=copy_rows)
    bind(stage,PEER,{"PeerPromptInputs","peer_storage_rows","prepare_peer_prompt_inputs","_local_grouped_eligible",
                    "_peer_experts_into","_PublishedRows","_try_published_routes","_published_experts_into"},pen)
    modules.append(module(PACKAGE+"peer_prompt_pack",peer_experts_into=pen["_peer_experts_into"],
                          published_experts_into=pen["_published_experts_into"]))
    men = dict(torch=torch,layernorm_sp=sp,get_parallel=lambda:parallel)
    bind(stage,MODEL,{"_forward_sp_replicated_experts"},men,"MuseSpark1xMoE")
    model = NS(top_k=K,num_experts=world*2,experts=experts)
    stack = torch.stack
    def skewed_stack(views,*args,**kwargs):
        if len(views)==world and views[0].ndim==1 and (state.epoch+rank)%3==1:
            time.sleep(.003)
        return stack(views,*args,**kwargs)
    with (torch.no_grad(),patch.dict(sys.modules,{m.__name__:m for m in modules}),
          patch.object(torch._C._distributed_c10d,"_SymmetricMemory",NS(empty_strided_p2p=allocate,rendezvous=rendezvous),create=True),
          patch.object(torch.cuda,"Stream",Stream),patch.object(torch.cuda,"current_stream",lambda *_:Stream()),
          patch.object(torch.cuda,"stream",lambda s:contextlib.nullcontext()),
          patch.object(torch.cuda,"is_current_stream_capturing",return_value=False),
          patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
          patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),
          patch.object(torch.Tensor,"record_stream",lambda *_:None),patch.object(torch,"stack",skewed_stack)):
        for mode in ("published","legacy","absent"):
            state.mode, state.probing = mode, True
            workspace = pen["prepare_peer_prompt_inputs"](group,WIDTH,K,CAPACITY*world,[CAPACITY*world],CAPACITY*world,CAPACITY*world)
            model._peer_prompt_inputs = workspace
            assert (workspace is None)==(mode=="absent")
            if workspace is not None: assert bool(workspace.route_packets)==(mode=="published")
            state.probing = False
            cases = (["ragged","accepted","empty","hook","ragged","wide_ids","marked","noncontig","oversize","short",
                      "all_remote","accepted","hook","accepted","accepted"] if mode=="published" else
                     ["ragged","hook","oversize","short"])
            for epoch,case in enumerate(cases):
                state.epoch, state.case, state.events = epoch,case,[]
                state.counts = dict(route_pairs=0,activation_words=0,empty_local=0)
                state.n = (13 if case=="short" else (CAPACITY+3)*world if case=="oversize" else
                           (CAPACITY-(epoch%4))*world-(epoch+1)%(world-1)-1)
                state.rows = (state.n+world-1)//world
                length = state.rows*world
                token = torch.arange(length)[:,None]
                fields = token*K+torch.arange(K)
                ids = ((fields*11+epoch)%(world*2)).int()
                if case=="empty": ids.fill_(0)
                if case=="all_remote": ids.fill_(-1)
                if case=="wide_ids": ids=ids.long()
                words = ((token*5923+torch.arange(WIDTH)*197+epoch*9991)&65535).short()
                scores = (fields*9091+epoch*779)&0x007fffff | 0x3f000000
                scores = scores.int()
                scores[::13,0] = -2147483648
                scores[::37,1] = 0x7fc0abcd
                scores = scores.view(torch.float32)
                state.ids,state.scores,state.gold = ids[:state.n],scores[:state.n],words[:state.n]
                local = slice(rank*state.rows,(rank+1)*state.rows)
                ls,li,h = scores[local],ids[local],words[local].view(torch.bfloat16)
                if case=="marked" and rank==world-1: ls=ls.as_subclass(Marked)
                if case=="noncontig": ls,li=ls.T.contiguous().T,li.T.contiguous().T
                experts._forward_hooks = {0:True} if case=="hook" and rank==world-1 else {}
                before = workspace.next_bank if workspace else None
                men["_forward_sp_replicated_experts"](model,h,ls,li)
                early = mode=="published" and case not in ("short","oversize")
                old = workspace is not None and case!="short" and (not early or case in ("hook","wide_ids","marked"))
                dense = workspace is None or case in ("short","oversize","hook")
                publications = int(early)+int(old and not dense)
                for label,count in dict(publication=publications,vote=int(old),scores=int(not early or old),
                         ids=int(not early or old),activation=int(dense),grouped=int(not dense),dense=int(dense)).items():
                    assert state.events.count(label)==count,(world,rank,mode,epoch,case,label,state.events)
                if workspace: assert workspace.next_bank==(before+publications)%2
                state.records.append(dict(mode=mode,case=case,events=state.events,**state.counts))
            dist.barrier()
    return state.records


def worker(rank,world,stage,folder):
    torch.set_num_threads(1)
    root = Path(folder)
    dist.init_process_group("gloo",init_method="file://"+str(root/f"g{world}"),world_size=world,rank=rank,
                            timeout=timedelta(seconds=25))
    try:
        checks = fixture(rank,world,Path(stage),root)
        (root/f"{world}-{rank}.json").write_text(json.dumps(checks))
    finally:
        dist.destroy_process_group()


def main():
    stage = Path(sys.argv[1]).resolve()
    detail = {}
    with tempfile.TemporaryDirectory(prefix="v295-independent-") as tmp:
        for world in (4,8):
            mp.spawn(worker,args=(world,str(stage),tmp),nprocs=world)
            rows = [json.loads((Path(tmp)/f"{world}-{rank}.json").read_text()) for rank in range(world)]
            for run in zip(*rows):
                first = run[0]
                key = f"tp{world}/{first['mode']}/{first['case']}"
                target = detail.setdefault(key,dict(observed_calls=0,route_pairs=0,activation_words=0,empty_local=0,events=first["events"]))
                for row in run:
                    assert row["events"]==target["events"]
                    target["observed_calls"]+=1
                    for stat in ("route_pairs","activation_words","empty_local"): target[stat]+=row[stat]
    print(json.dumps(dict(sources={p:hashlib.sha256((stage/p).read_bytes()).hexdigest() for p in (MODEL,SP,PEER,HUM)},
            torch=torch.__version__,totals={k:sum(x[k] for x in detail.values()) for k in ("observed_calls","route_pairs","activation_words","empty_local")},
            detail=detail,scope="Gloo, mapped CPU slabs, per-rank reader skew and staged Python SP model, storage, packet, rank checks, fallback gathers, and grouped routing. Host CUDA streams and devices, Triton route inversion, Humming arithmetic and final reduction are modeled."),indent=2))


if __name__=="__main__":
    main()
