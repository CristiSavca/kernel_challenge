"""Drive the staged model and actual Breakable graph callbacks with changing tensor contents and module hooks."""
import ast
from collections import Counter
from contextlib import nullcontext
from contextvars import ContextVar
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch

STAGE = Path(sys.argv[1])
TREE = subprocess.check_output(["git", "-C", STAGE, "write-tree"], text=True).strip()
P = "python/sglang/srt/"
PEER, HUM = [P+"layers/muse_spark/"+f for f in ("peer_prompt_pack.py","muse_spark_humming_moe.py")]
GRAPH = P+"model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py"
MODEL, SP = P+"models/muse_spark.py", P+"layers/layernorm_sp.py"
PKG = "sglang.srt.layers.muse_spark."
sources = {}
stats = Counter()
ctx = ContextVar("native_capture", default=None)


def bind(path, names, env, *, owner=None, assigns=()):
    blob = subprocess.check_output(["git","-C",STAGE,"show",TREE+":"+path])
    sources[path] = hashlib.sha256(blob).hexdigest()
    tree = ast.parse(blob)
    body = tree.body if owner is None else next(n.body for n in tree.body if getattr(n,"name",None)==owner)
    parts = [n for n in body if getattr(n,"name",None) in names or (
        isinstance(n, ast.Assign) and any(getattr(t,"id",None) in assigns for t in n.targets))]
    assert len(parts) == len(names)+len(assigns)
    exec("from __future__ import annotations\n"+ast.unparse(ast.Module(parts,[])), env)


def module(name, **attrs):
    m = ModuleType(name)
    vars(m).update(attrs)
    return m


def same(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def table_words(n, width, epoch):
    return (((torch.arange(n*width, dtype=torch.int32).view(n,width)*31 + epoch*127) % 2053)-1025).to(torch.bfloat16)


def test(world, start):
    # Three layers share the same two symmetric banks. Their distinct route dtypes
    # remain fixed for the lifetime of the captured graph.
    rank, width, k, n = world-1, 16, 4, 1029
    rows = (n+world-1)//world
    group = NS(world_size=world, rank_in_group=rank)
    state = NS(events=[], graph=None, phase=None, gather_packet=None, current=None)
    class Stream:
        def wait_stream(self, other):
            assert other is not self
            state.events.append("stream_wait")
    caller, side = Stream(), Stream()
    rt = dict(torch=torch, functools=functools, _current_capture_var=ctx)
    bind(GRAPH, ("_weak_ref_if_tensor","_copy_output","_EagerOnGraphCallable","eager_on_graph"), rt)
    hun = dict(torch=torch)
    bind(HUM, ("local_humming_expert_ids", "prompt_route_order"), hun)
    sp_env = dict(torch=torch,get_parallel=lambda:NS(tp_group=group))
    bind(SP, ("sp_exit_gather",), sp_env)
    def gather(output, value):
        assert ctx.get() is None
        case = state.current
        if value.ndim == 1:
            state.events.append("legacy_vote")
            assert bool(value[0]) == case.parent_votes[rank]
            output.copy_(torch.tensor(case.parent_votes,dtype=torch.int32))
        else:
            field = "shards" if value.shape[1]==width else "scores" if value.is_floating_point() else "ids"
            state.events.append("gather_"+field)
            data = getattr(case,field)
            same(data[rank],value)
            output.copy_(data.flatten(0,1))
    group.all_gather_into_tensor = gather
    sp = NS(sp_exit_gather=sp_env["sp_exit_gather"], sp_num_tokens=lambda:n,
            reduce_scatter_token_rows=lambda output,g:output)
    peer = dict(torch=torch,nn=torch.nn,threading=threading,eager_on_graph=rt["eager_on_graph"],layernorm_sp=sp)
    bind(PEER, ("PeerPromptInputs","_local_grouped_eligible","_peer_experts_into","_PublishedRows",
                "_try_published_routes","_published_experts_into","_capture_stub"), peer,
         assigns=("peer_experts_into","published_experts_into"))
    # Route map is identity at this grouped-input boundary. Backend math has its
    # own verification; the witness assembles a digest only after exact input checks.
    class Experts(torch.nn.Module):
        num_local_experts, intermediate_size_per_partition = 4, width
        use_w4a16 = use_humming = True
        _post_expert_norm_weight = object()
        post_expert_norm = _post_expert_norm = object()
        _expert_map = None
        def forward(self, x, scores, ids, norm):
            c = state.current
            assert norm is self.post_expert_norm
            same(x,c.shards.flatten(0,1)[:n]); same(scores,c.scores.flatten(0,1)[:n]); same(ids,c.ids.flatten(0,1)[:n])
            state.events.append("dense")
            stats["dense_original_score_and_id_bytes"] += scores.nbytes + ids.nbytes
            return c.golden
        def _grouped_humming_and_combine_impl(self, shard, ids, scores, out, *,peer_plan):
            c = state.current
            same(scores,c.scores.flatten(0,1)[:n]); same(ids,c.ids.flatten(0,1)[:n].int())
            use_legacy = peer_plan[0] is ws
            state.events.append("legacy_grouped" if use_legacy else "published_grouped")
            se,order,offs = hun["prompt_route_order"](ids,4)
            for a,b in zip((se,order,offs),peer_plan[1:4]):
                same(a,b)
            assert peer_plan[4] == int((ids<0).sum())
            b = ws.next_bank
            if use_legacy:
                state.phase = "legacy"
            # A deliberately empty output mapping still invokes the legacy
            # publication barrier. The published path already made its row copy.
            peer_plan[0].pack(shard,None,None)
            same(ws.banks[b][0][:rows],shard)
            if not use_legacy:
                assert peer_plan[0].addresses.data_ptr()==ws.banks[b][1].data_ptr()
            stats["grouped_route_bytes_checked"] += scores.nbytes+ids.nbytes
            out.copy_(c.golden)
    layer_sources = []
    def hook(*args):
        state.events.append("local_forward_hook")
    modules = [module(PKG+"muse_spark_fp4_moe", MuseSpark1xFp4Experts=Experts, _HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256),
        module(PKG+"muse_spark_humming_moe", **{n:hun[n] for n in ("local_humming_expert_ids","prompt_route_order")}),
        module(PKG+"peer_prompt_pack", **{n:peer[n] for n in ("peer_experts_into","published_experts_into")}),
        module("sglang.srt.layers.moe.token_dispatcher.standard",_map_topk_ids_to_local_experts=None),
        module("sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context",is_in_breakable_cuda_graph=lambda:ctx.get() is not None)]
    men = dict(torch=torch,layernorm_sp=sp,get_parallel=lambda:NS(tp_group=group))
    bind(MODEL, ("_forward_sp_replicated_experts",),men,owner="MuseSpark1xMoE")
    slots = rows*k
    packets = [[torch.full((2*slots+4,),-2345,dtype=torch.int32) for _ in range(world)] for _ in range(2)]
    def barrier(channel):
        assert channel==0 and ctx.get() is None
        b,c = ws.next_bank,state.current
        if state.phase == "legacy":
            state.events.append("legacy_publish")
            return
        state.events.append("route_publish")
        for r,t in enumerate(packets[b]):
            if c.routes[r]:
                value = torch.cat((c.scores[r].contiguous().view(torch.int32).flatten(),
                                   c.ids[r].flatten(), ws.yes))
            else:
                value = torch.zeros_like(t)
            if r == rank:
                same(t,value)
            else:
                t.copy_(value)
        if c.routes[rank]:
            same(ws.banks[b][0][:rows],c.shards[rank])
    class Workspace(peer["PeerPromptInputs"]):
        pass
    ws = Workspace.__new__(Workspace)
    ws.banks = [(torch.zeros(rows,width,dtype=torch.bfloat16),torch.tensor([10*b+q for q in range(world)])) for b in range(2)]
    ws.storage = ws.banks[0][0]; ws.route_packets = packets; ws.yes = torch.ones(4,dtype=torch.int32)
    ws.topk,ws.lock,ws.next_bank,ws.stream,ws.handle,ws.group = k,threading.Lock(),start,side,NS(barrier=barrier),group
    with (torch.no_grad(),patch.dict(sys.modules,{m.__name__:m for m in modules}),
          patch.object(torch.cuda,"is_current_stream_capturing",side_effect=lambda:ctx.get() is not None),
          patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
          patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),
          patch.object(torch.Tensor,"record_stream",lambda *_:None),
          patch.object(torch.cuda,"current_stream",lambda *_:caller),patch.object(torch.cuda,"stream",lambda _:nullcontext())):
        graph = NS(_break_fns=[])
        cap = NS(cuda_graph=graph,_barrier_fn=None,_end_current_segment=lambda:None,_begin_new_segment=lambda:None)
        t=ctx.set(cap)
        for i,(sd,idt) in enumerate(((torch.float32,torch.int32),(torch.float32,torch.int32),(torch.float64,torch.int64))):
            exp=Experts()
            model=NS(top_k=k,num_experts=4,experts=exp,_peer_prompt_inputs=ws)
            sh=torch.empty(rows,width,dtype=torch.bfloat16)
            scores=torch.empty(rows,k,dtype=sd); ids=torch.empty(rows,k,dtype=idt)
            out=men["_forward_sp_replicated_experts"](model,sh,scores,ids)
            assert not out.count_nonzero()
            layer_sources.append((exp,sh,scores,ids,out))
        ctx.reset(t)
        assert len(graph._break_fns)==3 and ws.next_bank==start and not state.events
        assert all((p==-2345).all() for bank in packets for p in bank)
        assert all(not sh.count_nonzero() for sh,_ in ws.banks)
        stats["empty_capture_stubs"] += len(graph._break_fns)
        for cycle in range(6):
            for i,(exp,sh,score,ids,out) in enumerate(layer_sources):
                mode = ("ok","local_hook","remote_hook")[(cycle+i)%3] if i<2 else ("wide_dense" if cycle%2==0 else "wide_legacy")
                local_hook = mode in ("local_hook","wide_dense")
                handle = exp.register_forward_hook(hook) if local_hook else None
                parent_votes=[True]*world
                parent_votes[rank] = not local_hook
                if mode=="remote_hook": parent_votes[0]=False
                routes = [v and i<2 for v in parent_votes]
                epoch=cycle*3+i+1
                shards=table_words(world*rows,width,epoch).view(world,rows,width)
                scores=((torch.arange(world*rows*k,dtype=score.dtype)*.001*(epoch+1))%3).reshape(world,rows,k)
                all_ids=((torch.arange(world*rows*k,dtype=ids.dtype)*17+epoch)%5-1).reshape(world,rows,k)
                if i<2:
                    # Preserve distinct NaN payloads and the sign of zero across FP32 views.
                    scores.view(torch.int32)[:,0,0]=0x7fc00000+epoch
                    scores.view(torch.int32)[:,0,1]=-2147483648
                elif mode=="wide_dense":
                    all_ids[:,0,0] = (1<<39)+epoch
                    scores[:,0,0] = 1+2.0**-40
                golden=table_words(n,width,epoch*29)
                state.current=NS(shards=shards,scores=scores,ids=all_ids,routes=routes,parent_votes=parent_votes,golden=golden)
                state.phase, state.events = "route", []
                sh.copy_(shards[rank]); score.copy_(scores[rank]); ids.copy_(all_ids[rank])
                ptrs = tuple(x.data_ptr() for x in (sh,score,ids,out))
                before=ws.next_bank
                graph._break_fns[i]()
                same(out,golden)
                assert tuple(x.data_ptr() for x in (sh,score,ids,out))==ptrs
                assert state.events.count("route_publish")==1
                if all(routes):
                    assert "published_grouped" in state.events and not any(s.startswith("gather") for s in state.events)
                else:
                    assert [s for s in state.events if s.startswith("gather_")][:2]==["gather_scores","gather_ids"]
                    assert "legacy_vote" in state.events
                    if all(parent_votes):
                        assert state.events.count("legacy_publish")==1 and "legacy_grouped" in state.events
                    else:
                        assert "dense" in state.events and "gather_shards" in state.events
                assert ("local_forward_hook" in state.events) == local_hook
                assert ws.next_bank == (before ^ 1 ^ state.events.count("legacy_publish"))
                if handle is not None: handle.remove()
                stats[mode] += 1
                stats["published_epochs"] += 1
                stats["legacy_publication_epochs"] += state.events.count("legacy_publish")


if __name__ == "__main__":
    torch.set_num_threads(1)
    for size in (2,4,8):
        for start in (0,1):
            test(size,start)
    print(json.dumps(dict(tree=TREE, source_sha256=sources, results=dict(sorted(stats.items())),
        coverage="Host model entry, production Breakable decorator/capture stub/replay and native fallback gathers. Remote packet and stream operations are synchronous simulations; backend compute is a checked witness. No device timing.",physical_gpu=False),indent=2))
