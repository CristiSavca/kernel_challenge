"""Exercise alternating modes through the same staged MoE, experts and peer banks."""
import argparse
import ast
import contextlib
from hashlib import sha256
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch, PropertyMock

import torch


SRT = "python/sglang/srt/"
MODEL = SRT + "models/muse_spark.py"
HUM = SRT + "layers/muse_spark/muse_spark_humming_moe.py"
PEER = SRT + "layers/muse_spark/peer_prompt_pack.py"
PACKAGE = "sglang.srt.layers.muse_spark."
WIDTH, TOPK = 64, 8


def methods(stage, path, owner, names, env):
    tree = ast.parse((stage / path).read_bytes())
    body = tree.body if owner is None else next(n.body for n in tree.body if getattr(n,"name","") == owner)
    for name in names:
        node = next(n for n in body if getattr(n,"name","") == name)
        exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return {n:env[n] for n in names}


def install(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


class Stream:
    def __init__(self, name, state):
        self.name, self.state = name, state

    def wait_stream(self, other):
        self.state.events.append(f"{self.name} waits {other.name}")


class Norm:
    hidden_size, eps, zero_centered_gamma = WIDTH, 1e-6, True

    def __call__(self, values):
        return torch.nn.functional.rms_norm(values, (WIDTH,), self.weight, self.eps)


def sorted_plan(ids):
    slots = ids.numel()
    local = ids.reshape(-1)
    valid = (local >= 0) & (local < 32)
    buckets = torch.where(valid, local + 1, 0)
    order = torch.argsort(buckets, stable=True)
    row_map = torch.full((slots,),-1,dtype=torch.int32)
    row_map[order] = torch.where(valid[order],torch.arange(slots),-1).to(torch.int32)
    return buckets, order, row_map.reshape_as(ids)


def arithmetic(values, ids):
    valid = (ids >= 0) & (ids < 32)
    # Id-dependent output distinguishes local, remote and duplicate slots.
    gain = torch.where(valid, ids + 1, 0).sum(dim=1).float().unsqueeze(-1) / 32
    return (values.float() * gain).bfloat16()


def run_world(stage, world):
    st = NS(active="main", sp=False, capture=False, events=[], plan_ids=[], cases=[])
    main, side, transport = [Stream(n,st) for n in ("main","router","peer")]
    @contextlib.contextmanager
    def use_stream(stream):
        previous = st.active
        st.active = stream.name
        try:
            yield
        finally:
            st.active = previous

    class Group:
        rank_in_group = 0
        world_size = world
        ca_comm = None

        def all_gather_into_tensor(self, votes, vote):
            votes.fill_(int(vote))
            if st.peer_veto:
                votes[-1] = 0
            st.events.append("peer vote")

    group = Group()
    norm = Norm()
    norm.weight = torch.linspace(.5,1.5,WIDTH).bfloat16()
    norm._frozen_weight = norm.weight
    def gather_tokens(tensor, count):
        assert count == st.n
        if tensor.dtype == torch.int64:
            return st.ids
        if tensor.shape[1] == TOPK:
            return torch.ones_like(st.ids, dtype=torch.float32)
        return norm(st.data)
    sp = NS(sp_num_tokens=lambda:st.n, sp_exit_gather=gather_tokens,
            reduce_scatter_token_rows=lambda x,g:x[:st.n//world])
    def column(group, x, weight, *,rank_major):
        assert group is not None and not st.sp
        return x.view(x.shape[0],world,WIDTH//world).transpose(0,1).contiguous() if rank_major else x
    def route(*args, **kwargs):
        ids = st.ids[:st.n//world] if st.sp else st.ids
        st.events.append("route:"+st.active)
        return torch.ones_like(ids,dtype=torch.float32), ids, None

    class PlanKernel:
        def __getitem__(self, grid):
            def fill(ids, expert_map, local, offsets, keys, rowmap, slots, **kw):
                buckets, order, expected = sorted_plan(ids)
                local.copy_(torch.where((ids>=0)&(ids<32),ids,-1))
                offsets.copy_(torch.bincount(buckets,minlength=33).cumsum(0))
                keys.copy_(buckets[order]*kw["SLOT_SPAN"]+order)
                rowmap.copy_(expected)
                st.events.append("plan:"+st.active)
            return fill

    hum_env = dict(torch=torch,triton=NS(next_power_of_2=lambda n:1<<(n-1).bit_length()),
                   _MAX_DECODE_SLOTS=2048, _ranked_pool_module=lambda:object(),
                   _plan_grouped_decode=PlanKernel())
    hum = methods(stage,HUM,None,["allocate_decode_plan","fill_decode_plan","prepare_ranked_pool",
                                  "humming_decode_eligible"],hum_env)
    def order(ids, experts):
        buckets, order, _ = sorted_plan(ids)
        return buckets[order], order, torch.bincount(buckets,minlength=33).cumsum(0)
    install(PACKAGE+"muse_spark_humming_moe", **hum, _MAX_DECODE_SLOTS=2048,
            local_humming_expert_ids=lambda e,i:torch.where((i>=0)&(i<32),i,-1),
            prompt_route_order=order)
    install(PACKAGE+"muse_spark_layers", MuseSpark1xRMSNorm=Norm)
    install("sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context",
            is_in_breakable_cuda_graph=lambda:st.breakable)

    class Experts:
        use_w4a16 = use_humming = True
        post_expert_norm = _post_expert_norm = object()
        _post_expert_norm_weight = object()
        _expert_map = None
        _forward_hooks, _forward_pre_hooks = {}, {}
        humming_decode_tunings = {"w13":[1],"w2":[1]}
        intermediate_size_per_partition, num_local_experts, global_num_experts = WIDTH,32,256

        def __call__(self, values, scores, ids, weight, **kw):
            if st.sp:
                st.events.append("prompt dense fallback")
                assert values.shape == (st.n,WIDTH) and "ranked_work" not in kw
                return arithmetic(values, ids)
            work = kw.get("ranked_work")
            if work is not None:
                staging, plan = work
                assert st.capture and st.active == "main" and staging[1] is values
                assert torch.equal(plan[3], sorted_plan(ids)[2])
                assert torch.equal(plan[0],torch.where((ids>=0)&(ids<32),ids,-1))
                st.plan_ids.append(plan[3].clone())
                st.events.append("consume ranked plan")
            elif st.expect_early:
                raise AssertionError("missing ranked plan")
            if values.ndim == 3:
                values = values.transpose(0,1).reshape(st.n,WIDTH)
            return arithmetic(norm(values),ids)

        def _grouped_humming_and_combine_impl(self, shard, local_ids, scores, output, *,peer_plan):
            peer, sorted_ids, slots, offsets, inactive = peer_plan
            temporary = output.new_empty(output.shape)
            previous = peer.next_bank
            peer.pack(shard, slots, temporary)
            output.copy_(arithmetic(temporary,local_ids))
            st.events.append(f"published bank {previous}")

    experts = Experts()
    install(PACKAGE+"muse_spark_fp4_moe", _HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256,
            MuseSpark1xFp4Experts=Experts)
    capacity = 8192//world
    slabs = [torch.empty((2*capacity,WIDTH),dtype=torch.bfloat16) for _ in range(world)]
    handle = NS(get_buffer=lambda i,*args:slabs[i], barrier=lambda channel:st.events.append("barrier"))
    class Workspace: pass
    def peer_pack(addresses, slots, dest, rows):
        # Read every rank's actual selected CPU bank, including this rank's fresh copy.
        bank = workspace.next_bank
        views = [s[bank*capacity:bank*capacity+rows] for s in slabs]
        assert addresses.tolist() == [v.data_ptr() for v in views]
        dest.copy_(torch.cat(views))
    peer_env = dict(torch=torch,nn=torch.nn,threading=threading,pack_peer_rows=peer_pack,
                    MuseSpark1xFp4Experts=Experts,layernorm_sp=sp)
    wdefs=methods(stage,PEER,"PeerPromptInputs",["__init__","supports","pack"],peer_env)
    for n,fun in wdefs.items(): setattr(Workspace,n,fun)
    helper=methods(stage,PEER,None,["_local_grouped_eligible","_peer_experts_into","_capture_stub"],peer_env)
    def peer_entry(*args):
        if st.capture and st.breakable:
            st.events.append("capture stub")
            return helper["_capture_stub"](*args)
        return helper["_peer_experts_into"](*args)
    install(PACKAGE+"peer_prompt_pack",peer_experts_into=peer_entry)
    parallels = NS(tp_group=group,attn_dp_size=1)
    menv = dict(torch=torch,contextlib=contextlib,_is_cpu=False,get_forward=lambda:NS(sp_active=st.sp),
                get_parallel=lambda:parallels,layernorm_sp=sp,is_dp_attention_enabled=lambda:False,
                get_global_experts_capturer=lambda:None,try_decode_column_gather=column,
                bf16_gemm_dispatch=lambda x,w,b:x,tensor_model_parallel_all_reduce=lambda x:x)
    mdefs = methods(stage,MODEL,"MuseSpark1xMoE",["forward","_forward_tp","_forward_experts",
                  "_forward_sp_experts","_forward_sp_replicated_experts","_reduce_and_unpool"],menv)
    moe=type("TransitionMoE",(),mdefs)()
    moe.nvfp4_checkpoint=True
    moe.sp_replicate_expert_input=True
    moe.router_in_fp32=False
    moe.pre_expert_proj=NS(weight=object())
    moe.pre_expert_norm=norm
    moe.post_expert_proj=moe.finalize_ar_workspace=None
    moe.tp_size,moe.moe_tp_size,moe.top_k,moe.num_experts=world,1,TOPK,256
    moe.experts=experts
    moe.gate=lambda x:x
    moe._routing_op=lambda:route
    moe._side_stream=lambda:side

    with (torch.no_grad(),
          patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),
          patch.object(torch.Tensor,"record_stream",lambda *a:None),
          patch.object(torch.cuda,"is_available",return_value=True),
          patch.object(torch.cuda,"is_current_stream_capturing",side_effect=lambda:st.capture),
          patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
          patch.object(torch.cuda,"current_stream",return_value=main),
          patch.object(torch.cuda,"Stream",return_value=transport),
          patch.object(torch.cuda,"stream",side_effect=use_stream)):
        workspace=Workspace(slabs[0],handle,group,banks=2)
        moe._peer_prompt_inputs=workspace
        sequence=[
            ("prompt",8192),("verify",8),("prompt_veto",8192),("eager_decode",8),
            ("short_prompt",64),("decode",8),("capture_stub",8192),("prompt",8192),
            ("rank_veto",8),("verify",8),("prompt",8192),("decode",8),
        ]
        for epoch,(mode,n) in enumerate(sequence):
            st.events=[]
            st.n=n
            st.sp=mode in ("prompt","prompt_veto","short_prompt","capture_stub")
            st.capture=mode in ("verify","decode","rank_veto","capture_stub")
            st.breakable=mode=="capture_stub"
            st.peer_veto=mode=="prompt_veto"
            st.expect_early=mode in ("verify","decode")
            norm._frozen_weight=None if mode=="rank_veto" else norm.weight
            rng=torch.Generator().manual_seed(29000+epoch)
            st.ids=torch.randint(0,64,(n,TOPK),generator=rng)
            st.data=torch.randn(n,WIDTH,generator=rng).bfloat16()
            expected_inputs=norm(st.data)
            previous=workspace.next_bank
            if st.sp:
                for r in range(1,world):
                    view=slabs[r][previous*capacity:previous*capacity+n//world]
                    view.copy_(expected_inputs[r*n//world:(r+1)*n//world])
            x=st.data[:n//world] if st.sp else st.data
            forward_mode=NS(is_decode=lambda:mode in ("decode","eager_decode","rank_veto"),
                            is_target_verify=lambda:mode=="verify")
            batch=NS(forward_mode=forward_mode,muse_decode_column_gather=st.capture and not st.sp,
                     muse_ranked_pool=st.capture and not st.sp)
            before_plans=len(st.plan_ids)
            result=moe.forward(x,forward_batch=batch)
            if mode=="capture_stub":
                assert torch.count_nonzero(result)==0
            else:
                reference=arithmetic(expected_inputs,st.ids)
                if st.sp: reference=reference[:n//world]
                assert torch.equal(result.view(torch.int16),reference.view(torch.int16))
            assert workspace.next_bank==(previous^(mode=="prompt"))
            assert len(st.plan_ids)-before_plans==int(st.expect_early)
            if st.expect_early:
                assert st.events.count("plan:router")==1
                assert st.events.index("plan:router")<st.events.index("main waits router")<st.events.index("consume ranked plan")
            assert st.events.count("barrier")==int(mode=="prompt")
            assert st.events.count("peer vote")==int(mode in ("prompt","prompt_veto"))
            assert st.events.count("prompt dense fallback")==int(mode in ("prompt_veto","short_prompt"))
            st.cases.append(dict(mode=mode,rows=n,before_bank=previous,after_bank=workspace.next_bank,
                                 compared_words=result.numel(),early_plan=st.expect_early,events=st.events))
        assert len(st.plan_ids)==4
        stale_differences=[int((x!=y).sum()) for x,y in zip(st.plan_ids,st.plan_ids[1:])]
        assert all(stale_differences)
    return dict(tp=world,calls=st.cases,wrong_previous_ranked_map_words=stale_differences)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",type=Path)
    stage=p.parse_args().stage
    torch.set_num_threads(1)
    worlds=[run_world(stage,world) for world in (4,8)]
    print(json.dumps(dict(source_sha256={path:sha256((stage/path).read_bytes()).hexdigest() for path in (MODEL,HUM,PEER)},
               worlds=worlds,scope="Model forward, TP/SP dispatch, admission, rate-independent plan, publish rotation and barriers execute Python source. CUDA, collectives, remote ranks, capture callbacks, kernels and numerical expert projections are host models."),indent=2))


if __name__=="__main__":
    main()
