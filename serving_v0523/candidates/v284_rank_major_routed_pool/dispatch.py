"""Run staged pooler, public W4A16 entry and grouped W13 staging on host tensors."""
import argparse
import ast
import contextlib
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch, PropertyMock

import torch

BASE = "python/sglang/srt/"
MODEL = BASE + "models/muse_spark.py"
HUMM = BASE + "layers/muse_spark/muse_spark_humming_moe.py"
FP4 = BASE + "layers/muse_spark/muse_spark_fp4_moe.py"
GROUP = BASE + "layers/communication/muse_spark_gemm_ag.py"
MODULE = "sglang.srt.layers.muse_spark."
H = 4096


def select(stage, path, owner, names, env):
    tree = ast.parse((stage / path).read_text())
    body = tree.body if owner is None else next(n.body for n in tree.body if getattr(n, "name", None) == owner)
    for name in names:
        node = next(n for n in body if getattr(n, "name", None) == name)
        exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return {n: env[n] for n in names}


def install(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


class Kernel:
    def __init__(self, fn):
        self.fn = fn
    def __getitem__(self, grid):
        return self.fn


def main():
    args = argparse.ArgumentParser()
    args.add_argument("stage", type=Path)
    stage = args.parse_args().stage
    st = NS()
    torch.set_num_threads(1)
    triton = NS(next_power_of_2=lambda x: 1 << (x-1).bit_length())
    def plan(ids, expert_map, local, offsets, keys, row_map, slots, **kwargs):
        positions = torch.arange(slots)
        local_flat = ids.reshape(-1)
        valid = (local_flat >= 0) & (local_flat < 32)
        buckets = torch.where(valid, local_flat+1, 0)
        offsets.copy_(torch.bincount(buckets, minlength=33).cumsum(0))
        sorted_slots = torch.argsort(buckets, stable=True)
        keys.copy_(buckets[sorted_slots]*kwargs["SLOT_SPAN"] + sorted_slots)
        dest = torch.full((slots,), -1, dtype=torch.int32)
        dest[sorted_slots] = torch.where(valid[sorted_slots],positions,-1).to(torch.int32)
        row_map.copy_(dest.reshape_as(row_map))
        local.copy_(torch.where(valid,local_flat,-1).reshape_as(local))
        st.events.append("plan")
    def gather(x, keys, offsets, out, slots, *, TOPK, SLOT_SPAN, **kwargs):
        first = int(offsets[0])
        out[first:].copy_(x[(keys[first:] % SLOT_SPAN)//TOPK])
        st.events.append("gather")
    def gate(layer, inputs, keys, offsets, tuning):
        st.first = int(offsets[0]); st.staged = inputs[st.first:].clone()
        st.events.append("W13")
        return inputs
    def finalize(values, scores, *, slot_row_indices, slot_expert_ids, out, **kwargs):
        st.maps = slot_row_indices.clone()
        out.zero_()
        for k in range(scores.shape[1]):
            idx=slot_row_indices[:,k]
            live=idx>=0
            out[live] = out[live] + values[idx[live]]
        return out
    def ranked_run(packed, gain, row_map, staging, eps):
        st.events.append("ranked_norm_stage")
        # Independent scalar-address pack interpretation; real CUDA math tested separately.
        x=torch.cat([packed[r] for r in range(packed.shape[0])], dim=-1)
        normalized=torch.nn.functional.rms_norm(x, (H,), gain, eps).bfloat16()
        for k in range(row_map.shape[1]):
            row=row_map[:,k]
            live=row>=0
            staging[row[live]] = normalized[live]

    module=NS(run=ranked_run)
    env=dict(torch=torch, triton=triton, _MAX_DECODE_SLOTS=2048, _GROUPED_COMPUTE="fixture",
             _plan_grouped_decode=Kernel(plan), _gather_grouped_decode=Kernel(gather),
             _ranked_pool_module=lambda: None if st.case=="jit_failure" else module)
    prod=select(stage,HUMM,None,["humming_decode_eligible","prepare_ranked_pool","humming_grouped_decode_outputs"],env)
    install(MODULE+"muse_spark_humming_moe", **prod, local_humming_expert_ids=None,humming_expert_outputs=None)
    install("humming.layer", HummingMethod=NS())
    install(MODULE+"decode_gate_up", run_gate_up=gate)
    install(MODULE+"decode_mma_pair", run_down=lambda *a: st.events.append("W2"))
    class NativeNorm:
        zero_centered_gamma=True
        hidden_size=H
        eps=1e-6
        def __call__(self,x):
            st.events.append("ordinary_norm")
            assert x.ndim==2 and torch.equal(x, st.rows)
            g=self._frozen_weight if self._frozen_weight is not None else self.weight.float().add(1).bfloat16()
            return torch.nn.functional.rms_norm(x, (H,), g, self.eps).bfloat16()

    install(MODULE+"muse_spark_layers", MuseSpark1xRMSNorm=NativeNorm)
    native=NativeNorm()
    class Group:
        rank_in_group=0
        def all_gather(self, local, dim):
            st.events.append("gather_columns"+str(dim))
            assert torch.equal(local, st.rows[:,:H//self.world_size])
            result=st.rows.view(st.rows.shape[0], self.world_size, H//self.world_size)
            return result.transpose(0,1).contiguous().view(-1,H//self.world_size) if dim==0 else st.rows
    group=Group()
    def gemm(x,w,_):
        return st.rows[:,:w.shape[0]]
    install("sglang.srt.layers.quantization.unquant", bf16_gemm_dispatch=gemm)
    column=select(stage,GROUP,None,["try_decode_column_gather"],{})["try_decode_column_gather"]
    def router_op(*args,**kwargs):
        st.events.append("route")
        return torch.ones_like(st.ids).float(), st.ids, None
    fenv=dict(torch=torch, fused_post_expert_norm_weighted_sum=finalize)
    funcs=select(stage,FP4,"MuseSpark1xFp4Experts",["forward","_forward_w4a16"],fenv)
    experts=type("Experts",(),dict(funcs,__call__=funcs["forward"]))()
    experts.use_w4a16=experts.use_humming=True
    experts.post_expert_norm=experts._post_expert_norm=object()
    experts._post_expert_norm_weight=torch.empty(1)
    experts.w2_global_scale=experts.humming_decode_gate_up=object()
    experts._post_expert_norm_eps=1e-6
    experts._post_expert_norm_weight_first=True
    experts._expert_map=None
    experts.num_local_experts=32; experts.global_num_experts=256
    experts.intermediate_size_per_partition=H
    experts.humming_decode_tunings={n:[(1<<25,n)] for n in ("w13","w2")}
    experts.humming_decode_down={"w2":object()}
    env=dict(torch=torch, contextlib=contextlib, _is_cpu=False,
             get_forward=lambda:NS(sp_active=False),get_parallel=lambda:NS(tp_group=NS(ca_comm=None)),
             get_global_experts_capturer=lambda:None,try_decode_column_gather=column,bf16_gemm_dispatch=gemm)
    method=select(stage,MODEL,"MuseSpark1xMoE",["_forward_experts"],env)["_forward_experts"]
    moe=NS(nvfp4_checkpoint=True,router_in_fp32=False,sp_replicate_expert_input=False,
           gate=lambda x:x[:,:1], _routing_op=lambda:router_op,top_k=8,
           pre_expert_norm=native,pre_expert_proj=NS(weight=torch.empty(H,8192,device="meta")),experts=experts)
    modes=[]
    with patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),          patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),          patch.object(torch.cuda,"is_available",return_value=False):
        for tp in (4,8):
            group.world_size=moe.tp_size=tp
            for case,rows in [("mixed",8),("duplicates",32),("all_remote",17),("all_local",8),("mixed",128),
                              ("mixed",256),("missing_gain",8),("wrong_gain",8),("misaligned",8),
                              ("strided_ids",8),("getter_override",8),("jit_failure",8)]:
                st.case=case
                rng=torch.Generator().manual_seed(291+rows)
                st.rows=torch.randn(rows,H,generator=rng).bfloat16()
                native.weight=torch.randn(H,generator=rng).bfloat16()
                native._frozen_weight=native.weight+1
                if case=="wrong_gain": native._frozen_weight=native._frozen_weight.float()
                if case=="missing_gain": native._frozen_weight=None
                if case=="misaligned":
                    tmp=torch.empty(H+1,dtype=torch.bfloat16)
                    tmp[1:].copy_(native._frozen_weight)
                    native._frozen_weight=tmp[1:]
                if case=="getter_override": native._get_effective_weight=lambda: native._frozen_weight
                st.ids=torch.randint(0,256,(rows,8),generator=rng)
                if case=="duplicates": st.ids[:,::2]=1
                if case=="all_remote": st.ids.fill_(100)
                if case=="all_local": st.ids.remainder_(32)
                if case=="strided_ids":
                    tmp=torch.empty((rows,16),dtype=st.ids.dtype)
                    tmp[:,::2]=st.ids
                    st.ids=tmp[:,::2]
                    # Native ungrouped fallback is outside this Humming fixture.
                    assert prod["prepare_ranked_pool"](experts,st.rows.view(tp,rows,H//tp),st.ids,native) is None
                    modes.append(dict(tp=tp,case=case,rows=rows)); continue
                x=torch.empty((rows,8192),dtype=torch.bfloat16)
                results=[]
                for enabled in (False,True):
                    st.events=[]
                    out=method(moe,x,grouped_decode=True,column_group=group,ranked_pool=enabled)
                    results.append((st.staged.clone(),st.maps.clone(),out.clone(),st.events.copy()))
                old,new=results
                assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(old[:3],new[:3]))
                fused=case in ("mixed","duplicates","all_local","all_remote")
                assert new[3]==(["gather_columns0","route","plan","ranked_norm_stage","W13","W2"] if fused else
                                 ["gather_columns0","route","ordinary_norm","plan","gather","W13","W2"]), new[3]
                assert old[3]==["gather_columns-1","route","ordinary_norm","plan","gather","W13","W2"]
                modes.append(dict(tp=tp,case=case,rows=rows,staged_words=st.staged.numel(),output_words=out.numel(),
                                  map_words=st.maps.numel(),new=new[3]))
                vars(native).pop("_get_effective_weight",None)
    legacy_calls = []
    legacy_env = dict(torch=torch, is_dp_attention_enabled=lambda: False)
    legacy_methods = select(stage, MODEL, "MuseSpark1xMoE", ["forward", "_forward_tp"], legacy_env)
    class Legacy:
        finalize_ar_workspace = None
        def _forward_experts(self, hidden, router, *, ar_workspace, grouped_decode, column_group):
            legacy_calls.append("experts")
            return hidden
        def _reduce_and_unpool(self, output, *, already_reduced, column_group):
            legacy_calls.append("unpool")
            return output
    obj = Legacy()
    t = torch.empty(1)
    assert legacy_methods["_forward_tp"](obj, t) is t
    obj._forward_tp = lambda hidden, router, batch: hidden
    assert legacy_methods["forward"](obj, t) is t
    old_fp4 = NS(use_w4a16=True, _forward_w4a16=lambda hidden, scores, ids, post, *, pre_expert_norm, grouped_decode: hidden)
    assert funcs["forward"](old_fp4, t, None, None, None) is t
    assert legacy_calls == ["experts", "unpool"]
    odd = torch.empty((8, 3, 768), dtype=torch.bfloat16)
    old_width = native.hidden_size
    native.hidden_size = 8 * 768
    with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True):
        assert prod["prepare_ranked_pool"](experts, odd, torch.zeros((3, 8), dtype=torch.int64), native) is None
    native.hidden_size = old_width
    print(json.dumps(dict(method="Staged model, expert.forward, _forward_w4a16, production grouped decode and producer; CPU projection, route-plan, W13/W2 and normalization spies",
                          hashes={p:sha256((stage/p).read_bytes()).hexdigest() for p in (MODEL,HUMM,FP4,GROUP)},
                          non_power_of_two_stripe_declined=True, legacy_optional_keyword_calls=4, cases=modes,totals={k:sum(m.get(k,0) for m in modes) for k in ("staged_words","output_words","map_words")}),indent=2))
if __name__=="__main__":
    main()
