"""Replay real preparation and ranked expert dispatch with instrumented image handles."""
import ast
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import patch

import torch


PREFIX="python/sglang/srt/layers/muse_spark/"
HUM, FP4, GATE, DOWN=[PREFIX+x+".py" for x in ("muse_spark_humming_moe","muse_spark_fp4_moe","decode_gate_up","decode_mma_pair")]
PACKAGE="sglang.srt.layers.muse_spark."
PARENT="2c712d66b6211c08845570d67423807ef7d54006"
FINAL="f53b3f6f04851484810f87276c77079498ce1a9e"
WIDTH=4096


def source(stage,rev,path):
    if rev: return subprocess.check_output(["git","-C",stage,"show",rev+":"+path])
    return (stage/path).read_bytes()


def defs(stage,rev,path,names,env,cls=None):
    tree=ast.parse(source(stage,rev,path))
    nodes=tree.body if not cls else next(x.body for x in tree.body if getattr(x,"name","")==cls)
    body=[n for n in nodes if getattr(n,"name","") in names]
    assert len(body)==len(names)
    exec("from __future__ import annotations\n"+ast.unparse(ast.Module(body,type_ignores=[])),env)


def mod(name,**kwargs):
    m=ModuleType(name);vars(m).update(kwargs)
    return m


def local_plan(ids):
    local=torch.where((ids>=0)&(ids<32),ids,-1).int()
    flat=local.flatten()
    perm=torch.argsort(flat,stable=True)
    buckets=flat+1
    span=1<<(ids.numel()-1).bit_length()
    keys=(buckets[perm]*span+perm).int()
    offsets=torch.bincount(buckets,minlength=33).cumsum(0).int()
    mapped=torch.full_like(flat,-1)
    valid=flat[perm]>=0
    mapped[perm[valid]]=torch.arange(flat.numel(),dtype=torch.int32)[valid]
    return local,offsets,keys,mapped.view_as(local)


def run(stage,rev,rows,mode,input,*,replace_map=False):
    raw,ids,scores=input
    st=NS(events=[],widths=[],seen=None)
    config={"gemm_type":"grouped_contiguous","mma_type":"wgmma","use_f16_accum":False,
            "a_dtype":"bfloat16","b_dtype":"float4e2m1","c_dtype":"bfloat16","bs_dtype":"float8e4m3",
            "weight_scale_group_size":16,"weight_scale_type":"group","weight_scale_2_type":"none",
            "shape_n":WIDTH,"shape_k":WIDTH,"num_experts":32}
    mc=json.dumps(config)
    def record(phase,m,n,kind): st.widths.append([phase,m,n,kind]);st.events.append(phase)
    def tuning(name,m):
        n,k=(256,128) if name=="w13" else (128,256)
        return json.dumps(dict(block_shape=[m,n,k],warp_shape=[m,32,64],num_stages=4,
                               use_stream_k=False,use_tma_c=False,num_ctas_per_sm=1))

    class Kernel:
        @staticmethod
        def current_context(): return ("host-admission",mode)
        def __init__(self,**kw): self.c=kw
        def check_context(self):
            m,n,_=self.c["block_shape"]
            if mode=="no_down" or (n==256 and mode=="refuse_down"): raise ValueError("fixture admission")
            self.kernel_id=m*10000+n
    class Launcher:
        def prepare(self,weight,scale,maps): maps.fill_(1);return True
        def prepare_wide(self,weight,scale,maps): maps.fill_(2);return mode!="map_veto"
        def run(self,inputs,out,keys,offsets,gain,maps,handle,shared,m,sms):
            mm,n=divmod(handle,10000)
            assert mm==m
            assert int(maps[0])==(2 if n==512 else 1),"wrong gate/up descriptor for compiled image"
            record("w13",m,n,"specialized")
            first=int(offsets[0]);st.seen=inputs[first:].clone()
            out[first:].copy_(inputs[first:]);return True
    launch=Launcher()
    def compiled(ctx,meta,compute,t, prompt=False):
        m,n,_=json.loads(t)["block_shape"]
        if mode=="no_gate" or n==512 and mode=="refuse_gate": return None
        return object(),m*10000+n,128,m
    def down(**kwargs):
        m,n=divmod(int(kwargs["configs"][2]),10000)
        record("w2",m,n,"specialized")
        first=int(kwargs["expert_layout"][0])
        kwargs["outputs"][first:].copy_(kwargs["inputs"][first:])
    def ordinary(layer,*,inputs,outputs,expert_layout,tuning_config,sublayer_name,**kw):
        m,n,_=json.loads(tuning_config)["block_shape"]
        record(sublayer_name,m,n,"native")
        first=int(expert_layout[0])
        if sublayer_name=="w13":
            st.seen=inputs[first:].clone()
            outputs[first:,:WIDTH].copy_(inputs[first:])
        else: outputs[first:].copy_(inputs[first:])
    def silu(gate,active,keys,offsets,scales,span):
        st.events.append("separate activation")
        first=int(offsets[0]);active[first:].copy_(gate[first:,:WIDTH])
    def normalize(packed,gain,mapped,out,eps):
        st.events.append("ranked norm")
        x=packed.transpose(0,1).reshape(rows,WIDTH)
        norm=torch.nn.functional.rms_norm(x,(WIDTH,),gain,eps)
        token,slot=torch.where(mapped>=0)
        out[mapped[token,slot].long()]=norm[token]
    def finalize(values,scores,*,slot_row_indices,out,**kw):
        out.zero_()
        for i in range(scores.shape[1]):
            ix=slot_row_indices[:,i];live=ix>=0
            out[live]+=(values[ix[live]]*scores[live,i,None]).bfloat16()
        st.events.append("finalize")
        return out
    triton=NS(next_power_of_2=lambda x:1<<(x-1).bit_length())
    hen=dict(torch=torch,triton=triton,_MAX_DECODE_SLOTS=2048,_GROUPED_COMPUTE="{}",
             _expert_scaled_silu_module=lambda:NS(run=silu))
    defs(stage,rev,HUM,{"humming_grouped_decode_outputs","humming_decode_eligible"},hen)
    common=dict(torch=torch,logging=logging,json=json,lru_cache=lru_cache,subprocess=subprocess,
                PackageNotFoundError=PackageNotFoundError,version=lambda _:"0.1.12")
    g=common|dict(_launcher=lambda:launch,_compile=compiled)
    d=common.copy()
    defs(stage,rev,GATE,{"prepare_gate_up","run_gate_up"},g)
    defs(stage,rev,DOWN,{"_prepare_down","prepare_down","run_down"},d)
    f=dict(torch=torch,fused_post_expert_norm_weighted_sum=finalize)
    defs(stage,rev,FP4,{"forward","_forward_w4a16"},f,"MuseSpark1xFp4Experts")
    experts=type("Experts",(),{n:f[n] for n in ("forward","_forward_w4a16")})()
    experts.use_w4a16=experts.use_humming=True
    experts.num_local_experts,experts.global_num_experts=32,256
    experts.hidden_size=experts.intermediate_size_per_partition=WIDTH
    norm=experts._post_expert_norm=object()
    experts._post_expert_norm_weight=experts.w2_global_scale=torch.ones(WIDTH,dtype=torch.bfloat16)
    experts._post_expert_norm_eps,experts._post_expert_norm_weight_first=1e-6,True
    experts.w13_global_scale=torch.ones(32,dtype=torch.bfloat16,device="meta")
    experts.w13_weight=torch.empty((32,WIDTH//16,4*WIDTH),dtype=torch.int32,device="meta")
    experts.w13_weight_scale=torch.empty((32,WIDTH//16,2*WIDTH),dtype=torch.float8_e4m3fn,device="meta")
    experts.w2_weight=experts.w2_weight_scale=None
    experts.humming_decode_tunings={n:[(b,tuning(n,m)) for b,m in ((112,8),(216,16),(320,24))] for n in ("w2","w13")}
    modules=[mod("humming.kernel.humming",HummingKernel=Kernel),mod("humming.jit.runtime",KernelRuntime=object),
             mod("humming.jit.compiler",NVCCCompiler=object),mod("humming",ops=NS(launch_kernel=down)),
             mod("humming.layer",HummingMethod=NS(forward_layer=ordinary)),mod(PACKAGE+"decode_gate_up",run_gate_up=g["run_gate_up"]),
             mod(PACKAGE+"decode_mma_pair",run_down=d["run_down"]),mod(PACKAGE+"muse_spark_humming_moe",
              **{k:hen[k] for k in ("humming_grouped_decode_outputs","humming_decode_eligible")},prepare_ranked_pool=None,
              humming_expert_outputs=None,local_humming_expert_ids=None)]
    with patch.dict(sys.modules,{m.__name__:m for m in modules}),patch.object(torch.cuda,"get_device_properties",return_value=NS(multi_processor_count=132)):
        experts.humming_decode_down=d["prepare_down"](mc,"{}",experts.humming_decode_tunings["w2"])
        experts.humming_decode_gate_up=g["prepare_gate_up"](experts,mc,"{}",experts.humming_decode_tunings["w13"])
        if replace_map:
            _,maps,_,kernels,_=experts.humming_decode_gate_up
            altered=0
            for key,entry in kernels.items():
                if entry is not None and len(entry)==5:
                    kernels[key]=(*entry[:4],maps)
                    altered+=1
            assert altered>0
        if mode=="stale_binding" and experts.humming_decode_gate_up:
            launch_obj,maps,binding,kernels,sms=experts.humming_decode_gate_up
            experts.humming_decode_gate_up=(launch_obj,maps,((1,()),(2,())),kernels,sms)
        packed=raw.view(rows,8,WIDTH//8).transpose(0,1).contiguous()
        plan=local_plan(ids)
        gain=torch.linspace(.625,1.375,WIDTH).bfloat16()
        output=experts.forward(packed,scores,ids,norm,pre_expert_norm=object(),grouped_decode=True,ranked_pool=True,
                               ranked_work=((NS(run=normalize),packed,gain,1e-6),plan))
        assert st.events.count("ranked norm")==1 and st.events.count("finalize")==1
    return st,output


def main():
    stage=Path(sys.argv[1]).resolve()
    assert all(source(stage,None,f)==source(stage,FINAL,f) for f in (FP4,HUM,GATE,DOWN))
    torch.set_num_threads(1)
    logging.disable(logging.WARNING)
    results=[]
    map_control=None
    skewed=[]
    for rows in (8,32,112,113,128,216,217,256):
        gen=torch.Generator().manual_seed(297+rows)
        args=(torch.randn(rows,WIDTH,generator=gen).bfloat16(),torch.randint(-3,256,(rows,8),generator=gen),
              torch.rand(rows,8,generator=gen))
        if rows==128:
            try:
                run(stage,None,rows,"ready",args,replace_map=True)
            except AssertionError as exc:
                assert str(exc)=="wrong gate/up descriptor for compiled image"
                map_control=str(exc)
            else:
                raise AssertionError("width-512 image accepted the width-256 map")
        for mode in ("ready","refuse_gate","refuse_down","map_veto","no_gate","no_down","stale_binding"):
            old,a=run(stage,PARENT,rows,mode,args)
            new,b=run(stage,None,rows,mode,args)
            assert torch.equal(a.view(torch.int16),b.view(torch.int16)) and torch.equal(old.seen,new.seen)
            width=8 if rows<=112 else 16 if rows<=216 else 24
            for state,wide in ((old,False),(new,width!=8)):
                gate_native=mode in ("no_gate","stale_binding")
                down_native=mode=="no_down"
                gate_n=512 if wide and not gate_native and mode not in ("refuse_gate","map_veto") else 256
                down_n=256 if wide and not down_native and mode!="refuse_down" else 128
                assert state.widths==[["w13",width,gate_n,"native" if gate_native else "specialized"],
                                     ["w2",width,down_n,"native" if down_native else "specialized"]],(rows,mode,state.widths)
                assert state.events.count("separate activation")==int(gate_native)
            results.append(dict(rows=rows,case=mode,called=new.widths,events=new.events,
                                normalized_live_words=new.seen.numel(),output_words=b.numel()))
        if rows in (128,256):
            for style,changed_ids in (("all remote",torch.full_like(args[1],200)),
                                      ("all local and duplicate",torch.zeros_like(args[1]))):
                skew=(args[0],changed_ids,args[2])
                old,a=run(stage,PARENT,rows,"ready",skew)
                new,b=run(stage,None,rows,"ready",skew)
                assert torch.equal(a,b) and torch.equal(old.seen,new.seen)
                assert new.seen.numel()==(0 if style=="all remote" else rows*8*WIDTH)
                if style=="all remote":assert not torch.count_nonzero(b)
                skewed.append(dict(rows=rows,distribution=style,canonical_bins_and_images=new.widths,
                                   actual_local_routes=new.seen.shape[0],output_words=b.numel()))
    print(json.dumps(dict(parent_tree=PARENT,candidate_tree=FINAL,source_sha256={f:sha256(source(stage,None,f)).hexdigest() for f in (FP4,HUM,GATE,DOWN)},
        calls=len(results),wrong_descriptor_control=map_control,normalized_live_words=sum(x["normalized_live_words"] for x in results),
        output_words=sum(x["output_words"] for x in results),cases=results,skew_controls=skewed,
        scope="Actual prepare_gate_up, prepare_down/_prepare_down, expert.forward/W4A16, ranked grouped adapter and both runtime callbacks. Synthetic image admission and math; descriptor sentinel validates wide/narrow map selection. The stored plan represents completed routing on the independent stream."),indent=2))


if __name__=="__main__":main()
