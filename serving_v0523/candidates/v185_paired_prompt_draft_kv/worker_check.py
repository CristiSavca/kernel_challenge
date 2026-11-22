"""Exercise the real prefill worker, KV-plan translations, norm/RoPE, physical writes, and partial-write recovery."""
import ast
import contextlib
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
repo=Path(sys.argv[1]).resolve()
pin="297e54fd105e974cdf401e5b4d20253a22c16f90"
digests={}
def load(path, owner, name, inject=None):
    rel="python/sglang/"+path
    f=repo/rel
    s=f.read_text() if f.is_file() else subprocess.check_output(["git","-C",str(repo),"show",pin+":"+rel],text=True)
    digests[rel]=hashlib.sha256(s.encode()).hexdigest()
    nodes=ast.parse(s).body
    if owner:nodes=next(x for x in nodes if getattr(x,"name",None)==owner).body
    n=next(x for x in nodes if getattr(x,"name",None)==name)
    ns=dict(torch=torch,F=F,**(inject or {}))
    exec("from __future__ import annotations\n"+ast.unparse(n),ns)
    return ns[name]

@dataclass
class Loc:
    loc:torch.Tensor
    swa_loc:torch.Tensor|None=None
    full_loc:torch.Tensor|None=None
    physical:bool=False

mem="srt/mem_cache/memory_pool.py"
rpath="srt/layers/rotary_embedding/"
dpath="srt/models/dflash.py"
wpath="srt/speculative/dflash_worker_v2.py"
fpath="kernels/ops/speculative/fused_kv_materialize.py"
spath="srt/mem_cache/kv_loc_plan.py"
apply=load(rpath+"utils.py",None,"apply_rotary_emb")

class Rope:
    __call__=load(rpath+"base.py","RotaryEmbedding","forward_native")
    _apply_rotary_emb_wrapped=staticmethod(apply)

class Norm:
    __call__=load("srt/layers/layernorm.py","RMSNorm","forward_native")

class Attention:
    kv_proj_only=load(dpath,"DFlashAttention","kv_proj_only",{"can_dflash_slice_qkv_weight":lambda w:(True,None)})
    apply_k_norm=load(dpath,"DFlashAttention","apply_k_norm")
    apply_k_rope=load(dpath,"DFlashAttention","apply_k_rope")

class Pool:
    _check_physical_write_loc=load(mem,"KVCache","_check_physical_write_loc",{"write_loc_is_physical":load(mem,None,"write_loc_is_physical",{"KVWriteLoc":Loc})})
    set_kv_buffer=load(mem,"MHATokenToKVPool","set_kv_buffer",{"unwrap_write_loc":load(mem,None,"unwrap_write_loc",{"KVWriteLoc":Loc}),
                "maybe_detect_oob":lambda x,lo,hi,name:torch._assert((x>=lo).all() and (x<hi).all(),name)})
    def _store_kv_layer(self, l, slots, k, v):
        assert torch.equal(slots,self.slots)
        self.data[l,0,slots]=k; self.data[l,1,slots]=v
        self.calls.append(l)
        if self.evict==l:
            self.evict=None
            raise RuntimeError("injected failure after a physical layer write")

class Plan:
    write_ids=load(spath,"KVLocPlan","write_ids",{"IdSpaceKind":S(FULL="full")})
    _window_in=load(spath,"KVLocPlan","_window_in")
    _cols=load(spath,"KVLocPlan","_cols",{"TokenSpan":tuple})

mode=S(is_extend=lambda:True,is_idle=lambda:False)
active=[]
@contextlib.contextmanager
def context(scope,helper):
    active.append(helper)
    try:yield
    finally:active.pop()
pospath="srt/model_executor/forward_batch_info.py"
positions=load(pospath,None,"compute_position",{"support_triton":lambda _:False,"compute_position_torch":load(pospath,None,"compute_position_torch")})
ws=dict(draft_tp_context=lambda _:contextlib.nullcontext(),logger=logging.getLogger("V185"),KVWriteLoc=Loc,
        _is_npu=False,publish_anchor=lambda **kw:kw["ctx_hidden"],ForwardMode=S(EXTEND=mode),CaptureHiddenMode=S(FULL=object()),
        dflash_sp_prefill_scope=context,nullcontext=contextlib.nullcontext,SpecTpSyncSite=S(DFLASH_TARGET="sync"),
        async_h2d=lambda m,**kw:torch.tensor(m,**kw),compute_position=positions)
class Worker:pass
for fun in ("_append_target_hidden_to_draft_kv_by_loc","_append_paired_prompt_kv","_append_target_hidden_sequential",
            "_append_target_hidden_fused","_draft_write_ids","forward_batch_generation"):
    setattr(Worker,fun,load(wpath,"DFlashWorkerV2",fun,ws))

def fused(kv, norm, eps, cos_sin, positions, heads, width, rot, k_out=None,v_out=None):
    b,l,_=kv.shape
    K,V=kv.view(b,l,2,heads,width).unbind(2)
    k=K.float()*torch.rsqrt(K.float().square().mean(-1,True)+eps[None,:,None,None])*norm[None,:,None,:].float()
    cos,sin=cos_sin[positions].float().chunk(2,-1)
    a,z=k[...,:rot//2],k[...,rot//2:rot]
    values=(torch.cat((a*cos[:,None,None,:]-z*sin[:,None,None,:],z*cos[:,None,None,:]+a*sin[:,None,None,:],k[...,rot:]),-1).to(kv.dtype),V)
    out=[]
    for dest,value in zip((k_out,v_out),values):
        src=value.transpose(0,1).contiguous()
        if dest is None:out.append(src)
        else:dest.copy_(src);out.append(dest)
    return out

class Helper:pass
for fun in ("__init__","materialize","_ensure_workspace","_ensure_rope_cache"):
    setattr(Helper,fun,load(fpath,"FusedKVMaterializeHelper",fun,{"_fused_norm_rope_stacked":fused}))

def build(sizes,require):
    L,heads,D,H=5,2,8,12
    layers=[]
    rope=Rope();rope.head_size=D;rope.rotary_dim=6;rope.is_neox_style=True
    phase=torch.outer(torch.arange(320).float(),torch.tensor([1,.23,.009]));rope.cos_sin_cache=torch.cat((phase.cos(),phase.sin()),1)
    for l,scale in enumerate(sizes):
        a=Attention();a.q_size=heads*D; a.kv_size=heads*D;a.num_kv_heads=heads;a.head_dim=D;a.rotary_emb=rope
        a.v_scale=scale;a.attn=S(layer_id=l,k_scale=None,v_scale=None)
        a.qkv_proj=S(weight=torch.randint(-3,4,(a.q_size+2*a.kv_size,H)).bfloat16(),bias=None)
        n=a.k_norm=Norm();n.variance_epsilon=1e-4*(l+1);n.weight=torch.linspace(.7,1.3,D).bfloat16()
        n.override_orig_dtype=n.variance_size_override=None;n.hidden_size=D;n.cast_x_before_out_mul=False
        layers.append(S(self_attn=a))
    class Draft:
        def project_target_hidden(self,x):
            if require:raise AssertionError("preprojected carrier reached draft FC")
            return x
        def prepare_context_hidden_for_kv(self,layer,x):return x
    draft=Draft();draft.layers=layers
    return draft,Helper(layers,rope,heads,D,torch.device("cpu"),max_position_hint=290)

def run(tag, fused_mode, remap, padded, reference=False):
    torch.manual_seed(185702)
    lengths=[7,3,12];prefix=[202,0,41];n=sum(lengths);size=n*5+3
    is_eligible=tag=="normal" and fused_mode!="off"
    raw_mode=is_eligible and not reference
    scales=[1,None,1.,None,1] if tag!="scaled" else [1,None,1.,2.,1]
    draft,helper=build(scales,raw_mode)
    hidden=torch.randint(-2,3,(n,12)).bfloat16()
    raw=hidden @ helper.flat_kv_weight_t
    pool=Pool();pool.dtype=pool.store_dtype=torch.bfloat16;pool.start_layer=0;pool.is_quantized_kv_cache=False;pool.use_hnd=False
    pool.requires_physical_write_loc=True;pool.size=size;pool.page_size=1
    pool.data=torch.full((5,2,size+1,2,8),-999.,dtype=torch.bfloat16);pool.calls=[];pool.evict=2 if fused_mode=="partial" else None
    plan=Plan();plan.write_virtual=torch.randperm(size)[1:n+1]+1;plan._write_ids={}
    translation=torch.cat((torch.zeros(1,dtype=torch.long),torch.randperm(size)+1))
    space=S(key="mapped" if remap else "native",write=(lambda x:translation[x]) if remap else None)
    reader=S(space=lambda kind:space);pool.slots=translation[plan.write_virtual] if remap else plan.write_virtual
    w=Worker();w.draft_parallel_layout=None;w.model_runner=S(device=torch.device("cpu"),prefill_attention_backend_str="cpu")
    w.draft_model_runner=S(token_to_kv_pool=pool,kv_index_translator=reader);w.draft_model=draft;w._fused_kv_helper=helper
    w._sp_prefill_context=None if tag=="disabled" else object();w._use_fused_kv_materialize=fused_mode!="off"
    w.lilicorr=object() if tag=="lilicorr" else None; w._draft_sampler=None; w._logged_padding_trim=True
    w._validate_phase1_sampling_support=lambda _:None;w._wait_for_fused_hicache_load=lambda _:None
    events=[]
    w._tp_sync=S(sync=lambda _,ids:events.append("sync"))
    w._make_next_draft_input_prefill=lambda **kw:kw
    if fused_mode=="warm":helper._ensure_workspace(n+5,torch.bfloat16)
    raw_before=raw.clone()
    prealloc=helper._proj_workspace
    actual_pos=positions("cpu",torch.tensor(prefix),torch.tensor(lengths),n)[0]
    ids=torch.tensor([11,15,29]);b=S(seq_lens=torch.tensor(prefix)+torch.tensor(lengths),out_cache_loc=plan.write_virtual,
        prefix_lens=prefix,extend_lens=lengths,return_logprob=tag=="logprob",return_hidden_states_mode=S(need_capture=lambda:tag=="external"),
        is_extend_in_batch=False,forward_mode=mode)
    def target(*args,**kw):
        events.append("target")
        assert bool(active and active[-1] is helper)==is_eligible,(tag,fused_mode,active)
        carrier=raw if raw_mode else hidden
        if padded:carrier=F.pad(carrier,(0,0,0,3),value=float('nan'))
        return S(kv_loc_plan=plan,next_token_ids=ids,logits_output=S(hidden_states=carrier,
            dflash_prefill_context_done=raw_mode or tag=="normal",dflash_prefill_kv_projected=raw_mode))
    w.target_worker=S(forward_batch_generation=target)
    with patch.object(torch.cuda,"is_current_stream_capturing",return_value=tag=="capture"), patch.object(torch,"mm",wraps=torch.mm) as mm:
        out=w.forward_batch_generation(b,on_publish=lambda _:events.append("publish"))
    assert events==["target","sync","publish"] and not active and out.kv_loc_plan is None
    assert torch.equal(out.next_draft_input["bonus_tokens"],ids) and torch.equal(out.new_seq_lens,b.seq_lens)
    assert out.logits_output.hidden_states is None and torch.equal(plan.write_virtual,b.out_cache_loc)
    if fused_mode=="partial":assert pool.calls==[0,1,2,0,1,2,3,4] and w._fused_kv_helper is None
    else:assert pool.calls==list(range(5))
    if raw_mode:
        assert helper._proj_workspace is prealloc and mm.call_count==0 and torch.equal(raw_before,raw)
        if fused_mode!="partial":assert w._use_fused_kv_materialize and w._fused_kv_helper is helper
    untouched=torch.ones(size+1,dtype=torch.bool);untouched[pool.slots]=False
    assert (pool.data[:,:,untouched]==-999).all()
    if raw_mode and fused_mode=="partial":
        assert w._use_fused_kv_materialize is False
    return pool.data,w,draft,raw,actual_pos,pool.slots

def main():
    logging.disable(logging.WARNING)
    cases=0;scalars=0;rejects=0
    for tag in ("normal","disabled","scaled","lilicorr","capture","logprob","external"):
        for mode_ in ("cold","warm","partial","off"):
            for mapped in (False,True):
                for padded in (False,True):
                    a,*_=run(tag,mode_,mapped,padded)
                    baseline,*_=run(tag,mode_,mapped,padded,reference=True)
                    assert torch.equal(a,baseline),(tag,mode_,mapped,padded)
                    cases+=1;scalars+=baseline.numel()
    _,w,draft,raw,positions,slots=run("normal","cold",True,True)
    for fault in ("unmarked","decode","lilicorr","shape","positions"):
        kw=dict(target_hidden=raw,cache_loc=slots,positions=positions,prefill_context_done=True,prefill_kv_projected=True)
        if fault=="unmarked":kw["prefill_context_done"]=False
        if fault=="decode":kw.update(cache_loc_2d=slots[None],commit_lens=torch.tensor([1]))
        if fault=="lilicorr":w.lilicorr=True
        if fault=="shape":kw["target_hidden"]=raw[:,:-1]
        if fault=="positions":kw["positions"]=positions[:-1]
        try:w._append_target_hidden_to_draft_kv_by_loc(**kw)
        except ValueError:rejects+=1
        else:raise AssertionError(fault)
        w.lilicorr=None
    return dict(paired_real_worker_cases=cases,compared_physical_KV_elements=scalars,invalid_marker_or_shape_rejections=rejects,
            ordinary_prompt_geometry=[7,3,12],layers=5,heads=2,head_dim=8,rotary_dim=6,
            methods_from_stage=digests,torch=torch.__version__,physical_GPU=False,
            surrogate="Only CUDA norm/RoPE primitive, terminal NHD scatter, model-tap producer and TP runtime emulated; pinned host worker, pool dispatch, absolute positions, physical ID translation and sequential norm/RoPE executed.")
if __name__=="__main__":print(json.dumps(main(),indent=2))
