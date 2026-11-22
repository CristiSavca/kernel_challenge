"""Compare actual V239 and candidate Breakable callbacks, RMW and native/paged dispatch."""
import argparse
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
from unittest.mock import patch

import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"v240_mixed_paged_verify_tail"))
from kernel_replay import KERNEL, build as build_base, dense_reference


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "v241_split_serving_decode"
ap = argparse.ArgumentParser()
ap.add_argument("--stage",type=Path)
ap.add_argument("--candidate",type=Path,default=HERE)
args = ap.parse_args()
parent_replay = HERE.parent / "v239_split_mixed_full_attention/replay.py"
# Import the parent's checked functions, before its top-level experiment starts.
module = ast.parse(parent_replay.read_text())
stop = next(n.lineno for n in module.body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id",None) == "env" and n.lineno > 100)
parent = {"__file__":str(parent_replay)}
with patch.object(sys,"argv",[str(parent_replay)]):
    exec(compile(ast.Module([n for n in module.body if n.lineno < stop], []), str(parent_replay), "exec"), parent)
f = parent["f"]

def replace_once(s, before, after):
    assert s.count(before) == 1, before
    return s.replace(before,after)

# Keep full normal model / graph / cache-store entry. Instrument only the native
# CUDA launches and initialize policy fields omitted from the portable mock.
src = inspect.getsource(f.run)
end = 'assert len(stores)==1 and len(dense)+len(paged)==1'
src=src[:src.rindex(end)]+'return check_attention(locals())\n'
src=replace_once(src,'dense.clear(); paged.clear(); stores.clear()',
                 'dense.clear(); paged.clear(); stores.clear()\n        role(fb,prompt_count,cert,1)\n'
                 '        configure(fb, backend, layer, attention_globals, paged)')
src=replace_once(src,'return result.fill_(sentinel)','return result.fill_(-sentinel)')
# Provide the newly introduced helper in the same namespace as the real
# backend. Source extraction leaves imports to the test's CUDA stand-ins.
src=replace_once(src,'names=[*names, "prepare_batched_prompt_kv"]',
    'names=[*names, "prepare_batched_prompt_kv"]\n'
    '            if "def allocate_mixed_prompt_storage(" in source(path, child):\n'
    '                names.append("allocate_mixed_prompt_storage")')
src=replace_once(src,'observed=[]',
    'observe_workspace_helpers(modules, attention_globals)\n    observed=[]')
# A legacy provider may accept an out keyword but return different storage.
src=replace_once(src,'result=original_paged(**kw)\n        result.fill_',
    'result=original_paged(**kw)\n        if legacy(): result=torch.empty_like(result)\n        result.fill_')
src=replace_once(src,'return result.fill_(-sentinel)',
    'if legacy(): result=torch.empty_like(result)\n        return result.fill_(-sentinel)')
check_src=inspect.getsource(parent["check_attention"])
check_src=replace_once(check_src,'assert (s["out"][cut:cut+n] == label).all()',
                       'if is_dense or not special_outputs:\n            assert (s["out"][cut:cut+n] == label).all()\n'
                       '        else:\n            assert torch.equal(s["out"][cut:cut+n], special_outputs[0].view_as(s["out"][cut:cut+n]))')
env=dict(vars(f), role=parent["role"])
check_env=dict(parent)
exec(check_src,check_env)
env["check_attention"]=check_env["check_attention"]
exec(src,env)
torch.set_num_threads(1)
def build(source):
    host,launches=build_base(source)
    nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    if "verify_partition_workspace" in nodes:
        exec(compile(ast.Module([nodes["verify_partition_workspace"]],[]),KERNEL,"exec"),host.__globals__)
    return host,launches

def observe_workspace_helpers(modules,backend):
    if "allocate_mixed_prompt_storage" not in backend:
        return
    from types import ModuleType
    sub=ModuleType("paged_verify_attention")
    host,_=build(f.source(KERNEL,True))
    sub.verify_partition_workspace=host.__globals__["verify_partition_workspace"]
    modules["sglang.kernels.ops.attention.paged_verify_attention"]=sub
    alloc=backend["allocate_mixed_prompt_storage"]
    def record(*a):
        storage=alloc(*a)
        active["storage"]=storage
        return storage
    backend["allocate_mixed_prompt_storage"]=record

# The prior pack emulation copied all new rows. The new path has those
# bytes already at their final dense positions and writes only halo + guard.
previous=f.fx.KernelBoundary.__getitem__
def select_pack(self,grid):
    parent_call=previous(self,grid)
    def invoke(ck,cv,nk,nv,table,lengths,cuq,starts,used,ok,ov,kp,vp,tp,*,ROW,BLOCK,GUARD,INLINE_NEW=False):
        if not INLINE_NEW:
            return parent_call(ck,cv,nk,nv,table,lengths,cuq,starts,used,ok,ov,kp,vp,tp,
                               ROW=ROW,BLOCK=BLOCK,GUARD=GUARD)
        assert grid[1]==1 and len(used)==1 and active.get("storage") is not None
        live=int(used[0]);n=int(cuq[1]);halo=live-n
        assert starts.tolist()==[0,live+GUARD] and len(nk)==len(nv)==n
        assert grid[0]==min((halo+GUARD+BLOCK-1)//BLOCK,1024)
        for bank,new,out in ((ck,nk,ok),(cv,nv,ov)):
            assert new.data_ptr()==out[halo:].data_ptr()
            assert out.untyped_storage().data_ptr()==active["storage"].untyped_storage().data_ptr()
            written=bank[:,0][table[0,halo:live].long()]
            assert torch.equal(new,written)
            prefix=bank[:,0][table[0,:halo].long()]
            out[:halo].copy_(prefix);out[live:live+GUARD].zero_()
        active["inline_calls"]+=1
        active["resident_rows"]=len(ok)
    return invoke
f.fx.KernelBoundary.__getitem__=select_pack
env.update(observe_workspace_helpers=observe_workspace_helpers,legacy=lambda:case=="legacy" and active.get("replay"))


def configure(fb,backend,layer,globals_,paged):
    active["replay"]=True
    fb.spec_algorithm=NS(is_dflash=lambda:case!="other_draft")
    backend.topk=2 if case=="tree" else 1
    backend._disable_scheduler_metadata_precompute=case=="metadata_veto"
    backend._hopper_verify_sms=0 if case=="runtime_off" else 132
    backend.speculative_num_draft_tokens=4 if case=="wrong_width" else 8
    layer.logit_cap=2.0 if case in ("softcap","legacy") else 0.
    api, launches=build(f.source(KERNEL,True))

    def run(q,k,v,pages,lengths,cu,**kw):
        assert not special_outputs and kw["window_left"] < 0
        charged=(sum(current_rows)*2*f.fx.H*f.fx.D if active.get("storage") is None else active["storage"].nbytes)
        assert kw["workspace_limit_bytes"] == 32*1024**2 - charged
        assert kw.get("origins") is None
        result=api(q,k,v,pages,lengths,cu,**kw)
        if result is None:
            assert not launches
            return None
        assert result.data_ptr()==kw["out"].data_ptr() and [k for k,_ in launches]==["_partition_attention","_merge_partitions"]
        assert q.shape[0]==sum(current_rows[prompt_count:]) and cu[0].item()==0
        expected=dense_reference(q,k,v,pages,lengths,cu,kw["k_descale"],kw["v_descale"],kw["softmax_scale"])
        relative=float((result.double()-expected).norm()/expected.norm())
        assert relative < .008, relative
        special_outputs.append(result.clone())
        numerical.append({"tail_rows": q.shape[0], "output_scalars":result.numel(), "relative_l2":relative,
                          "workspace_limit":kw["workspace_limit_bytes"],"launches":launches})
        # Reuse V239's named operand checks to compare physical slots, raw
        # FP8 bytes and independent sampled reference results across both roles.
        native=dict(q=q,k_cache=k,v_cache=v,page_table=pages,cache_seqlens=lengths,cu_seqlens_q=cu,
                    max_seqlen_q=kw["max_q"],softmax_scale=kw["softmax_scale"],window_size=(-1,-1),
                    out=result,causal=True,softcap=0.)
        for key in ("k_descale","v_descale"):
            if kw[key] is not None:
                native[key]=kw[key]
        paged.append(native)
        return result

    globals_["paged_verify_attention"]=run

env["configure"]=configure
cases = [
    ([4096,8,8],[0,7,131],1,-1,"approved",False,"enabled",True),
    ([4096,8],[113000,55],1,-1,"approved",True,"budget",True),
    ([4096,8,8],[83,33,53],1,-1,"approved",True,"legacy",False),
    ([4096,8],[53,75],1,-1,"approved",True,"enabled",True),
    ([3210,1111,8,8,8],[80,90,0,131,256],2,-1,"approved",False,"enabled",True),
    ([4096]+[8]*15,[64]+list(range(9,24)),1,-1,"approved",True,"enabled",True),
    ([4096]+[8]*31,[64]+list(range(16,47)),1,-1,"approved",False,"enabled",True),
    ([4096,8,7],[53,75,33],1,-1,"approved",True,"ragged",False),
]
for case in ("tree","metadata_veto","runtime_off","other_draft","wrong_width","softcap"):
    cases.append(([4096,8,8],[12,11,10],1,-1,"approved",True,case,False))
for cert in ("masked","uncertified","wrong_count","ordinary"):
    cases.append(([4096,8,8],[12,11,10],1,-1,cert,False,"enabled",False))
cases.append(([4096,8,8],[12,11,10],1,2047,"approved",True,"enabled",False))
cases.append(([4096,8,8],[126900,11,10],1,-1,"oversize",True,"enabled",False))
responses=[]
numerical=[]
for current_rows,prefixes,prompt_count,left,cert,scale,case,eligible in cases:
    paired=[]
    for child,folder in enumerate((PARENT,args.candidate)):
        manifest=json.loads((folder/"validation.json").read_text())
        f.reader.CHILD,_=f.checked_patch(folder,manifest)
        f.STAGE=f.reader.STAGE=args.stage if child else None
        f.source.cache_clear()
        active={"inline_calls":0,"storage":None,"replay":False}
        special_outputs=[]
        expected=(1,1) if cert=="approved" and left<0 else (1,0) if left>=0 or cert=="ordinary" else (0,1)
        check_env.update(expected=expected,special_outputs=special_outputs)
        env.update(cert=cert,prompt_count=prompt_count)
        paired.append(env["run"](True,current_rows,prefixes,left,None,scale))
        assert bool(special_outputs)==bool(eligible), (case,child,eligible)
    assert paired[0]==paired[1]
    inline=bool(prompt_count==1 and left<0 and cert=="approved" and case!="budget")
    assert active["inline_calls"]==int(inline),(case,cert,active["inline_calls"])
    responses.append(dict(inline_storage=inline,case=case,certificate=cert,rows=current_rows,window=left,specialized=eligible,
                          logical_KV_bytes=paired[1]["visible_KV_bytes"]))
print(json.dumps(dict(candidate=manifest["candidate"],tree=manifest["tree"],patch_sha256=manifest["patch_sha256"],
                  cases=responses,custom_kernel=numerical,pairs=len(cases),
                  candidate_tail_scalars=sum(x["output_scalars"] for x in numerical[1::2]), inline_cases=sum(x["inline_storage"] for x in responses),physical_H200=False),indent=2))
