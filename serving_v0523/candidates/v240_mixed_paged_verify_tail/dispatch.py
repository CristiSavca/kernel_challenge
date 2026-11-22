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
from kernel_replay import KERNEL, build, dense_reference


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "v239_split_mixed_full_attention"
ap = argparse.ArgumentParser()
ap.add_argument("--stage",type=Path)
ap.add_argument("--candidate",type=Path,default=HERE)
args = ap.parse_args()
parent_replay = PARENT / "replay.py"
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

def configure(fb,backend,layer,globals_,paged):
    fb.spec_algorithm=NS(is_dflash=lambda:case!="other_draft")
    backend.topk=2 if case=="tree" else 1
    backend._disable_scheduler_metadata_precompute=case=="metadata_veto"
    backend._hopper_verify_sms=0 if case=="runtime_off" else 132
    backend.speculative_num_draft_tokens=4 if case=="wrong_width" else 8
    layer.logit_cap=2.0 if case=="softcap" else 0.
    api, launches=build(f.source(KERNEL,True))

    def run(q,k,v,pages,lengths,cu,**kw):
        assert not special_outputs and kw["window_left"] < 0
        assert kw["workspace_limit_bytes"] == 32*1024**2 - sum(current_rows)*2*f.fx.H*f.fx.D
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
        special_outputs=[]
        expected=(1,1) if cert=="approved" and left<0 else (1,0) if left>=0 or cert=="ordinary" else (0,1)
        check_env.update(expected=expected,special_outputs=special_outputs)
        env.update(cert=cert,prompt_count=prompt_count)
        paired.append(env["run"](True,current_rows,prefixes,left,None,scale))
        assert bool(special_outputs)==bool(child and eligible), (case,child,eligible)
    assert paired[0]==paired[1]
    responses.append(dict(case=case,certificate=cert,rows=current_rows,window=left,specialized=eligible,
                          logical_KV_bytes=paired[1]["visible_KV_bytes"]))
print(json.dumps(dict(candidate=manifest["candidate"],tree=manifest["tree"],patch_sha256=manifest["patch_sha256"],
                  cases=responses,custom_kernel=numerical,pairs=len(cases),
                  total_tail_scalars=sum(x["output_scalars"] for x in numerical),physical_H200=False),indent=2))
