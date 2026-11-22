"""Run the real mixed _get_logits producer branch on both sides of its shard/cap contract."""
import ast
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as N

path = Path(sys.argv[1]) / "python/sglang/srt/layers/logits_processor.py"
source = path.read_bytes()
cls = next(n for n in ast.parse(source).body if getattr(n,"name","")=="LogitsProcessor")
names = {"_get_logits","is_deferred_shard"}
module = ast.parse("from __future__ import annotations")
module.body.extend(n for n in cls.body if getattr(n,"name","") in names)
tiny, maximum = 2.0**-126, 2.0**128-2.0**104
world = N(world_size=None)
scope = dict(torch=N(float32="fp32",finfo=lambda _:N(tiny=tiny,max=maximum)),
             get_parallel=lambda:N(tp_group=world),_trace_e2e_logits=lambda *a,**kw:None,
             envs=N(SGLANG_TRACE_LOGITS_E2E_SYNC=N(get=lambda:False)),
             LoRABatchLayout=N(TP_GLOBAL="global",DP_LOCAL="local"),
             get_forward=lambda:N(scoped=lambda **kw:nullcontext()))
exec(compile(module,str(path),"exec"),scope)


def check():
    valid=declined=legacy=0
    for tp in (2, 3, 4, 8):
        world.world_size=tp
        for width,vocab in ((1,1),(32,31),(32,32),(32,33),(33,32),(8,15),(7,17),
                            (256,513),(0,39),((2**24-1)//tp,2**24-1),(2**24//tp,2**24//tp+1)):
            for cap in (-math.inf,-2.,0.,tiny/2,tiny,1.,30.,maximum,maximum*2,math.inf,math.nan):
                for mixed,mark in ((True,True),(True,False),(False,True),(False,False)):
                    local = N(shape=(6,width))
                    dense=object()
                    q=N(shape=(6,12))
                    proc=N(do_tensor_parallel_all_gather=True,do_tensor_parallel_all_gather_dp_attn=False,
                           vocab_size=vocab,final_logit_softcapping=cap,defer_verify_gather=True,logit_scale=None,
                           _gather_dp_attn_hidden_states=lambda *args:(q,q),_compute_lm_head=lambda *a,**k:local,
                           _can_gather_softcap_fused=lambda *args:True,_gather_softcap_fused=lambda *args:dense)
                    meta=N(mixed_verify_sharded_logits=mixed,mixed_verify_logits_indices=object() if mark else None,
                           forward_mode=N(is_target_verify=lambda:False))
                    got=scope["_get_logits"](proc,q,None,meta)
                    correct=mixed and mark and 0<width<vocab<=width*tp<2**24 and tiny<=cap<=maximum
                    assert (got is local) == correct and (got is dense) != correct
                    if correct:
                        assert scope["is_deferred_shard"](proc,got)
                        valid+=1
                    else:declined+=1
            # The existing TARGET_VERIFY return remains untouched.
            meta.forward_mode=N(is_target_verify=lambda:True)
            assert scope["_get_logits"](proc,q,None,meta) is local
            legacy+=1
    return dict(source_sha256=hashlib.sha256(source).hexdigest(),mixed_shards=valid,full_fallback=declined,
                native_verify_no_change=legacy,real_h200=False)


if __name__=="__main__":
    print(json.dumps(check(),indent=2))
