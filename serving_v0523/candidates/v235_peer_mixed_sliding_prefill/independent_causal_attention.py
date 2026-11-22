"""Replay V233 and V235 through the same mixed and subsequent ordinary attention graph."""
import argparse
import ctypes
from copy import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch


DIR=Path(__file__).resolve().parent
ROOT=DIR.parent
args=argparse.ArgumentParser(description=__doc__)
args.add_argument("--candidate",type=Path,default=DIR)
args.add_argument("--stage",type=Path)
args=args.parse_args()
PARENT=ROOT/"v233_peer_prompt_overflow"
CHILD=args.candidate.resolve()
helper=ROOT/"v231_prompt_query_encoding/independent_reach.py"
spec=importlib.util.spec_from_file_location("causal_graph_fixture",helper)
f=importlib.util.module_from_spec(spec)
argv=[str(helper),"--candidate",str(CHILD)]
if args.stage:
    argv.extend(["--stage",str(args.stage)])
with patch.object(sys,"argv",argv):
    spec.loader.exec_module(f)
H,D,QH=f.fx.H,f.fx.D,f.fx.QH
FP8=torch.float8_e4m3fn


def digest(tensor):
    raw = tensor.contiguous().view(torch.uint8)
    assert raw.device.type == "cpu"
    return hashlib.sha256(ctypes.string_at(raw.data_ptr(), raw.numel())).hexdigest()


def set_role(fb,state):
    rows=fb.extend_seq_lens_cpu
    live=sum(rows)
    role=copy(fb)
    role.spec_info=None
    role.capture_hidden_mode="full"
    role.return_logprob=role.is_prefill_only=False
    role.tbo_split_seq_index=role.token_indices_to_pool=role.multi_item_delimiter_indices=None
    role.contains_mm_inputs=lambda: False
    role.input_ids=torch.zeros(live,dtype=torch.long)
    role.positions=role.positions[:live]
    role.out_cache_loc=role.out_cache_loc[:live]
    env={"torch":torch,"CaptureHiddenMode":NS(FULL="full"),"ForwardMode":f.harness.Mode}
    real_rows=f.define(f.fx.SR+"model_executor/forward_batch_info.py","set_mixed_verify_rows",env,True,"ForwardBatch")
    real_rows(role,1)
    assert torch.equal(role.mixed_verify_logits_indices,torch.cat((torch.tensor([rows[0]-1]),torch.arange(rows[0],live))))
    role.mixed_verify_is_causal=False
    # Exercise the production tag assignment. Its cohort agreement is the integration boundary.
    class Consumed(Exception):
        pass
    def consume(*_):
        raise Consumed()
    plan=object()
    env.update(_prepare_views=lambda *a,**kw:(plan,(1,2,3,4,5,6,7),None,role),
               agree_mixed_dflash_target=lambda current,**kw:None if state["guard"]=="nonlinear" else current)
    enter=f.define(f.fx.SR+"speculative/dflash_mixed_forward.py","try_forward_proposed_mixed",env,True)
    try:
        result=enter(NS(consume_decode_proposal=consume),None,None,None)
        assert result is None and state["guard"]=="nonlinear"
    except Consumed:
        assert state["guard"]!="nonlinear"
    fb.mixed_verify_logits_indices=role.mixed_verify_logits_indices
    fb.mixed_verify_is_causal=role.mixed_verify_is_causal and state["guard"]!="uncertified"
    fb.spec_info=NS(custom_mask=torch.tensor([False,True]),topk=2) if state["guard"]=="masked" else None
    state["minted"]=role.mixed_verify_is_causal


def witness(call, dense, rows, prefixes, left, scale):
    """Consume exactly each request's real, visible keys with FA3's bottom-right causal mask."""
    kw=call
    assert kw["causal"] and kw["q"].dtype==FP8 and kw["out"].dtype==torch.bfloat16
    assert tuple(kw["window_size"])==((-1,-1) if left<0 else (left,0))
    q_start=0
    data=[]; vals=[]
    for rank,(p,n) in enumerate(zip(prefixes,rows)):
        cut=0 if left<0 else max(0,p-left)
        if dense:
            begin,end=kw["cu_seqlens_k"][rank:rank+2].tolist()
            assert end-begin==p+n-cut+256 and int(kw["seqused_k"][rank])==p+n-cut
            kv=[kw[key][begin:begin+p+n-cut] for key in ("k","v")]
            for key in ("k","v"):
                assert not kw[key][begin+p+n-cut:end].view(torch.uint8).any()
        else:
            assert int(kw["cache_seqlens"][rank])==p+n
            ids=kw["page_table"][rank,cut:p+n].long()
            kv=[kw[key][:,0].view(torch.uint8)[ids].view(FP8) for key in ("k_cache","v_cache")]
        data.extend(digest(t) for t in kv)
        for j in (0,n//2,n-1):
            end=p+j+1
            start=0 if left<0 else max(0,end-1-left)
            for head in (0,QH-1):
                q=kw["q"][q_start+j,head].double()
                k,v=[a[start-cut:end-cut,head//(QH//H)].double() for a in kv]
                scores=(k@q)*kw["softmax_scale"]*(.75 if scale else 1.)
                if kw["softcap"]:
                    scores=(scores/kw["softcap"]).tanh()*kw["softcap"]
                out=scores.softmax(0)@v*(1.625 if scale else 1.)
                assert out.isfinite().all()
                vals.append(out)
        q_start+=n
    return {"logical_kv":data,"q":digest(kw["q"]),"out64":digest(torch.stack(vals)),
            "attention_scalars":len(vals)*D,"logical_kv_bytes":sum((p+n-(0 if left<0 else max(0,p-left)))*H*D*2 for p,n in zip(prefixes,rows))}


def next_ordinary(s,state):
    """Commit three tail tokens per request; exchange the five rejected slots between the two tails."""
    fb,backend,pool=s["fb"],s["backend"],s["pool"]
    rows,prefixes,left=s["rows"],s["prefixes"],s["left"]
    new_p=[prefixes[0],prefixes[1]+3,prefixes[2]+3]
    new_l=[p+r for p,r in zip(new_p,rows)]
    crossed=[]; new_t=[]; new_w=[]
    physical_pool=pool.swa_kv_pool if left>=0 else pool.full_kv_pool
    identities=[t.data_ptr() for t in physical_pool.data]
    for table,owner in zip(s["tables"],[pool.full_kv_pool,pool.swa_kv_pool]):
        nt=table.new_zeros((3,max(new_l)+9))[:,:max(new_l)]
        old_totals=[p+r for p,r in zip(prefixes,rows)]
        used={slot for i,n in enumerate(old_totals) for slot in table[i,:n].tolist()}
        free=[i for i in range(1,owner.data[0].shape[0]) if i not in used][:6]
        assert len(free)==6
        nt[0,:new_l[0]]=table[0,:new_l[0]]
        rejected=[table[i,prefixes[i]+3:old_totals[i]] for i in (1,2)]
        assert all(t.shape==(5,) for t in rejected)
        for i,old in enumerate(reversed(rejected),1):
            nt[i,:new_p[i]]=table[i,:new_p[i]]
            nt[i,new_p[i]:new_p[i]+5]=old
            nt[i,new_p[i]+5:new_l[i]]=table.new_tensor(free[(i-1)*3:i*3])
            assert not torch.isin(old,nt[i,:new_p[i]]).any()
        new_t.append(nt)
        new_w.append(torch.cat([nt[i,p:p+r] for i,(p,r) in enumerate(zip(new_p,rows))]+[nt.new_zeros(s["pad"])]).long())
        crossed.append(torch.cat(rejected).long())
    bank=crossed[int(left>=0)]
    before=[t.view(torch.uint8)[bank].clone() for t in physical_pool.data]
    fb.mixed_verify_logits_indices=None; fb.mixed_verify_is_causal=False; fb.spec_info=None
    fb.seq_lens_cpu=torch.tensor(new_l,dtype=torch.int32); fb.extend_prefix_lens_cpu=new_p
    fb.out_cache_loc=new_w[0]
    f.fx.initialize(backend,pool,fb,new_t,new_w,len(rows),fb.seq_lens_cpu,rows,None)
    # The two no-op graph segment stand-ins emulate fresh producer writes into their fixed addresses.
    q,k,v=state["producer"]
    q.fill_(1.0); k.fill_(4.5); v.fill_(13.0)
    s["dense"].clear();s["paged"].clear();s["stores"].clear()
    with s["graph"]["prefill_graph_scope"](full_graph=False,raw_num_tokens=sum(rows)):
        s["cap"].cuda_graph.replay()
    assert len(s["dense"])==1 and not s["paged"] and len(s["stores"])==1
    assert [t.data_ptr() for t in physical_pool.data]==identities
    assert torch.equal(s["stores"][0][0],new_w[int(left>=0)][:sum(rows)])
    for i,t in enumerate(physical_pool.data):
        after=t.view(torch.uint8)[bank]
        assert ((after!=before[i]).reshape(10,-1).any(1)).all()
        expected=(4.5 if i==0 else 13.0)/(.75 if i==0 else 1.625)
        assert torch.equal(after,torch.full(after.shape,expected).to(FP8).view(torch.uint8))
    return witness(s["dense"][0],True,rows,new_p,left,True),[digest(t) for t in physical_pool.data]


def after(s,state):
    result=witness(s["call"],bool(s["dense"]),s["rows"],s["prefixes"],s["left"],True)
    result["physical"]=[digest(t) for o in (s["pool"].full_kv_pool,s["pool"].swa_kv_pool) for t in o.data]
    result["written"]=[digest(t) for t in s["stores"][0]]
    result["direct"]=bool(s["dense"]);result["minted"]=state["minted"]
    if state["reuse"]:
        with torch.inference_mode(),patch.dict(sys.modules,s["modules"]),patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),patch.object(torch.cuda,"is_current_stream_capturing",return_value=False):
            result["ordinary"],result["ordinary_physical"]=next_ordinary(s,state)
    return result


def adapt():
    # Existing fixture owns the producer and graph capture. Inject only the live role and evidence observations.
    src=inspect.getsource(f.run)
    changes={
        "observed.append((query.dtype":'state["producer"]=(query,k,v)\n        observed.append((query.dtype',
        "dense.clear(); paged.clear(); stores.clear()":"dense.clear(); paged.clear(); stores.clear()\n        set_role(fb,state)",
        "assert direct==(guard not in (\"writer\",\"full_graph\")),(guard,count,left,list(call))":
            'assert direct==state["expect_dense"],(direct,state["guard"],left)',
        "    return [(x.shape,sha256":"    return after(locals(),state)\n    return [(x.shape,sha256",
    }
    for old,new in changes.items():
        assert src.count(old)==1,old
        src=src.replace(old,new)
    scope=dict(vars(f),after=after,set_role=set_role)
    exec(src,scope)
    return scope["run"],scope


run,scope=adapt()
torch.set_num_threads(1)
report=[]
manifests={}; sources={}
cases=[("accepted",2047,True,4096),("uncertified",2047,False,4096),("nonlinear",2047,False,4096),
       ("masked",2047,False,4096),("accepted",-1,False,4096),("accepted",2047,False,16384)]
for guard,left,reuse,prompt_rows in cases:
    data=[]
    for candidate in (PARENT,CHILD):
        m=json.loads((candidate/"validation.json").read_bytes())
        child=candidate==CHILD
        f.reader.CHILD,manifests[m["candidate"]]=f.checked_patch(candidate,m)
        f.reader.STAGE=args.stage.resolve() if args.stage and child else None
        f.STAGE=f.reader.STAGE
        f.source.cache_clear();f.source_hashes["candidate"].clear()
        scope["state"]={"guard":guard,"expect_dense":child and guard=="accepted" and left>=0,"reuse":reuse}
        out=run(True,[prompt_rows,8,8],[405,2201,71],left,None,True)
        sources[m["candidate"]]=dict(f.source_hashes["candidate"])
        data.append(out)
    a,b=data
    assert not a.pop("direct")
    selected=b.pop("direct")
    am,bm=a.pop("minted"),b.pop("minted")
    assert not am and bm==(guard!="nonlinear")
    assert a==b,(guard,left)
    report.append({"guard":guard,"window":left,"rows":[prompt_rows,8,8],
                   "candidate_direct":selected,"graph_query":"torch.float8_e4m3fn",
                   "logical_kv_bytes":a["logical_kv_bytes"],"sampled_fp64_scalars":a["attention_scalars"],
                   "mixed_witness":a["out64"],"cross_request_reused_slots":10 if reuse else 0,
                   "subsequent_ordinary":a.get("ordinary")})
out={"passed":True,"patch_sha256":manifests,"cases":report,"source_sha256":sources,
     "scope":"CPU model/graph recording and replay, production FA3 metadata and KV writer, physical full/sliding translation. Native role setup and post-agreement tag assignment executed with synthetic agreement. Triton and FA3 launch boundaries simulated."}
(CHILD/"independent_causal_attention.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k not in ("source_sha256","cases")}))
