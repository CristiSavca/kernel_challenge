"""Carry one draft helper through real scheduler rounds, compare V212/V214 at four and eight ranks."""
import ast
from collections import Counter
from datetime import timedelta
from functools import lru_cache
from hashlib import sha256
import inspect
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import ModuleType
from unittest.mock import patch
import weakref

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"v212_mixed_paired_draft_kv"
STAGE=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(BASE))
import mixed_scheduler as previous

PROJ="python/sglang/srt/layers/dflash_context_projection.py"
PACK="python/sglang/kernels/ops/speculative/fused_kv_materialize.py"
V212=json.loads((BASE/"validation.json").read_text())
V214=json.loads((HERE/"validation.json").read_text())
assert sha256((HERE/"patch.diff").read_bytes()).hexdigest()==V214["patch_sha256"]
BANKS={}


@lru_cache(None)
def published(version,rel):
    manifest=V212 if version=="V212" else V214
    raw=subprocess.check_output(["git","-C",str(STAGE),"show",manifest["tree"]+":"+rel])
    if version=="V214":assert raw==(STAGE/rel).read_bytes()
    return raw


def projection(version):
    module=ModuleType("active_paired_projection")
    exec(compile(published(version,PROJ),PROJ,"exec"),module.__dict__)
    return module


class Stream:
    """Record ordering at CUDA Python boundaries while Gloo transports the CPU tensors."""
    def __init__(self):
        self.records=[]
        self.waits=[]
        self.uses=[]
        self.slot=0
        self.handles=[StreamHandle(self,i) for i in range(2)]
        self.cross_stream_waits=0


class StreamHandle:
    def __init__(self,probe,index):
        self.probe,self.index=probe,index
    def record_event(self):
        event=(self.index,object())
        self.probe.records.append(event)
        return event
    def wait_event(self,event):
        assert event in self.probe.records
        self.probe.waits.append(event)
        self.probe.cross_stream_waits+=event[0]!=self.index


def bank(real,rank,hidden,version,case):
    key=(version,hidden)
    if key not in BANKS or (case=="new_helper" and rank==dist.get_world_size()-1):
        cls=next(n for n in ast.parse(published(version,PACK)).body if getattr(n,"name","")=="FusedKVMaterializeHelper")
        init=next(n for n in cls.body if getattr(n,"name","")=="__init__")
        real.Helper.__init__=previous.bind([init],real.Helper.__init__.__globals__,PACK)["__init__"]
        BANKS[key]=previous.kv_model(real,rank,hidden)
    layers,helper=BANKS[key]
    weight=helper.flat_kv_weight_t
    if case in ("mutate","replace") and rank==dist.get_world_size()-1:
        if case=="mutate":
            weight.neg_()
        else:
            with torch.inference_mode(False),torch.no_grad():
                helper.flat_kv_weight_t=weight.clone().neg_()
        # The ordinary per-layer path may run after a helper is retired.
        # It reads the same refreshed checkpoint views as the packed path.
        weights=helper.flat_kv_weight_t.T.view(5,4,hidden)
        for idx,l in enumerate(layers):
            l.self_attn.qkv_proj.weight[2:].copy_(weights[idx])
    return layers,helper


def native_transport(module,rank,stream):
    def run(rows,n,group):
        helper=module._draft_kv_helper.get()
        if helper is None or not hasattr(helper,"flat_kv_weight_t"):
            return module.project_paired_draft_kv(rows,n,group)
        w=helper.flat_kv_weight_t
        # The computation and communication stay on CPU. Only the native stream
        # branch sees the probe weight as CUDA-backed; all other flags stay native.
        is_cuda=torch.Tensor.is_cuda
        def where(tensor):
            return True if tensor is w else is_cuda.__get__(tensor,type(tensor))
        def record(tensor,used_stream):
            assert used_stream is stream.handles[stream.slot]
            stream.uses.append(weakref.ref(tensor))
        actual_mm=torch.mm
        def check_mm(x,y,**kw):
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            return actual_mm(x,y,**kw)
        with (patch.object(torch.Tensor,"is_cuda",property(where)),
              patch.object(torch.cuda,"current_stream",return_value=stream.handles[stream.slot]),
              patch.object(torch.Tensor,"record_stream",record),patch.object(torch,"mm",check_mm)):
            return module.project_paired_draft_kv(rows,n,group)
    return run


def build_rig():
    # Reuse the V212 physical oracle and scheduler data construction, replacing
    # only synthetic fixture setup. Production entry points below come from git.
    text=inspect.getsource(previous.ranked)
    setup,rest=text.split("    snapshots={}\n",1)
    header,rest=rest.split("        cases=",1)
    _,header=header.split('    for version in ("V211","V212"):\n',1)
    oracle=rest.split("                with torch.inference_mode():\n",1)[1].split("                touched_transport=",1)[0]
    def modify(old,new):
        nonlocal setup
        assert setup.count(old)==1,old
        setup=setup.replace(old,new)
    modify("def ranked(rank,size,folder):","def execute(rank,size,version,backend):")
    modify("    projection=context.projection",
           "    projection=make_projection(version)\n    stream=Stream()")
    modify("mr=old_policy((list(range(1,2049)),grids[1]))",
           "mr=old_policy(([256*size] if case.startswith('padded_') else list(range(1,2049)),grids[1]))")
    modify('        processor=model.logits_processor.__self__',
           '''        if case.startswith("padded_"):
            original_load=graph.load_batch
            def padded(fb,**kwargs):
                static,meta=original_load(fb,**kwargs)
                original_rows=fb.input_ids.numel()
                static.input_ids=torch.cat((fb.input_ids,torch.full((256*size-original_rows,),-65537)))
                static.positions=torch.arange(256*size)
                return static,meta
            graph.load_batch=padded
        processor=model.logits_processor.__self__''')
    modify("            h,taps=model_input(ref,ids)",
           """            h,taps=model_input(ref,ids)
            if case.startswith("padded_"):
                for x in (h,*taps):x[ids==-65537]=float("nan")""")
    modify("    def gather(dest,value):",
           """    original_send=comm.send
    original_end=comm.group_end
    def check_send(x,peer):
        assert torch.isfinite(x).all()
        original_send(x,peer)
    def end_group():
        if comm.ops==[]:
            comm.ops=None
        else:
            original_end()
    comm.send=check_send
    comm.group_end=end_group
    def gather(dest,value):""")
    modify("                   project_paired_draft_kv=paired,",
           "                   project_paired_draft_kv=native_transport(projection,rank,stream),")
    modify('hidden=16 if case=="traffic_veto" else 64','hidden=26 if size==4 else 25')
    modify("layers,helper=kv_model(real,rank,hidden)","layers,helper=bank(real,rank,hidden,version,case)")
    modify("        worker._use_fused_kv_materialize=True",'''        if case=="padded_cold":
            helper._paired_kv_peer=None
        worker._use_fused_kv_materialize=True''')
    modify("        def expected_rows(raw):","        spy['helper']=helper\n        def expected_rows(raw):")
    modify("test.settle.prepare=prepare","test.settle.prepare=prepare\n    BANKS.clear()")
    env=dict(vars(previous),make_projection=projection,native_transport=native_transport,
             Stream=Stream,bank=bank,BANKS=BANKS,weakref=weakref)
    env["verify_padded_messages"]=verify_padded_messages
    def native_entries(version,path,cls=None):
        body=ast.parse(published(version,"python/sglang/srt/"+path)).body
        if cls:body=next(n.body for n in body if getattr(n,"name",None)==cls)
        return {n.name:n for n in body if isinstance(n,ast.FunctionDef)}
    env["entries"]=native_entries
    code=(setup+textwrap.indent(textwrap.dedent(header),"    ")
          +textwrap.indent(textwrap.dedent(ROUND_PREFIX),"    ")
          +textwrap.indent(textwrap.dedent(oracle),"            ")
          +textwrap.indent(textwrap.dedent(ROUND_SUFFIX),"        ")
          +"    return result\n")
    exec(compile(code,"<V212 synthetic scheduler rig with persistent V214 bank>","exec"),env)
    return env["execute"]


ROUND_PREFIX='''
sequence=["short_cold","prime","normal","short","mutate","reprime","post_mutate","replace","reprime_again",
          "post_replace","rank_retired","new_helper","post_retirement","different_tokens"]
if backend=="FULL": sequence+=["padded_cold","padded_warm"]
result=[]
cached_remote=None
for j,case in enumerate(sequence):
    stream.slot=j%2
    small=case in ("short_cold","short")
    span=128 if small else 252
    total=128*size+17 if case.startswith("padded_") else span*size
    sizes=[total-23,7]
    before=comm.sent
    offset=len(comm.group_shapes)
    nrecord,nwait,nuses=len(stream.records),len(stream.waits),len(stream.uses)
    nremote=stream.cross_stream_waits
    comm.available=True
    spy.clear()
    spy["head_gathers"]=[]
    with torch.inference_mode():
'''


ROUND_SUFFIX='''
helper=spy["helper"]
assert torch.is_inference(helper.flat_kv_weight_t)==(version=="V212" and not (rank==size-1 and j>=7 and j<11))
sw=(ref.V+size-1)//size
assert spy["head_gathers"]==[(2,sw),(16,2)]
assert spy["head_cols"]==sw
group_ops=comm.group_shapes[offset:]
sent_weights=[peer for ops in group_ops for direction,peer,shape in ops
              if direction=="send" and shape==(helper.flat_kv_weight_t.shape[0],20)]
got_peer=getattr(helper,"_paired_kv_peer",None)
paired=spy["projected"]
if version=="V212":
    expected=case not in ("rank_retired","short","short_cold","padded_cold","padded_warm")
    assert not got_peer and not stream.records
else:
    expected=case not in ("short_cold","mutate","replace","rank_retired")
    if expected:
        assert got_peer is not None and got_peer.group is group
        assert got_peer.local_weight is helper.flat_kv_weight_t
        assert got_peer.version==helper.flat_kv_weight_t._version
        assert not got_peer.remote_weight._base
        buddy=(rank+size//2)%size
        cold=case in ("prime","reprime","reprime_again","new_helper","padded_cold")
        assert (buddy in sent_weights)==cold,(rank,size,backend,case,sent_weights)
        assert len(stream.records)-nrecord==int(cold)
        assert len(stream.waits)-nwait==int(not cold)
        assert len(stream.uses)-nuses==1+int(not cold)
        assert stream.uses[nuses]() is helper.flat_kv_weight_t
        if not cold:
            assert stream.waits[-1] is got_peer.ready_event
            assert stream.uses[-1]() is got_peer.remote_weight
        cached_remote=weakref.ref(got_peer.remote_weight)
    if case in ("mutate","replace"):
        assert got_peer is None
        assert cached_remote is None or cached_remote() is None
        cached_remote=None
assert paired==expected,(rank,size,backend,version,case,paired,group_ops)
assert len(group_ops)==2*int(paired)
if case.startswith("padded_") and paired:
    verify_padded_messages(group_ops,rank,size,256,total,helper.flat_kv_weight_t.shape,case=="padded_cold")
padding=None
if case.startswith("padded_"):
    padding=[None]*size
    pitch=256
    live=max(0,min(pitch,total-rank*pitch))
    dist.all_gather_object(padding,dict(rank=rank,local_live_rows=live,paired_send_bytes=comm.sent-before))
result.append(dict(case=case,backend=backend,paired=paired,weight_sends=len(sent_weights),
    buddy_weight_sent=(rank+size//2)%size in sent_weights,wire_bytes=comm.sent-before,
    cache_words=sum(t.numel() for t in pool.k_buffer+pool.v_buffer),
    stream_records=len(stream.records)-nrecord,stream_waits=len(stream.waits)-nwait,
    cross_stream_waits=stream.cross_stream_waits-nremote,
    protected_weights=len(stream.uses)-nuses,commits=commits.tolist(),
    rejected=int((~accepted).sum()),digest=digest.hexdigest(),report=report,padding=padding))
del got_peer
'''


def verify_padded_messages(groups,rank,size,pitch,live,weight_shape,cold):
    hidden,columns=weight_shape
    def cost(width):
        peers=size//width-1
        return (width-1)*pitch*hidden+(peers-int(not cold))*hidden*columns+peers*width*pitch*columns
    width=min((w for w in (2,4) if w<size),key=cost)
    start=rank//width*width
    def rows(first,count):
        return max(0,min(count*pitch,live-first*pitch))
    local=rows(rank,1)
    a,b=[],[]
    for other in range(start,start+width):
        if other==rank:continue
        if local:a.append(("send",other,(local,hidden)))
        if rows(other,1):a.append(("recv",other,(rows(other,1),hidden)))
    for peer in range(rank%width,size,width):
        if peer==rank:continue
        if cold or peer!=(rank+size//2)%size:
            a.extend((("send",peer,(hidden,columns)),("recv",peer,(hidden,columns))))
        source=rows(start,width)
        incoming=rows(peer//width*width,width)
        if source:b.append(("send",peer,(source,columns)))
        if incoming:b.append(("recv",peer,(incoming,columns)))
    assert groups==[a,b],(rank,live,groups,[a,b])
    assert rows(size-1,1)==0 and any(0<rows(k,1)<pitch for k in range(size))


def worker(rank,size,folder):
    logging.disable(logging.WARNING)
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(folder)/"init"),rank=rank,
                            world_size=size,timeout=timedelta(seconds=75))
    try:
        execute=build_rig()
        output=[]
        for backend in ("eager","FULL","BREAKABLE"):
            base=execute(rank,size,"V212",backend)
            new=execute(rank,size,"V214",backend)
            for a,b in zip(base,new,strict=True):
                for k in ("report","digest","commits","rejected","cache_words"):
                    assert a[k]==b[k],(rank,size,backend,a["case"],k)
                output.append(dict(b,base_paired=a["paired"],base_bytes=a["wire_bytes"],
                                   base_weight_sends=a["weight_sends"]))
        if rank==0:Path(folder,"out.json").write_text(json.dumps(output))
    finally:dist.destroy_process_group()


def main():
    result={}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker,args=(size,tmp),nprocs=size,join=True)
            data=json.loads(Path(tmp,"out.json").read_text())
        result[size]=dict(comparisons=len(data),physical_cache_words_compared_per_rank=sum(x["cache_words"] for x in data),
            stream_producer_events=sum(x["stream_records"] for x in data),stream_consumer_waits=sum(x["stream_waits"] for x in data),
            waits_across_two_distinct_streams=sum(x["cross_stream_waits"] for x in data),
            local_and_remote_allocator_streams=sum(x["protected_weights"] for x in data),
            formerly_vetoed_admissions=sum(x["paired"] and not x["base_paired"] for x in data),
            padded_full={x["case"]:x["padding"] for x in data if x["padding"]},
            full_replay_transitions=[{k:x[k] for k in ("case","paired","base_paired","weight_sends","base_weight_sends",
                "wire_bytes","base_bytes","stream_records","stream_waits")} for x in data if x["backend"]=="FULL"])
    print(json.dumps(dict(parent_sha=V212["patch_sha256"],candidate_sha=V214["patch_sha256"],worlds=result,
                          source_hashes={v:{f:sha256(published(v,f)).hexdigest() for f in (PROJ,PACK)} for v in ("V212","V214")},
                          cpu_gloo=True,physical_H200=False,cuda_stream_boundary_emulated=True),indent=2))


if __name__=="__main__":main()
