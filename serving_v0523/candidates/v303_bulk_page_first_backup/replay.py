"""Replay normal multi-pool index preparation, DMA, fallback and native per-layer restore."""
from bisect import bisect_right
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass,replace
from types import SimpleNamespace as S
from pathlib import Path
import ast
import json
import tempfile
import torch
from native import compile_cpu,HEADER,RELAY
from verify import BASE,MHA,COMMON,candidate,digest

def expose(source,names,environ,cls=None):
    root=ast.parse(source)
    if cls:root=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name==cls)
    methods={}
    for key in names.split():
        fn=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name==key)
        exec("from __future__ import annotations\n"+ast.unparse(fn),environ)
        methods[key]=environ[key]
    return methods
@dataclass
class Auxiliary:
    name:str
    host_indices:object
    device_indices:object

def main():
    torch.set_num_threads(1)
    hc=BASE+"hybrid_cache/hybrid_cache_controller.py";host=BASE+"pool_host/base.py";engine=BASE+"l2_transfer.py"
    wrapper="python/sglang/kernels/ops/kvcache/hicache.py"
    with candidate() as (_,tree,git):src={p:git("show",tree+":"+p).decode() for p in (MHA,COMMON,host,hc,engine,wrapper,HEADER,RELAY)}
    transfer=namedtuple("L2Transfer","host_pool device_pool host_indices device_indices layer_mapper is_draft",defaults=(None,False))
    env=dict(torch=torch,bisect_right=bisect_right,replace=replace,L2Transfer=transfer,
             debug_kernel_api=lambda fn:fn,_default_unroll=lambda _:4,DEFAULT_BLOCK_QUOTA=2,uses_shared_host_layout=lambda _:False,
             _CUDA_HOST_REGISTERED_RANGES_ATTR="regions",envs=S(SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES=S(get=lambda:0)))
    register=expose(src[COMMON],"registered_kv_row_ends",env)["registered_kv_row_ends"]
    env["jit_transfer_hicache_all_layer_staged_lf_pf"]=expose(src[wrapper],"transfer_hicache_all_layer_staged_lf_pf",env)["transfer_hicache_all_layer_staged_lf_pf"]
    pm=expose(src[MHA],"backup_from_device_all_layer load_to_device_per_layer _try_bulk_page_first_backup",env,"MHATokenToKVPoolHost")
    pm.update(expose(src[host],"backup_from_device_all_layer_physical load_to_device_per_layer_physical prepare_transfer_indices",env,"HostKVCache"))
    Pool=type("Pool",(),pm)
    cm=expose(src[hc],"_move_write_operation _l2_transfers move_hybrid_indices _move_pool_indices",env,"HybridCacheController")
    Ctrl=type("Controller",(),cm)
    stream_log=[]
    class Stream:
        def __init__(self,name):self.name=name
        def synchronize(self):stream_log.append("synchronize")
    class Event:
        def __init__(self,name="producer"):self.name=name
        def record(self):stream_log.append("record "+self.name)
        def wait(self,stream):stream_log.append(stream.name+" wait")
    class Domain:
        def acquire_layout(self):stream_log.append("acquire")
        def release_layout(self,event,key):stream_log.append("release "+(event.name if event else "error"))
    @contextmanager
    def switch(stream):
        stream_log.append("enter "+stream.name)
        try:yield
        finally:stream_log.append("leave "+stream.name)
    env.update(contextmanager=contextmanager,device_module=S(Event=Event,stream=switch),
        make_timing_event_pair=lambda:(Event("start"),Event("finish"),False),
        TransferCompletion=lambda *args:args,shared_host_layout_domains=lambda pools:[Domain()])
    em=expose(src[engine],"submit_device_to_host submit_host_to_device _submission _prepare_transfers _start_event _record_stream _resolve_device_indices",env,"L2TransferEngine")
    em.update(io_backend="kernel",device_to_host_stream=Stream("write"),host_to_device_stream=Stream("read"))
    eng=type("Engine",(),em)()
    checks=[];size_total=0;failure_order=0
    with tempfile.TemporaryDirectory() as temp:
        cpu,hsh=compile_cpu(temp,src)
        for case,n in (("cold",16384),("swa heavy",16384),("shuffled runs",257),("registration cuts",513),("partial and duplicate",515),("mapped fallback",257),
                       ("small",63),("legacy fallback",129),("int32 source",262),("allocation failure",260),("unregistered",256),
                       ("invalid bounds",256),("malformed registration",250),("second copy fault",264),("empty",0)):
            group=Ctrl();group.is_kv_sharded=False;group.io_backend="kernel"
            group.mem_pool_host=S(layout="page_first",supports_per_pool_backup_indices=True,entry_map={})
            all_data=[];entries=[];stats=[];aux=[];gen=torch.Generator().manual_seed(2841)
            for which,(name,layers,width,dtype) in enumerate((("full",16,128,torch.uint8),("swa",46,128,torch.uint8),("draft",5,128,torch.bfloat16))):
                rows=n if case!="cold" or which!=1 else 2048
                if case=="swa heavy" and which!=1:rows=257
                slots=rows+311 if case!="mapped fallback" else 2*rows+14
                h=Pool();h.layout="page_first";h.page_size=h.dcp_size=1;h.size=slots
                h._transfer_device=torch.device("cpu");h.dtype=dtype;h.layer_num=h.target_layer_num=layers;h.element_dim=width
                h.can_use_jit=h.can_use_write_back_jit=h.device_rows_packed=True;h.can_use_mapped_backup=case!="legacy fallback"
                h.staging_page_capacity=64;h.token_stride_size=h.device_row_stride_bytes=width*dtype.itemsize;h.layout_dim=layers*h.token_stride_size
                buf=torch.full((2,slots,layers,width),81,dtype=dtype);h.k_buffer,h.v_buffer=buf
                base=buf.data_ptr();row=h.layout_dim;plane=slots*row; cuts=[0,2*plane]
                if case=="registration cuts":cuts=[0,105*row,plane+71*row,2*plane]
                if case=="malformed registration":cuts=[0,59*row+1,2*plane]
                if case!="unregistered":buf.regions=[(base+a,b-a) for a,b in zip(cuts,cuts[1:])]
                h.bulk_backup_row_ends=register(buf,slots,row)
                if case=="registration cuts":assert h.bulk_backup_row_ends==(71,105,slots)
                if case in ("malformed registration","unregistered"):assert h.bulk_backup_row_ends==()
                tables=buf.transpose(1,2);h.k_data_refs,h.v_data_refs=tables;h.k_data_ptrs,h.v_data_ptrs=tables
                payload=torch.randint(0,256,(2,layers,slots,width*dtype.itemsize),dtype=torch.uint8,generator=gen).view(dtype)
                h.device_k_buffers,h.device_v_buffers=[tuple(x) for x in payload]
                h.device_k_data_ptrs,h.device_v_data_ptrs=h.device_k_buffers,h.device_v_buffers
                h.staging_k_buffer,h.staging_v_buffer=torch.empty(2,64,layers,width,dtype=dtype)
                ids=torch.randperm(slots,generator=gen)[:rows];dest=torch.arange(rows)+3
                if case=="int32 source":ids=ids.int()
                if case=="shuffled runs":dest=dest.roll(117)
                if case=="partial and duplicate":dest[130:265]=dest[3:138].clone()
                if case=="mapped fallback":dest=2*dest
                if case=="legacy fallback":dest=dest.flip(0)
                if case=="invalid bounds":dest=dest+slots
                if case=="allocation failure":
                    class Fail:
                        shape=h.staging_k_buffer.shape
                        def new_empty(self,*a,**k):raise torch.OutOfMemoryError("bounded scratch failed")
                    h.staging_k_buffer=Fail()
                rec=dict(name=name,rows=rows,layers=layers,capacity=slots,runs=[],mapped=0,staged=0,bytes=0)
                stats.append(rec);all_data.append((h,ids,dest,buf,payload,tables,cuts))
                e=S(name=name,host_pool=h,device_pool=None,layer_mapper=lambda i:i)
                entries.append(e);group.mem_pool_host.entry_map[name]=e
                if which==0:group.mem_pool_host.anchor_entry=e
                else:aux.append(Auxiliary(name,dest,ids))
            _,p0,d0,*_=all_data[0]
            op=S(host_indices=d0,device_indices=p0,pool_transfers=aux)
            hs,ds,other=group._move_write_operation(op)
            tx=group._l2_transfers(hs,ds,other)
            assert len(tx)==(3 if n else 0)
            def record(ks):
                index=next(i for i,x in enumerate(all_data) if x[0].device_k_data_ptrs is ks)
                return all_data[index],stats[index]
            def bulk(k,v,d,sk,sv,ids,ks,vs):
                (h,_,_,buf,_,_,cuts),rec=record(ks)
                assert stream_log[-1] in ("record start","bulk")
                stream_log.append("bulk")
                a=int(d[0]);b=a+len(d)
                assert torch.equal(d,torch.arange(a,b))
                for out in (k,v):
                    offset=out.data_ptr()-buf.data_ptr()
                    assert any(x<=offset and offset+out.nbytes<=y for x,y in zip(cuts,cuts[1:]))
                assert sk.nbytes+sv.nbytes<=8<<20
                rec["runs"].append([a,b]);rec["bytes"]+=k.nbytes+v.nbytes
                if case=="second copy fault" and rec["name"]=="swa":raise RuntimeError("DMA submission failed")
                cpu(k,v,d,sk,sv,ids,ks,vs)
            def old(k,v,d,sk,sv,ids,ks,vs,page):
                (_,_,_,_,_,_,_),rec=record(ks);rec["staged"]+=1
                for i,slot in enumerate(d):
                    for layer in range(k.shape[1]):k[slot,layer]=ks[layer][ids[i]];v[slot,layer]=vs[layer][ids[i]]
            def mapped(**kw):
                (_,_,_,_,_,_,_),rec=record(kw["k_ptr_src"]);rec["mapped"]+=1
                if case=="invalid bounds":return
                for dst,srcs in ((kw["k_ptr_dst"],kw["k_ptr_src"]),(kw["v_ptr_dst"],kw["v_ptr_src"])):
                    for i,origin in enumerate(srcs):dst[i][kw["indices_dst"]]=origin[kw["indices_src"].long()]
            env.update(_jit_hicache_staged_module=lambda **_:S(launch_all_lf_pf_bulk=bulk,launch_all_lf_pf_staged=old),
                jit_transfer_hicache_all_layer=mapped)
            def restore(**kw):
                for s in ("k","v"):kw[s+"_cache_dst"][kw["indices_dst"].long()]=kw[s+"_cache_src"][kw["indices_src"]]
            env["jit_transfer_hicache_one_layer"]=restore
            stream_log.clear()
            try:eng.submit_device_to_host(tx)
            except RuntimeError as err:
                if case!="second copy fault" or str(err)!="DMA submission failed":raise
                assert stats[0]["runs"] and stats[1]["runs"] and not stats[2]["runs"] and not any(s["mapped"]+s["staged"] for s in stats)
                assert stream_log[-3:]==["leave write","synchronize","release error"] and "record finish" not in stream_log
                failure_order+=1;continue
            assert [s for s in stream_log if s!="bulk"]==["record producer","acquire","enter write","write wait","record start","record finish","leave write","release finish"]
            if case=="invalid bounds":
                assert all(not s["runs"] for s in stats)
                continue
            for rec,(h,ids,dest,buf,payload,tables,_) in zip(stats,all_data):
                if case in ("cold","swa heavy","shuffled runs","registration cuts","partial and duplicate","int32 source"):
                    assert rec["runs"] and not rec["mapped"] and not rec["staged"],(case,rec)
                elif case!="empty":assert not rec["runs"] and (rec["mapped"] or rec["staged"]),(case,rec)
                for side in range(2):
                    for l in range(h.layer_num):
                        want=torch.full_like(payload[side,l],81);origin=payload[side,l]
                        if case=="partial and duplicate":
                            for j,i in enumerate(dest):want[i]=origin[ids[j]]
                        else:want[dest]=origin[ids.long()]
                        assert torch.equal(want.view(torch.uint8),tables[side,l].view(torch.uint8)),(case,side,l)
                size_total+=buf.nbytes
                count=min(len(dest),4);payload[:,:,:count]=0
                back=transfer(h,None,dest[:count],torch.arange(count))
                done=[];eng.submit_host_to_device([back],transfer_layer_id_max=h.layer_num,on_layer_done=done.append)
                assert done==list(range(h.layer_num))
                assert torch.equal(payload[:,:,:count].view(torch.uint8),tables[:,:,dest[:count]].view(torch.uint8))
                size_total+=payload[:,:,:count].nbytes
            checks.append(dict(case=case,pools=stats))
    print(json.dumps(dict(tree=tree,source_sha256={k:digest(v.encode()) for k,v in src.items()},native_slices=hsh,
        compared_bytes=size_total,second_pool_failure_no_retry=failure_order,cases=checks,
        scope="Native CPU hybrid control, index move, D2H/H2D submission, native event/lease ordering, MHA fast/fallback and wrapper. Compiled native gather loops and both actual DMA commands with a host memory runtime. Streams and device addresses simulated."),indent=2))
if __name__=="__main__":main()
