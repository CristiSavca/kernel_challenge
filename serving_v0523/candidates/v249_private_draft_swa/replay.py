"""Exercise V249's native allocation, projected stores, target SWA rebind and host routing on CPU."""
import argparse
import ast
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import logging
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS

import torch

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parents[1]
BASE = SERVICE/"sglang"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
SRT = "python/sglang/srt/"
KW = "mem_cache/kv_cache_configurator.py"
WR = "speculative/dflash_worker_v2.py"
SW = "mem_cache/swa_memory_pool.py"
AS = "mem_cache/hybrid_cache/hybrid_pool_assembler.py"
SC = "mem_cache/unified_cache/components/swa.py"
FILES = {}


def source(stage, path):
    target = stage/(SRT+path)
    if target.is_file():
        raw = target.read_bytes()
    else:
        raw = subprocess.check_output(["git","-C",BASE,"show",f"{PIN}:{SRT}{path}"])
    FILES[path] = sha256(raw).hexdigest()
    return raw.decode()


def method(stage, path, cls, name, scope):
    module = ast.parse(source(stage,path))
    owner = next((n for n in module.body if getattr(n,"name","") == cls),module)
    node = next(n for n in owner.body if getattr(n,"name","") == name)
    node.decorator_list = []
    env = dict(scope)
    exec("from __future__ import annotations\n"+ast.unparse(node),env)
    return env[name]


class Plane:
    def __init__(self,size,*,dtype,layer_num,head_dim,head_num,device,**kw):
        self.size=self.pre_capacity=size
        self.device=device
        self.start_layer=0
        self.layer_num=layer_num
        self.post_capture_active=kw.get("post_capture_active",False)
        self.k_buffer=[torch.full((size+1,head_num,head_dim),-913.,dtype=dtype) for _ in range(layer_num)]
        self.v_buffer=[v.clone() for v in self.k_buffer]
        self.prefix_calls=0

    def get_kv_size_bytes(self):
        return tuple(sum(t.numel()*t.element_size() for t in v) for v in (self.k_buffer,self.v_buffer))

    def _finalize_backing_tokens(self,n):
        assert n<=self.pre_capacity
        self.size=n

    def get_kv_buffer(self,layer): return self.k_buffer[layer],self.v_buffer[layer]
    def get_key_buffer(self,layer): return self.k_buffer[layer]
    def get_value_buffer(self,layer): return self.v_buffer[layer]

    def set_kv_buffer(self,attn,loc,k,v,ks,vs,layer_id_override=None):
        layer=attn.layer_id if layer_id_override is None else layer_id_override
        ids=loc.loc if hasattr(loc,"loc") else loc
        self.k_buffer[layer][ids]=k
        self.v_buffer[layer][ids]=v

    def set_kv_buffer_prefix_valid(self,attn,loc,commit,k,v,*_):
        selected=torch.arange(loc.shape[1])[None,:]<commit[:,None]
        self.set_kv_buffer(attn,loc[selected],k[selected.reshape(-1)],v[selected.reshape(-1)],1.,1.)
        self.prefix_calls+=1


class Allocator:
    def __init__(self,mapping):
        self.full_to_swa_index_mapping=mapping
        self.swa_req_ring=False
        self.page_size=1
        self.freed=[]

    def get_kvcache(self):return self.target_pool
    def translate_swa_indices_for_transfer(self,ids):return self.full_to_swa_index_mapping[ids]
    def clear_full_to_swa_mapping(self,ids):self.full_to_swa_index_mapping[ids]=0
    def set_full_to_swa_mapping(self,ids,phys):self.full_to_swa_index_mapping[ids]=phys
    def free_full_segment(self,ids,start_pos):self.freed.append(ids.clone())


def run(stage):
    torch.set_num_threads(1)
    alloc=Allocator(torch.zeros(1404,dtype=torch.long))
    par=NS(attn_dp_enabled=False,attn_cp_size=1,attn_dcp_size=1,attn_tp_size=8,enable_kv_cache_sharding=False)
    mem=NS(enable_unified_memory=False,enable_unified_cache_external_linker=False,hicache_storage_backend=None)
    dis=NS(disaggregation_mode="null",enable_pdmux=False)
    kind=NS(FULL="full",SLIDING_WINDOW="swa")
    env=dict(torch=torch,dataclass=dataclass,replace=replace,SWATokenToKVPoolAllocator=Allocator,
        MHATokenToKVPool=Plane,MLATokenToKVPool=type("Mla",(),{}),DSATokenToKVPool=type("Dsa",(),{}),
        logger=logging.getLogger("replay"),maybe_init_custom_mem_pool=lambda **kw:(False,None,None),
        GB=2**30,get_parallel=lambda:par,get_exec=lambda:NS(features=NS(enable_memory_saver=False)),
        get_schedule=lambda:NS(prefill_only_disable_kv_cache=False),get_spec=lambda:NS(speculative_algorithm="DFLASH"),
        get_memory=lambda:mem,get_disagg=lambda:dis,is_cuda=lambda:True,draft_tp_context=lambda _:nullcontext(),
        _is_npu=False,IdSpaceKind=kind,KVWriteLoc=lambda ids,**kw:NS(loc=ids,**kw))
    def bind(path,cls,name):return method(stage,path,cls,name,env)
    pool_type=type("SWAKVPool",(),{
        n:bind(SW,"SWAKVPool",n) for n in ("__init__","get_kv_size_bytes","register_mapping",
            "translate_loc_from_full_to_swa","finalize_backing","_wait_for_layer","get_kv_buffer")})
    pool_type.post_capture_active=property(bind(SW,"SWAKVPool","post_capture_active"))
    env["SWAKVPool"]=pool_type
    methods=("_can_share_swa_addresses","alloc_memory_pool","_draft_write_ids","_context_kv_pool",
        "_append_target_hidden_to_draft_kv_by_loc","_append_target_hidden_sequential","_append_target_hidden_fused")
    w=type("Worker",(),{n:bind(WR,"DFlashWorkerV2",n) for n in methods})()
    alloc.target_pool=pool_type.__new__(pool_type)
    req=object()
    w.model_runner=NS(token_to_kv_pool=alloc.target_pool,req_to_token_pool=req,sliding_window_size=2047,
        canary_manager=None,pp_group=NS(world_size=1),device=torch.device("cpu"))
    w.draft_model_runner=NS(draft_attention_backend="fa4",kv_cache_dtype=torch.bfloat16,is_hybrid_swa=False,
        use_mla_backend=False,layer_info=NS(num_effective_layers=5))
    w.use_compact_draft_cache=False
    w._is_domino=False
    w.lilicorr=None
    w.draft_parallel_layout=None
    w.server_args=NS()
    class Attention:
        def __init__(self,i):
            self.num_kv_heads=1
            self.head_dim=8
            self.attn=NS(layer_id=i,is_cross_attention=False,sliding_window_size=2047,k_scale=None,v_scale=None)
        def kv_proj_only(self,h):
            return (h*(self.attn.layer_id+1)).bfloat16(),(h+self.attn.layer_id).bfloat16()
        def apply_k_norm(self,x): return (x*.75).bfloat16()
        def apply_k_rope(self,p,x): return (x+p[:,None].float()*.0001).bfloat16()
    layers=[NS(self_attn=Attention(i)) for i in range(5)]
    w.draft_model=NS(layers=layers,project_target_hidden=lambda h:h.bfloat16(),prepare_context_hidden_for_kv=lambda layer,h:h)
    @dataclass
    class Config:
        full_max_total_num_tokens:int=1400
        swa_max_total_num_tokens:int=340
        dflash_private_swa:bool=False
    cfg=Config()
    for backend in ("fa3","fa4"):
        w.draft_model_runner.draft_attention_backend=backend
        assert w._can_share_swa_addresses(cfg,req,alloc)
    declines=0
    for obj,name,value in [(w,"use_compact_draft_cache",True),(w,"_is_domino",True),(alloc,"swa_req_ring",True),
            (alloc,"page_size",16),(w.model_runner,"sliding_window_size",1023),(w.model_runner,"canary_manager",object()),
            (w.draft_model_runner,"draft_attention_backend","triton"),(w.draft_model_runner,"kv_cache_dtype",torch.float16),
            (w.draft_model_runner,"is_hybrid_swa",True),(w.draft_model_runner,"use_mla_backend",True),
            (w.model_runner.pp_group,"world_size",2),(par,"attn_dp_enabled",True),(par,"attn_cp_size",2),
            (par,"attn_dcp_size",2),(par,"enable_kv_cache_sharding",True),(mem,"enable_unified_memory",True),
            (mem,"enable_unified_cache_external_linker",True),(mem,"hicache_storage_backend","file"),
            (dis,"disaggregation_mode","decode"),(dis,"enable_pdmux",True),(layers[0].self_attn.attn,"sliding_window_size",-1),
            (layers[1].self_attn.attn,"layer_id",0),(layers[4].self_attn.attn,"is_cross_attention",True)]:
        old=getattr(obj,name);setattr(obj,name,value)
        assert not w._can_share_swa_addresses(cfg,req,alloc),(name,value)
        setattr(obj,name,old);declines+=1
    for config,requests,allocator in ((None,req,alloc),(cfg,None,alloc),(cfg,object(),alloc),(cfg,req,object())):
        assert not w._can_share_swa_addresses(config,requests,allocator)
        declines+=1
    build=bind(KW,"KVCacheConfigurator","_build_mha_kv_pool")
    kw=NS(kv_cache_dtype_str="bfloat16",kv_cache_dtype=torch.bfloat16,kv_shard_rank=None,
        device="cpu",post_capture_kv_active=True,pool_page_size=1,token_to_kv_pool_allocator=alloc,
        model_config=NS(get_num_kv_heads=lambda *_:1,head_dim=8,v_head_dim=8),is_draft_worker=True,
        layer_info=NS(num_effective_layers=5,start_layer=0,end_layer=5))
    def allocate(**params):
        assert params["req_to_token_pool"] is req and params["token_to_kv_pool_allocator"] is alloc
        kw.memory_pool_config=params["memory_pool_config"]
        w.draft_model_runner.token_to_kv_pool=build(kw,max_total_num_tokens=1400,mha_pool_class=Plane)
    w._draft_worker=NS(alloc_memory_pool=allocate)
    w.alloc_memory_pool(cfg,req,alloc)
    pool=w.draft_model_runner.token_to_kv_pool
    assert not cfg.dflash_private_swa and pool.dflash_private_swa
    assert pool.full_layer_nums==0 and pool.swa_layer_nums==5 and pool.full_to_swa_index_mapping is alloc.full_to_swa_index_mapping
    assert pool.layers_mapping=={i:(i,True) for i in range(5)}
    pool.finalize_backing(NS(full_max_total_num_tokens=1300,swa_max_total_num_tokens=300))
    assert (pool.full_kv_pool.size,pool.swa_kv_pool.size)==(1300,300)
    dense=Plane(1400,dtype=torch.bfloat16,layer_num=5,head_dim=8,head_num=1,device="cpu")
    rng=torch.Generator().manual_seed(249)
    full=torch.randperm(1200,generator=rng)[:256]+1
    swa=torch.randperm(297,generator=rng)[:256]+1
    alloc.full_to_swa_index_mapping[full]=swa
    # Build the two actual KVIndexTranslator instances and the iteration plan.
    # The draft owns storage, while its rows follow the same physical allocator.
    kt="mem_cache/kv_index_translator.py"
    kp="mem_cache/kv_loc_plan.py"
    native=dict(env,BaseSWAKVPool=pool_type,DeepSeekV4TokenToKVPool=type("Dsv4",(),{}),
        UnifiedMambaTokenToKVPoolAllocator=type("Mamba",(),{}),UnifiedSWAAllocatorBase=type("Unified",(),{}),
        fused_draft_host_allocator=lambda _:None,IdSpace=lambda **kw:NS(**{"write":None,**kw}))
    translator=type("Translator",(),{name:method(stage,kt,"KVIndexTranslator",name,native)
        for name in ("__init__","space")})
    req_table=torch.empty((5,64),dtype=torch.int32)
    def make_translator(p,is_draft):
        return translator(req_to_token=req_table,token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=p,page_size=1,device="cpu",is_draft_worker=is_draft)
    alloc.target_pool.full_to_swa_index_mapping=alloc.full_to_swa_index_mapping
    target_reader=make_translator(alloc.target_pool,False)
    draft_reader=make_translator(pool,True)
    w.draft_model_runner.kv_index_translator=draft_reader
    assert target_reader.space(kind.SLIDING_WINDOW).key!=draft_reader.space(kind.SLIDING_WINDOW).key
    plan_cls=type("Plan",(),{name:method(stage,kp,"KVLocPlan",name,env)
        for name in ("__init__","write_ids","_window_in","_cols")})
    def locate(full_ids):
        plan=plan_cls(source=target_reader,req_pool_indices=torch.arange(4),seq_lens=torch.ones(4),
            seq_lens_cpu=None,write_virtual=full_ids)
        assert torch.equal(plan.write_ids(target_reader),full_ids)
        actual=w._draft_write_ids(plan)
        assert torch.equal(actual,plan.write_ids(target_reader,kind=kind.SLIDING_WINDOW))
        return actual
    def expected(layer,h,pos):
        a=layers[layer].self_attn;k,v=a.kv_proj_only(h)
        return a.apply_k_rope(pos,a.apply_k_norm(k))[:,None],v[:,None]
    def fused(**params):
        for i in range(5):
            params["write_layer_kv"](i,*expected(i,params["ctx_hidden"],params["positions"]))
            if i==2 and partial_failure:raise RuntimeError("synthetic partial completion")
    checked=0
    for index,(fused_on,partial_failure,accepted) in enumerate((
            (False,False,False),(True,False,False),(True,True,False),
            (False,False,True),(True,False,True),(True,True,True))):
        actual=full[index*32:(index+1)*32]
        pos=torch.arange(index*32,(index+1)*32)
        h=torch.randn(32,8,generator=rng,dtype=torch.bfloat16)
        ids=locate(actual)
        counts=torch.tensor([0,8,1,5],dtype=torch.int32)
        keep=(torch.arange(8)[None]<counts[:,None]).reshape(-1) if accepted else torch.ones(32,dtype=torch.bool)
        w._use_fused_kv_materialize=fused_on
        w._fused_kv_helper=NS(materialize=fused)
        w._append_target_hidden_to_draft_kv_by_loc(target_hidden=h,positions=pos,cache_loc=ids,
            cache_loc_2d=ids.reshape(4,8) if accepted else None,commit_lens=counts if accepted else None)
        for layer in range(5):
            k,v=expected(layer,h,pos)
            dense.k_buffer[layer][actual[keep]]=k[keep];dense.v_buffer[layer][actual[keep]]=v[keep]
            for d,sm in zip(dense.get_kv_buffer(layer),pool.get_kv_buffer(layer)):
                assert torch.equal(d[actual],sm[ids])
                checked+=d[actual].numel()
    # Restore the five physical SWA planes under fresh ownership; the native
    # full-id rebind must carry the smaller private draft by the same mapping.
    comp_type=type("Component",(),{n:bind(SC,"SWAComponent",n) for n in ("_set_swa","_translate_full_to_swa")})
    comp=comp_type();comp.cache=NS(token_to_kv_pool_allocator=alloc);comp._unified_allocator=lambda:None
    old=full[192:224]; incoming=full[:32]
    fresh=alloc.full_to_swa_index_mapping[incoming].clone()
    alloc.full_to_swa_index_mapping[old]=0
    new=comp._set_swa(old,incoming)
    assert torch.equal(new,fresh) and torch.equal(locate(old),fresh) and not locate(incoming).any()
    for i in range(5):
        for d,sm in zip(dense.get_kv_buffer(i),pool.get_kv_buffer(i)):
            assert torch.equal(sm[locate(old)],d[incoming]);checked+=sm[locate(old)].numel()

    # Execute the stock host sidecar builder and make its actual layer map and
    # source index domain drive a later physical restore on all five layers.
    names=NS(KV="kv",MAMBA="mamba",SWA="swa",DRAFT_SWA="draft_swa")
    hostenv=dict(env,PoolName=names,PoolHitPolicy=NS(TRAILING_PAGES="trailing"),SidecarPoolSpec=NS,
        build_pool_entry=lambda **kw:NS(**kw),DeepSeekV4PagedHostPool=type("Dsv4Host",(),{}),
        _get_allocator_type=lambda:None,_build_mha_mla_host_pool=lambda **kw:NS(layer_num=kw["pool"].layer_num,**kw))
    host=NS(size=600,page_size=1,layout="page_first")
    tree=NS(cache_controller=NS(mem_pool_host=NS(entry_map={"swa":NS(host_pool=host)})))
    build_host=method(stage,AS,None,"build_swa_draft_pools",hostenv)
    spec,entries=build_host(draft_kv_pool=pool,tree_cache=tree)
    assert spec[0].indices_from_pool=="swa" and spec[0].hit_policy=="trailing"
    entry,=entries
    assert entry.device_pool is pool.swa_kv_pool and entry.transfer_layer_id_max==5
    assert entry.host_pool.host_to_device_ratio==2 and entry.layer_mapping=={i:i for i in range(5)}
    # Run the radix builder, both L2 index resolvers and concrete host/device
    # plan. Distinct FULL, SWA and host indices expose domain mix-ups.
    phases=NS(BACKUP_HOST="backup",LOAD_BACK="load",PREFETCH="prefetch")
    def transfer(**kw):
        return NS(**{"indices_from_pool":None,"device_indices":None,"host_indices":None,**kw})
    hostenv.update(CacheTransferPhase=phases,PoolTransfer=transfer,
        ComponentType=NS(SWA="window component",MAMBA="mamba component"),L2Transfer=NS)
    route=method(stage,"mem_cache/unified_radix_cache.py","UnifiedRadixCache","_build_sidecar_transfers",hostenv)
    resolve_host=method(stage,"mem_cache/pool_host/group.py","HostPoolGroup","resolve_host_transfers",hostenv)
    hc="mem_cache/hybrid_cache/hybrid_cache_controller.py"
    resolve_device=method(stage,hc,"HybridCacheController","_resolve_device_transfers",hostenv)
    plan_l2=method(stage,hc,"HybridCacheController","_l2_transfers",hostenv)
    restored=swa[224:256]
    host_ids=torch.arange(450,482)
    full_host=torch.arange(900,932)
    kv=transfer(name=names.KV,device_indices=incoming,host_indices=full_host,keys=["full"])
    sw=transfer(name=names.SWA,device_indices=fresh,keys=["swa"])
    sw_entry=NS(name=names.SWA,device_pool=NS(),host_pool=host,layer_mapper=lambda i:i,
        device_alloc_fn=lambda n:restored,device_free_fn=lambda _:None,device_evict_fn=None,host_evict_fn=None)
    entry.layer_mapper=entry.layer_mapping.get
    group=NS(anchor_entry=NS(name=names.KV,device_pool=NS(),host_pool=NS(),layer_mapper=lambda i:i),
        entry_map={names.DRAFT_SWA:entry,names.SWA:sw_entry},alloc=lambda n,**kw:host_ids)
    controller=NS(mem_pool_host=group)
    model=NS(sidecar_pool_specs=spec)
    backup,=route(model,phases.BACKUP_HOST,kv,{"window component":[sw]})
    assert not route(model,phases.BACKUP_HOST,kv,{})
    assert not route(model,phases.LOAD_BACK,kv,{})
    assert backup.keys==sw.keys and backup.keys!=kv.keys
    transfers=resolve_host(group,[sw,backup],primary_host_indices=kv.host_indices,primary_device_indices=kv.device_indices)
    ab=plan_l2(controller,kv.host_indices,kv.device_indices,transfers)[-1]
    assert ab.host_indices is host_ids and ab.device_indices is fresh
    assert ab.device_pool is pool.swa_kv_pool and ab.host_pool is entry.host_pool
    sw=transfer(name=names.SWA,host_indices=host_ids,keys=["swa"])
    load,=route(model,phases.LOAD_BACK,kv,{"window component":[sw]})
    transfers=resolve_device(controller,[sw,load],kv_device_indices=incoming,kv_host_indices=full_host)
    rb=plan_l2(controller,full_host,incoming,transfers)[-1]
    assert rb.device_indices is restored and rb.host_indices is host_ids
    assert rb.device_pool is ab.device_pool and rb.host_pool is ab.host_pool
    for i in range(5):
        for buf,d in zip(ab.device_pool.get_kv_buffer(ab.layer_mapper(i)),dense.get_kv_buffer(i)):
            host_plane=torch.empty((600,*buf.shape[1:]),dtype=buf.dtype)
            host_plane[ab.host_indices]=buf[ab.device_indices]
            buf[rb.device_indices]=host_plane[rb.host_indices]
            assert torch.equal(buf[rb.device_indices],d[incoming]);checked+=buf[rb.device_indices].numel()
    alloc.full_to_swa_index_mapping[old]=restored
    assert torch.equal(locate(old),restored)
    return dict(source_sha256=FILES,admitted_backends=["fa3","fa4"],decline_cases=declines,
        projected_append_cases=6,matched_and_preserved_bf16_words=checked,pin=PIN,
        five_layer_host_policy=spec[0].hit_policy,host_index_domain=spec[0].indices_from_pool,
        parent_target_config_unmodified=True,partial_fused_failure_and_prefix_mask=True,
        original_coverage_rebind_resolved=True,restored_five_layers=True,native_slot_plans=True,
        real_hicache_backup_restore_transfer_routing=True,
        scope="Native worker and configurator branches, native SWAKVPool wrapper and physical owner rebinding, original HiCache builder. Reduced BF16 projections, device stores and host copies run on CPU; CUDA allocation, DMA and FA kernels unmeasured.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path,nargs="?")
    parser.add_argument("--manifest",type=Path,default=HERE/"validation.json")
    parser.add_argument("--output",type=Path,default=HERE/"replay.json")
    args=parser.parse_args()
    validation=json.loads(args.manifest.read_bytes())
    patch=args.manifest.with_name("patch.diff")
    assert sha256(patch.read_bytes()).hexdigest()==validation["patch_sha256"]
    # Reconstitute the two changed modules exercised here and check a supplied
    # staging checkout byte-for-byte before connecting the native entry points.
    with tempfile.TemporaryDirectory(prefix="okc-private-swa-replay-") as td:
        scratch=Path(td)
        interesting=[SRT+KW,SRT+WR]
        for name in interesting:
            destination=scratch/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(subprocess.check_output(["git","-C",BASE,"show",f"{PIN}:{name}"]))
        subprocess.run(["git","apply","--no-index",*["--include="+p for p in interesting],str(patch.resolve())],
            cwd=scratch,check=True)
        if args.stage:
            for name in interesting:
                assert (args.stage/name).read_bytes()==(scratch/name).read_bytes(),name
        result=run(args.stage or scratch)
    result["patch_sha256"]=validation["patch_sha256"]
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result["matched_and_preserved_bf16_words"], "BF16 words")
