"""Read and write all five draft layers through native FA4 metadata and SWA ownership."""
from argparse import ArgumentParser
import ast
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import torch

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parents[1]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
sys.path.insert(0, str(HERE.parent/"v248_shared_verify_raster"))
from independent_geometry import Pointer, TritonSubset

SRT = "python/sglang/srt/"
BACK = SRT+"layers/attention/flashattention_backend.py"
META = "python/sglang/kernels/ops/attention/metadata.py"
PLAN = SRT+"mem_cache/kv_loc_plan.py"
TRANS = SRT+"mem_cache/kv_index_translator.py"
POOL = SRT+"mem_cache/swa_memory_pool.py"
MHA = SRT+"mem_cache/memory_pool.py"
VIEWS = SRT+"mem_cache/layout/paged_view.py"
WORKER = SRT+"speculative/dflash_worker_v2.py"
COMP = SRT+"mem_cache/unified_cache/components/swa.py"
HOST = SRT+"mem_cache/pool_host/base.py"
ALLOC = SRT+"mem_cache/allocator/swa.py"
RUST = SRT+"mem_cache/rust_tree_core/adapter.py"


class Mode:
    EXTEND=object()
    def __init__(self,idle=False): self.idle=idle
    def is_extend(self): return not self.idle
    def is_target_verify(self): return not self.idle
    def is_decode(self): return False
    def is_decode_or_idle(self): return self.idle
    def is_draft_extend_v2(self): return False
    def is_dllm_extend(self): return False


class MetadataOps(TritonSubset):
    def load(self,ptr,mask=True,other=0,**kw):
        return super().load(ptr,mask,other)
    def store(self,ptr,val,mask=True,**kw):
        return super().store(ptr,val,mask)


def check(stage):
    cache={}
    manifest=json.loads((HERE/"validation.json").read_bytes())
    assert sha256((HERE/"patch.diff").read_bytes()).hexdigest()==manifest["patch_sha256"]
    def src(path):
        if path not in cache:
            p=stage/path
            cache[path]=p.read_text() if p.is_file() else subprocess.check_output(
                ["git","-C",str(SERVICE/"sglang"),"show",f"{PIN}:{path}"],text=True)
            expected=subprocess.check_output(["git","-C",str(stage),"show",manifest["tree"]+":"+path])
            assert cache[path].encode()==expected, path
        return cache[path]
    tl=MetadataOps()
    env=dict(torch=torch,dataclass=dataclass,tl=tl,threading=threading,ForwardMode=Mode,
        triton=NS(cdiv=lambda a,b:-(-a//b)),get_spec=lambda:NS(speculative_algorithm="DFLASH"),
        SpeculativeAlgorithm=NS(from_string=lambda _:NS(is_dflash=lambda:True)),IdSpaceKind=NS(FULL="full",SLIDING_WINDOW="swa"),
        maybe_create_verify_mask=lambda **kw:None,is_cp_active=lambda batch:False,
        AttentionType=NS(ENCODER_ONLY="enc",DECODER_BIDIRECTIONAL="bi"),_is_npu=False,
        get_parallel=lambda:NS(),fused_draft_host_allocator=lambda p:None,
        IdSpace=lambda **kw:NS(**({"read_v2p":None,"write":None}|kw)),
        get_allocator_from_storage=lambda t:None,
        assert_buffer_fits=lambda actual,limit,label: actual<=limit or (_ for _ in ()).throw(ValueError(label)))
    def bind(path,name,cls=None,strip=True):
        owner=ast.parse(src(path))
        if cls: owner=next(n for n in owner.body if getattr(n,"name",None)==cls)
        item=next(n for n in owner.body if getattr(n,"name",None)==name)
        if strip: item.decorator_list=[]
        exec("from __future__ import annotations\n"+ast.unparse(item),env)
        return env[name]
    def methods(path,cls,names):
        return {n:bind(path,n,cls) for n in names.split()}
    for n in ("UnifiedSWAAllocatorBase","UnifiedMambaTokenToKVPoolAllocator","DeepSeekV4TokenToKVPool",
              "MLATokenToKVPool","TokenSpan"):
        env[n]=type(n,(),{})
    for n in ("window_read_extent","pad_with_sink"): bind(PLAN,n)
    env["KVLocPlan"]=type("KVLocPlan",(),methods(PLAN,"KVLocPlan",
        "__init__ write_ids _window_in _cols bind bind_replay virtual_write_ids is_translated_for"))
    attrs=methods(TRANS,"KVIndexTranslator","__init__ space plan own_plan bind_runner_slots write_ids _reads_translated")
    for n in ("window_kv_kind","reads_are_translated"):
        attrs[n]=property(bind(TRANS,n,"KVIndexTranslator"))
    Translator=type("Translator",(),attrs)
    env["KVWriteLoc"]=bind(MHA,"KVWriteLoc",strip=False)
    bind(MHA,"unwrap_write_loc")
    for n in ("paged_view","paged_kv_view"):bind(VIEWS,n)
    Pool=type("SWAKVPool",(),methods(POOL,"SWAKVPool","get_kv_buffer _wait_for_layer set_kv_buffer translate_loc_from_full_to_swa register_mapping"))
    env["BaseSWAKVPool"]=Pool
    words=0
    class Bank:
        def __init__(self):
            self.k=torch.full((5,97,1,4),float("nan"),dtype=torch.bfloat16)
            self.v=self.k.clone()
        def get_kv_buffer(self,i):return self.k[i],self.v[i]
        def set_kv_buffer(self,layer,ids,k,v,*rest,layer_id_override):
            self.k[layer_id_override,ids]=k
            self.v[layer_id_override,ids]=v
    pool=Pool();pool.layer_transfer_counter=None
    pool.layers_mapping={i:(i,True) for i in range(5)};pool.swa_kv_pool=Bank();pool.dflash_private_swa=True
    # Accesses to canonical FULL plane would fail immediately, including fallback.
    pool.full_kv_pool=object()
    mapping=torch.zeros(513,dtype=torch.int64);mapping[-1]=-1
    pool.register_mapping(mapping)
    raw=torch.zeros((2,40),dtype=torch.int32)
    raw[0,:16]=torch.arange(201,217);raw[1,:16]=torch.arange(301,317)
    reader=Translator(req_to_token=raw,token_to_kv_pool_allocator=object(),token_to_kv_pool=pool,
                      page_size=1,device="cpu",is_draft_worker=True)
    assert not reader.is_translating and reader.window_kv_kind=="swa"
    plan_source=Translator(req_to_token=raw,token_to_kv_pool_allocator=object(),token_to_kv_pool=object(),
        page_size=1,device="cpu",is_draft_worker=False)
    worker=NS(draft_model_runner=NS(token_to_kv_pool=pool,kv_index_translator=reader))
    writeids=bind(WORKER,"_draft_write_ids","DFlashWorkerV2")
    assert bind(WORKER,"_context_kv_pool","DFlashWorkerV2")(worker) is pool.swa_kv_pool
    order=raw[:,:16].long().reshape(-1)
    old=torch.arange(1,33);fresh=old+32
    mapping[order]=old
    bind(META,"_resolve_metadata_page_ids")
    fn=bind(META,"_fused_decode_metadata_kernel")
    count=[0]
    class Kernel:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                kwargs.pop("num_warps",None)
                kwargs.pop("num_stages",None)
                args=[Pointer(t.numpy().reshape(-1)) if isinstance(t,torch.Tensor) else t for t in args]
                for b in range(grid[0]):
                    for c in range(grid[1]):
                        tl.pid=(b,c);fn(*args,**kwargs);count[0]+=1
            return launch
    env["_fused_decode_metadata_kernel"]=Kernel()
    bind(META,"normal_decode_set_metadata")
    bind(BACK,"FlashAttentionMetadata",strip=False)
    backend_names=("init_cuda_graph_state init_forward_metadata_out_graph _init_full_cg_decode_metadata "
        "_bind_metadata_buffers _set_decode_page_metadata _apply_cuda_graph_metadata _capture_unified_verify_metadata "
        "init_forward_metadata_in_graph _get_swa_write_locations _mxfp8_sf_kwargs forward_extend get_paged_mha_kv_cache "
        "init_forward_metadata _spec_read_seq_len_delta")
    be=type("Backend",(),methods(BACK,"FlashAttentionBackend",backend_names))()
    be.__dict__.update(device="cpu",max_context_len=40,page_size=1,_get_scheduler_metadata=None,local_attn_builder=None,
        use_sliding_window_kv_pool=True,topk=1,has_swa=True,needs_cpu_seq_lens=False,head_dim=4,num_kv_heads=1,
        _decode_uses_static_max_seqlen_k=True,is_encoder_decoder=False,is_prefill_aware_swa=False,_kv_shard_pool=None,skip_prefill=False,
        speculative_step_id=0,speculative_num_draft_tokens=8,speculative_num_steps=1,is_draft_runner=True,
        kv_index_translator=reader,_hopper_verify_sms=0,use_mla=False,req_to_token=raw,sliding_window_size=7,
        fa_impl_ver=4,num_splits=1,fa_skip_kv_cache=False,kv_cache_is_mxfp8=False,
        _disable_scheduler_metadata_precompute=False,token_to_kv_pool=pool,req_to_token_pool=NS(req_to_token=raw))
    be.prepare_paged_mha_query=lambda q,qr,kr,*a,**kw:(q,qr,kr,None,None)
    be._maybe_init_local_attn_metadata=lambda *a:None
    be.init_cuda_graph_state(2,16)
    cfg=NS(draft_token_num=8,num_tokens_per_req=8,ragged_verify_layout=None)
    def batch(idle=False):
        return NS(forward_mode=Mode(idle),batch_size=2,req_pool_indices=torch.arange(2),seq_lens=torch.tensor([8,8],dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8,8],dtype=torch.int32),spec_info=None if idle else cfg,encoder_lens=None,
            out_cache_loc=torch.zeros(2 if idle else 16,dtype=torch.long),positions=torch.arange(2 if idle else 16),
            dcp_kv_mask=None,spec_algorithm=NS(is_dflash=lambda:True),_attn_output=None)
    graph=batch();reader.bind_runner_slots(graph)
    be.init_forward_metadata_out_graph(graph,in_capture=True)
    md=be.forward_metadata
    assert md.swa_out_cache_loc.tolist()==[0]*16
    pointers=(md.swa_out_cache_loc.data_ptr(),md.swa_page_table.data_ptr())
    frozen=md.swa_page_table[:,:16].clone()
    idle=batch(True);reader.bind_runner_slots(idle)
    be.init_forward_metadata_out_graph(idle,in_capture=True)
    assert not be.forward_metadata.swa_out_cache_loc.any()
    latest={}
    checks=[]
    def fake_fa(**k):
        nonlocal words
        assert k["window_size"]==latest["window"]
        assert k["page_table"] is be.forward_metadata.swa_page_table and k["cache_seqlens"].tolist()==[16,16]
        actual=k["v_cache"].reshape(-1,1,4)[k["page_table"][:,:16].long()]
        assert torch.equal(actual,latest["values"]), "FA4 followed a stale or canonical FULL slot"
        words+=actual.numel()
        return k["q"]
    env["flash_attn_with_kvcache"]=fake_fa
    generator=torch.Generator().manual_seed(2492)
    vals=torch.randn((5,2,16,1,4),generator=generator,dtype=torch.bfloat16)
    def run(mode,phys,payload,*,bidirectional=True):
        fw=batch()
        write=raw[:,8:16].reshape(-1).long()
        plan=plan_source.plan(req_pool_indices=fw.req_pool_indices,seq_lens=fw.seq_lens,seq_lens_cpu=fw.seq_lens_cpu,write_virtual=write,read_extent=8)
        plan.bind(fw,reader)
        assert torch.equal(writeids(worker,plan),mapping[write])
        if mode=="graph":
            be.init_forward_metadata_out_graph(fw,in_capture=False)
            assert be.forward_metadata is md and pointers==(md.swa_out_cache_loc.data_ptr(),md.swa_page_table.data_ptr())
            assert not be._capture_unified_verify_metadata(fw)
            before=md.swa_page_table[:,:16].clone()
            be.init_forward_metadata_in_graph(fw)
            assert torch.equal(before,md.swa_page_table[:,:16])
        else: be.init_forward_metadata(fw)
        assert torch.equal(be.forward_metadata.swa_out_cache_loc.reshape(2,8),phys.reshape(2,16)[:,8:])
        for i in range(5):
            layer=NS(layer_id=i,sliding_window_size=7,tp_k_head_num=1,tp_v_head_num=1,tp_q_head_num=1,
                head_dim=4,v_head_dim=4,k_scale=None,v_scale=None,logit_cap=0.,attn_type="enc" if bidirectional else "decoder",is_cross_attention=False,scaling=.5)
            latest["values"]=payload[i]
            latest["window"]=(7,7) if bidirectional else (7,0)
            v=payload[i,:,8:].reshape(16,1,4)
            be.forward_extend(v.clone(),v+1,v,layer,fw)
        checks.append(mode)
    # Populate Old and test. Restore on a different physical SWA page while
    # preserving the same full ids. Poison old physical pages to expose a stale read.
    for i in range(5):
        pool.swa_kv_pool.v[i,old]=vals[i].reshape(32,1,4)
    run("graph",old,vals)
    snapshot=pool.swa_kv_pool.v[:,old].clone()
    pool.swa_kv_pool.v[:,old]=float("nan")
    mapping[order]=fresh
    # Drive the real L2 engine loop and final-prefill wait using an event-log
    # stand-in for GPU streams. Both raw and graph restore consumers must be
    # ordered after every live draft layer; layer 61 is the target's last.
    env["envs"]=NS(SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES=NS(get=lambda:0))
    calls=[]
    class HostSource:
        def load_to_device_per_layer_physical(self,dest,hids,dids,i,io_backend,is_draft):
            calls.append(("copy",self.label,i))
            if self.label=="sidecar":
                dest.v[i,dids]=snapshot[i,hids]
    target_host=HostSource();target_host.label="target"
    draft_host=HostSource();draft_host.label="sidecar"
    target_xfer=NS(layer_mapper=lambda i:i,host_pool=target_host,device_pool=object(),host_indices=None,
        device_indices=None,is_draft=False)
    draft_xfer=NS(layer_mapper=lambda i:i if i<5 else None,host_pool=draft_host,device_pool=pool.swa_kv_pool,
        host_indices=torch.arange(32),device_indices=fresh,is_draft=False)
    @contextmanager
    def submission(transfers,*args):
        yield transfers,object()
    engine=NS(_submission=submission,host_to_device_stream=None,io_backend="kernel")
    engine_submit=bind(SRT+"mem_cache/l2_transfer.py","submit_host_to_device","L2TransferEngine")
    engine_submit(engine,[target_xfer,draft_xfer],transfer_layer_id_max=62,
        on_layer_done=lambda i:calls.append(("event",i)))
    for i in range(5):
        assert calls.index(("copy","sidecar",i)) < calls.index(("event",i))
    counter=NS(consumer_index=0,num_layers=62,wait_until=lambda i:calls.append(("wait",i)))
    model=NS(token_to_kv_pool=NS(layer_transfer_counter=counter))
    wait=bind(SRT+"model_executor/runner/prefill_cuda_graph_runner.py","_wait_for_layer_loads","PrefillCudaGraphRunner")
    wait(NS(_is_full_backend=True,model_runner=model))
    assert calls[-1]==("wait",61) and calls.index(("event",61))<len(calls)-1
    target_pool=NS(layer_transfer_counter=counter,start_layer=0)
    per_layer=bind(POOL,"_wait_for_layer","SWAKVPool")
    for i in range(62):per_layer(target_pool,i)
    assert [("wait",i) for i in range(62)]==calls[-62:]
    run("graph",fresh,vals)
    assert not torch.equal(frozen,md.swa_page_table[:,:16])
    run("eager",fresh,vals)
    run("eager",fresh,vals,bidirectional=False)
    # Original recover-with-locked-full Rust action publishes incoming SWA IDs
    # under kept full slots. Feed its output straight to the same graph pointers.
    incoming=order+64
    assert not torch.isin(order,incoming).any()
    mapping[order]=0;mapping[incoming]=old
    for i in range(5):pool.swa_kv_pool.v[i,old]=snapshot[i]+2
    alloc=NS(translate_swa_indices_for_transfer=lambda ids:mapping[ids],
        clear_full_to_swa_mapping=lambda ids:mapping.index_fill_(0,ids,0),
        set_full_to_swa_mapping=lambda full,phys:mapping.index_copy_(0,full,phys),
        free_full_segment=lambda ids,start_pos:None)
    env["RecoverSWAWithLockedFull"]=namedtuple("RecoverSWAWithLockedFull","node_id kept_full incoming_full")
    for n in ("FreeComponentDeviceSlot","FreeComponentHostSlot","RebuildFullToSWAMapping","RecoverEvictedSWA","SWARebuild"):
        env[n]=type(n,(),{})
    c=type("Component",(),methods(COMP,"SWAComponent","_set_swa _translate_full_to_swa _page_pairs _transfer_swa_pages apply_component_action"))()
    c.cache=NS(token_to_kv_pool_allocator=alloc);c._unified_allocator=lambda:None
    published=[]
    c.tree_core=NS(set_component_device_value=lambda node,kind,ids:published.append((node,kind,ids)))
    c.component_type="swa"
    tagged=bind(RUST,"_cache_action_from_tagged")
    c.apply_component_action(tagged(("recover_swa_with_locked_full",7,order,incoming)))
    assert published[0][:2]==(7,"swa") and torch.equal(published[0][2],old)
    assert not mapping[incoming].any()
    run("graph",old,vals+2)
    # Host-capacity base constructor, exact deployed page-one path; skip OS pinned allocation.
    host=NS(clear=lambda:None)
    init=bind(HOST,"__init__","HostKVCache")
    target=600
    init(host,device_capacity_tokens=300,dtype=torch.bfloat16,size_per_token=2560,start_layer=0,end_layer=5,
        host_to_device_ratio=target/300,host_size=0,page_size=1,layout="page_first",pin_memory=True,
        device="cpu",is_dummy=True)
    assert host.size==target+1
    # Replay V243's real shortened-checkpoint and later interior fork. Its
    # surviving indices must resolve identically on target and private draft.
    sys.path.insert(0,str(HERE.parent/"v243_draft_prefix_coverage"))
    import coverage_replay
    files=("managers/scheduler.py","mem_cache/common.py","mem_cache/unified_radix_cache.py",
           "mem_cache/unified_cache/components/swa.py","mem_cache/unified_cache/unified_tree_core.py")
    maker=coverage_replay.pinned_reader()
    fork=coverage_replay.fork_witness(maker,{f:src(SRT+f) for f in files})["native_branch_checkpoint"]
    lo,hi=fork["native_live_swa_span"]
    endpoint=fork["later_divergence_endpoint"];needed=fork["old_rows_needed_at_next_draft"]
    assert needed==1791 and lo<=endpoint-needed
    ids=torch.arange(16384);swa=torch.zeros(16385,dtype=torch.int64)
    swa[lo:hi]=torch.arange(hi-lo,0,-1)
    pool.register_mapping(swa)
    window=plan_source.plan(req_pool_indices=None,seq_lens=None,seq_lens_cpu=None,write_virtual=ids)
    kept_ids=writeids(worker,window)
    bank=torch.full((hi-lo+1,5,2),float("nan"),dtype=torch.bfloat16)
    payload=torch.randn(16384,5,2,generator=generator,dtype=torch.bfloat16)
    live=kept_ids>0
    bank[kept_ids[live]]=payload[live]
    logical=ids[endpoint-needed:endpoint]
    for tier in ("value","host_value"):
        assert fork["later_acceptance_by_tier"][tier]["hierarchy_accepts"]
        physical=swa[logical]
        assert torch.equal(bank[physical],payload[logical])
    return dict(pin=PIN,tree=manifest["tree"],patch_sha256=manifest["patch_sha256"],native_source_hashes={p:sha256(v.encode()).hexdigest() for p,v in cache.items()},
        tests=checks,full_graph_table_and_write_buffer_addresses_preserved=True,capture_idle_sinks=True,
        native_Triton_metadata_programs=count[0],five_layer_FA4_reference_BF16_values_checked=words,
        host_rows_for_600_target_SWA_host_rows=host.size,sidecar_host_loads_before_corresponding_target_events=5,
        full_graph_target_wait_last_layer=61,per_layer_target_accessor_waits=62,earlier_checkpoint_old_draft_rows_retained=needed,
        earlier_checkpoint_device_and_hierarchical_matches=True,
        scope="Native source bodies run on CPU, including the Triton metadata body; BF16 bank storage and FA kernel boundary are instrumented. CUDA, DMA and device attention arithmetic unmeasured.")

if __name__=="__main__":
    ap=ArgumentParser();ap.add_argument("stage",type=Path);args=ap.parse_args()
    torch.set_num_threads(1);print(json.dumps(check(args.stage),indent=2))
