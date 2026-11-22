"""Reconcile cache-checkpoint opportunities and conditional host-transfer costs."""
import json
import math
import runpy
from pathlib import Path
from verify import HERE, PIN, candidate, digest
ROOT=HERE.parents[2]
helper=ROOT/"experiments/iterations/20261119-serving-cold-wave.py"
cold=runpy.run_path(str(helper))["cold_wave"]
prev=HERE.parent/"v244_cold_prompt_scaling/check.py"
archives=runpy.run_path(str(prev))["ARCHIVES"]
match=HERE.parent/"v288_paired_cold_and_verify_intervals/check.json"
intervals=json.loads(match.read_bytes())["matched_cold_summary"]
results=[]
for name,dirname,stem in archives:
    folder=ROOT/"experiments/modal_node"/dirname
    log=folder/(stem+"server.log");client=folder/(stem+"load.json");server=folder/(stem+"server_info.json")
    args=json.loads(server.read_bytes())["internal_states"][0]
    wanted={"enable_hierarchical_cache":True,"hicache_write_policy":"write_through","hicache_mem_layout":"page_first",
            "hicache_io_backend":"kernel","hicache_storage_backend":None,"page_size":1,"enable_metrics":False}
    assert {k:args[k] for k in wanted}==wanted
    info=json.loads(client.read_bytes())
    sections=log.read_text().split("Cache flushed successfully!")[1:]
    waves=[]
    for (users,i),section in zip(((r["concurrency"],j) for r in info["rungs"] for j in range(r["reps"])),sections):
        requests=[x for x in info["requests"] if (x["load"],x["rep"],x["turn"])==(users,i,0)]
        events=cold(section,requests)
        solo=users==1
        if solo:
            assert all(e["running"]==e["cached"]==0 and e["seqs"]==1 for e in events)
            assert all(e["new"]==16384 for e in events[:-1])
        waves.append(dict(users=users,rep=i,prompt_forwards=len(events),fresh_rows=sum(e["new"] for e in events),
            cached_rows=sum(e["cached"] for e in events),elapsed_to_last_first_text_s=max(x["t0"]+x["ttft"] for x in requests),
            solo_ongoing_checkpoints=len(events)-1 if solo else None,solo_ongoing_fresh_rows=sum(e["new"] for e in events[:-1]) if solo else None))
    results.append(dict(build=name,arguments=wanted,waves=waves,sha256={p.name:digest(p.read_bytes()) for p in (client,log,server)}))
b=16384;per=(4096,11776,2560)
scenarios=[]
for swa in (0,2048,16384):
    n=(b,swa,b)
    total=sum(k*c for k,c in zip(n,per))
    chunks=sum(math.ceil(k/64) for k in n);calls=sum(2*k for k in n)
    ops=sum(k>0 for k in n)
    scenarios.append(dict(full_swa_draft_rows=n,bytes_per_rank=total,native_relayouts=chunks,
        native_page_D2H_calls=calls,v154_contiguous_D2H_calls=chunks*2,
        mapped_KV_kernels=ops,uploaded_slot_index_bytes_int64=sum(n)*8,
        ideal_link_ms_at_assumed_oneway_GBps={str(g):1000*total/(g*1e9) for g in (16,32,64)},
        removed_serial_microseconds_per_native_small_copy_for_chunk_speedup={
            str(users):{str(f):intervals[str(users)]["new_mean_s"]*(1-1/f)*1e6/calls for f in (2,3,4)} for users in (1,4)}))
paths={
 "scheduler":"managers/scheduler.py","policy":"managers/schedule_policy.py","req":"managers/schedule_batch.py",
 "checkpoint":"mem_cache/common.py","radix":"mem_cache/unified_radix_cache.py","hybrid":"mem_cache/hybrid_cache/hybrid_cache_controller.py",
 "engine":"mem_cache/l2_transfer.py","controller":"managers/cache_controller.py","pool":"mem_cache/pool_host/common.py"}
with candidate() as (_,tree,git):
    sources={k:git("show",tree+":python/sglang/srt/"+p).decode() for k,p in paths.items()}
    rust=git("show",tree+":rust/sglang-radix-tree/src/unified_tree_core.rs").decode()
    def ordered(src,*keys):
        pos=-1
        for key in keys:pos=src.index(key,pos+1)
    ordered(sources['scheduler'],"self.chunked_req.init_next_round_input()","self.chunked_req = adder.add_chunked_req","req.init_next_round_input(self.tree_cache)")
    c=sources['policy'].split("    def add_chunked_req(")[1].split("\n    def ")[0];assert "init_load_back" not in c
    ordered(sources['checkpoint'],"def checkpoint_kv_cache(","tree_cache.checkpoint(req, up_to=req.extend_end)","req.prefix_len = req.extend_end")
    ordered(rust,"fn should_backup_after_insert_(","if self.is_write_back","if state.is_new_leaf","state.aligned_key_len > state.inserted_len","self.inc_hit_count_and_check_")
    ordered(sources['engine'],"with device_module.stream(stream):","start_event.wait(stream)","yield prepared, completion","ack_finish.record()")
    ordered(sources['controller'],"self.l2_transfer_engine.device_to_host_stream.wait_stream(","self.l2_transfer_engine.submit_device_to_host(")
    ordered(sources['radix'],"if not ack.finish_event.query()","cc.ack_write_queue.pop(0)")
    assert "registered_ranges.append((ptr, size))" in sources["pool"]
    aot=git("show",tree+":python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu").decode()
    assert "cudaHostGetDevicePointer(&device_ptr, ptr, 0)" in aot and "OptionalCUDAGuard device_guard(target_device)" in aot
official=ROOT/"serving_v0523/runs/04b01f958aa5487c.json";raw=json.loads(official.read_bytes())
assert raw["serving_flags"]==json.loads((HERE/"flags.json").read_bytes())
print(json.dumps(dict(waves=results,copy_scenarios=scenarios,source_tree=tree,
    native_sha256={p:digest(s.encode()) for p,s in {**sources,"rust_new_leaf":rust,"pointer_resolver":aot}.items()},
    matched_chunk_seconds=intervals,trace_helpers={p.name:digest(p.read_bytes()) for p in (helper,prev,match,official)},
    official_source_pool=raw["result"]["launch_settings"],official_V68_rungs=raw["result"]["rungs"],
    scope="Client and scheduler elapsed time, released transfer semantics and conditional per-pool arithmetic. Actual writes, host submission, transfer overlap, device mappings and H200 service gain are not timed."),indent=2))
