"""Read the scored public receipt and pinned native cache paths. Does not launch a server."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from fnmatch import fnmatchcase
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
RUN = "abd217baf0ed49fd"

def source(path):
    return subprocess.check_output(["git", "-C", str(ROOT / "sglang"), "show", f"{PIN}:{path}"], text=True)

def method(src, name, **env):
    node = next(f for f in ast.walk(ast.parse(src)) if isinstance(f, ast.FunctionDef) and f.name == name)
    node.decorator_list = []
    exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return env[name]

def run():
    receipt = json.loads((ROOT / "runs" / (RUN + ".json")).read_text())
    res = receipt["result"]
    log = res["startup_log"]
    assert log["cut"] == "first request" and log["truncated"]
    text = log["text"]
    assert "RustUnifiedTreeCore" in text and "hicache_attached=True" in text
    assert res["passed"] and res["rungs"] and res["patch"]["sha256"] == sha256(
        (ROOT/"candidates/v64_humming_grouped_sequence_parallel/patch.diff").read_bytes()).hexdigest()
    assert not any("mem_cache/" in p or "hicache_hook" in p for p in res["patch"]["files"])
    pattern = r"Allocating (full|swa|draft) hierarchical KV host pool: (\d+) tokens, ([0-9.]+) GB"
    pools = {name: (int(count), float(gb)) for name,count,gb in re.findall(pattern,text)}
    assert set(pools) == {"full","swa","draft"}
    # Infer whole row bytes from logged allocation rounded to the nearest 0.01 GB (decimal).
    rows = {name: round(gb*1e9/n) for name,(n,gb) in pools.items()}
    assert rows == dict(full=4096,swa=11776,draft=2560)
    gpu_full, gpu_swa = map(int,re.search(r"full_layer_tokens=(\d+), swa_layer_tokens=(\d+)",text).groups())

    m="python/sglang/srt/mem_cache/"
    srt="python/sglang/srt/"
    cache=source(m+"unified_radix_cache.py")
    core=source(m+"unified_cache/unified_tree_core.py")
    rust=source("rust/sglang-radix-tree/src/unified_tree_core.rs")
    bridge=source(m+"l2_transfer.py")
    flags=source(srt+"arg_groups/fields/memory.py")
    declared={n.target.id: n.value.value for n in ast.walk(ast.parse(flags))
              if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)
              and isinstance(n.value,ast.Constant) and n.target.id.startswith("hicache_")}
    assert declared["hicache_write_policy"]=="write_through" and declared["hicache_host_memory_mode"]=="cache"
    assert declared["hicache_io_backend"]=="kernel" and declared["hicache_mem_layout"]=="page_first"
    assert '1 if get_memory().hicache_write_policy == "write_through" else 2' in cache
    validator=method(source(srt+"arg_groups/hicache_hook.py"),"validate_hicache_host_memory_mode",resolving_view=lambda x:x)
    for policy in ("write_back","write_through","write_through_selective"):
        validator(NS(hicache_host_memory_mode="cache",hicache_size=160,hicache_write_policy=policy))
    fixed=re.findall(r"--[\w*-]+",(ROOT/"flags").read_text().split("Fixed, so refused")[1])
    assert not any(fnmatchcase("--hicache-write-policy",p) for p in fixed)
    assert "if self.is_write_back {\n            return false;" in rust
    assert "if state.is_new_leaf {" in rust and "&& self.inc_hit_count_and_check_(target_node_id);" in rust
    check_hit=method(core,"_inc_hit_count_and_check")
    cases=0
    for writeback, evicted, backed, enabled, threshold, hits in product((False,True),(False,True),(False,True),
                                                                     (False,True),(1,2),(0,1,2)):
        node=NS(evicted=evicted,backuped=backed,hit_count=hits)
        settings=NS(is_write_back=writeback,enable_external_cache_linker=False,enable_hicache=enabled,
                    write_through_threshold=threshold)
        actual=check_hit(settings,node)
        assert actual == (not writeback and not evicted and not backed and enabled and hits+1>=threshold)
        cases+=1
    do_load=method(cache,"init_load_back")
    calls=[]
    for evicted,host,swa,mamba in product((0,1),repeat=4):
        tree=NS(is_full_device_evicted=lambda nid:evicted)
        obj=NS(buffer_pipeline=None,linker=None,tree_core=tree,load_back=lambda *a,**kw:calls.append(1))
        params=NS(best_match_node=10,mem_quota=None,host_hit_length=host,
                  req=NS(last_node=9,swa_host_hit_length=swa,mamba_host_hit_length=mamba))
        before=len(calls);assert do_load(obj,params)==(0,9)
        assert len(calls)-before==bool(evicted or host or swa or mamba)

    @contextmanager
    def scope(transfers,*args):
        yield transfers,object()
    done=[];loads=[];writes=[]
    engine=NS(_submission=scope,host_to_device_stream=None,device_to_host_stream=None,io_backend="kernel")
    bank=NS(load_to_device_per_layer_physical=lambda pool,hi,di,i,io,is_draft:loads.append((pool,i)),
            backup_from_device_all_layer_physical=lambda pool,hi,di,io:writes.append(pool))
    # Interleave two target pools and attach an independent five-layer draft sidecar.
    full_global=list(range(0,62,4));swa_global=[i for i in range(62) if i not in full_global]
    def transfer(name,layers):
        return NS(layer_mapper=lambda i:layers.index(i) if i in layers else None,device_pool=name,
                  host_pool=bank,host_indices=None,device_indices=None,is_draft=name=="draft")
    transfers=[transfer("full",full_global),transfer("swa",swa_global),transfer("draft",list(range(5)))]
    d2h=method(bridge,"submit_device_to_host"); d2h(engine,transfers)
    envs=NS(SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES=NS(get=lambda:0))
    h2d=method(bridge,"submit_host_to_device",envs=envs);h2d(engine,transfers,transfer_layer_id_max=62,on_layer_done=done.append)
    assert writes==["full","swa","draft"] and len(set(loads))==67 and done==list(range(62))

    archive_points=[]
    for path in (ROOT/"runs").glob("*.json"):
        if not re.fullmatch(r"[0-9a-f]{16}",path.stem):
            continue
        archived=json.loads(path.read_text()).get("result") or {}
        if isinstance(archived,dict) and archived.get("rungs"):
            archive_points.append(archived)
    archive_cuts=sorted({r.get("startup_log",{}).get("cut","unknown") for r in archive_points})
    reported_cache_metrics=sorted({k for r in archive_points for rung in r["rungs"] for k in rung
                                   if "cache" in k or "transfer" in k})
    fwd=source(srt+"layers/attention/flashattention_backend.py")
    assert "metadata.swa_page_table =" in fwd and "page_table = metadata.swa_page_table" in fwd
    assert "Every rank must enter the all_reduce below" in cache and "digest == -count_values[-1]" in cache
    first=res["shape"]["first_turn"];turns=res["shape"]["turns"]
    per_user_example=first+(turns-1)*(res["shape"]["new_per_turn"]+res["shape"]["output_mean"])
    return dict(
        archived_receipts_with_rungs=len(archive_points), archived_log_cut_types=archive_cuts,
        archived_cache_metrics=reported_cache_metrics,
        pinned_source=PIN,scored_v64_receipt=RUN,source_sha=res["patch"]["sha256"],
        recorded_pool_rows=pools,bytes_per_rank_per_slot=rows,full_gpu_slots_per_rank=gpu_full,
        swa_gpu_slots_per_rank=gpu_swa,observed_log_cut=log["cut"],captured_log_truncated=log["truncated"],
        public_points=[{k:x[k] for k in ("concurrency","cache_hit","ttft_first_p90_s","ttft_later_p90_s")}
                       for x in res["rungs"]],
        post_warmup_host_hit_fraction=None,post_warmup_h2d_bytes=None,post_warmup_d2h_time=None,
        example_full_draft_backup_bytes=first*(rows["full"]+rows["draft"]),
        example_extra_swa_2048_bytes=2048*rows["swa"],example_extra_swa_all_bytes=first*rows["swa"],
        illustrative_active_slots_32_users_13_turns=32*per_user_example,
        passed_checks=dict(source_equivalent_python_policy_cases=cases,native_loadback_combinations=16,
                           native_engine_d2h_pool_dispatches=len(writes),native_engine_h2d_layer_dispatches=len(loads),
                           native_h2d_readiness_events=len(done),native_cache_mode_resolution_cases=3,
                           fixed_flag_patterns=len(fixed)),
        hypothetical_flags=receipt["serving_flags"]|{"--hicache-write-policy":"write_back"},
        issued_check=False)

if __name__=="__main__":
    print(json.dumps(run(),indent=2))
