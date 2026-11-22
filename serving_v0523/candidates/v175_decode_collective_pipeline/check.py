"""Compare output-feature cohorts and two token microbatches of native verification."""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
CAND=HERE.parent
ROOT=HERE.parents[2]
TREE="b5a37c2923f8d1ab24c1c5cf7920a21e2c086774"
MIB=2**20
HBM=4.8e12
LAYERS=62
LOCAL_E=32


def half_counts(n,live):
    left=n//2
    return [(k,math.comb(left,k)*math.comb(n-left,live-k)/math.comb(n,live))
            for k in range(max(0,live-(n-left)), min(live,left)+1)]


def geometry(n,pairs,full_unpool_us):
    cap1,cap2=(8,8) if n<=32 else (16,32)
    def sweeps(count):
        return 2*math.ceil(count/cap1)+math.ceil(count/cap2)
    before=after=shared=0
    for live,number in pairs:
        before+=number*sweeps(live)
        for k,prob in half_counts(n,live):
            after+=number*prob*(sweeps(k)+sweeps(live-k))
            shared+=number*prob*(0<k<live)
    extra_mib=9*(after-before)
    transfer_us=extra_mib*MIB/HBM*1e6
    # Favorable to pipelining: give it the entire old unpool GEMM as hideable time.
    return dict(cap_w13_w2=[cap1,cap2],old_nominal_MiB=before*9,two_halves_nominal_MiB=after*9,
        expert_banks_used_in_both_halves=shared,extra_nominal_MiB=extra_mib,
        physical_time_if_extra_nominal_goes_to_HBM_us=transfer_us,
        additional_collective_us_needed_if_even_whole_unpool_hidden=max(0,transfer_us-full_unpool_us),
        absolute_weight_sweep_ratio=after/before)


def exhaustive_test():
    cases=0
    for n in range(2,13,2):
        for live in range(n+1):
            exact=Counter(sum(i<n//2 for i in pick) for pick in itertools.combinations(range(n),live))
            old=dict(half_counts(n,live))
            assert set(old)==set(exact) and all(old[k]==float(Fraction(v,math.comb(n,live))) for k,v in exact.items())
            cases+=math.comb(n,live)
    return cases


def pinned_path():
    model="python/sglang/srt/models/muse_spark.py"
    src=subprocess.check_output(["git","-C",str(ROOT/"serving_v0523/sglang"),"show",TREE+":"+model],text=True)
    funcs={n.name:ast.get_source_segment(src,n) for c in ast.parse(src).body if isinstance(c,ast.ClassDef)
           for n in c.body if isinstance(n,ast.FunctionDef) and n.name in {"_forward_tp","_reduce_and_unpool"}}
    assert funcs["_forward_tp"].index("output = self._forward_experts(")<funcs["_forward_tp"].index("return self._reduce_and_unpool(")
    assert funcs["_reduce_and_unpool"].index("tensor_model_parallel_all_reduce(output)")<funcs["_reduce_and_unpool"].index("gemm_local(comm")
    patch=(CAND/"v168_streamed_experts_sparse_return/patch.diff").read_text()
    assert "constexpr uint32_t tile_rows = Gate ? 16 : 32;" in patch and "constexpr uint32_t kRows = 8;" in patch
    assert "def _reduce_and_unpool" not in patch
    assert "Communication is left out: each all-reduce and all-gather is the identity" in (ROOT/"muse_decode_layer/TASK.md").read_text()
    return hashlib.sha256(src.encode()).hexdigest()


def physical_shape_proxies():
    ids={json.loads(f.read_text())["job_id"] for f in
         (ROOT/"experiments/candidates/muse_moe_w4a16/profiles").glob("*.json")}
    observed={n:{"baseline_raw_ns":[],"median_raw_ns":[]} for n in (32,256)}
    accepted=[]
    for jid in sorted(ids):
        f=ROOT/"experiments/runs"/(jid+".json")
        if not f.is_file(): continue
        result=json.loads(f.read_text())["response"]["result"]
        if result["gpu"]!="H200" or result["passed"] is not True: continue
        rows={x["case"]["bs"]:x for x in result["records"] if x["event"]=="benchmark"}
        if not all(n in rows for n in (16,32,128,256)): continue
        accepted.append(jid)
        for n, series in observed.items():
            for key,times in series.items():
                times.append((2*rows[n//2][key]-rows[n][key])/1000)
    def describe(times):
        return dict(count=len(times),min_us=min(times),median_us=statistics.median(times),
                    max_us=max(times),positive=sum(t>0 for t in times))
    return dict(job_ids=accepted,summary={n:{k:describe(v) for k,v in data.items()} for n,data in observed.items()},
                method="Two times the isolated smaller-row raw median minus one larger-row raw median in the same job; different generated input cases and no TP collective.")

def main():
    board=json.loads((ROOT/"experiments/runs/b5538a0dd25d46be.json").read_text())["response"]["result"]
    v64=json.loads((ROOT/"serving_v0523/runs/abd217baf0ed49fd.json").read_text())["result"]
    standalone=json.loads((ROOT/"experiments/iterations/20261104-moe-hardware-profile-recovered.json").read_text())["cases"]
    measurements={(r["case"]["m"],r["case"]["layer"]):r["baseline_raw_ns"]/1000
        for r in board["records"] if r["event"]=="benchmark"}
    results=[]
    for rung in v64["rungs"]:
        b=rung["concurrency"]; rows=b*8
        pool_us=measurements[rows,"pooler"]; unpool_us=measurements[rows,"unpooler"]
        period_ms=1000*rung["acceptance"]/rung["tok_s_per_user_p50"]
        two_proj_ms=LAYERS*(pool_us+unpool_us)/1000
        saved=[]
        for size in (2,4):
            # Relative to V142's full TP8 projection of both matrices. Each matrix is 64 MiB.
            old_send=(2*rows*(4096+8192))*7//8
            new_send=(2*rows*(4096+8192))*(size-1)//size
            extra=128*MIB//size-128*MIB//8
            saved.append(dict(size=size,added_cold_matrix_MiB=extra/MIB,logical_send_KiB_saved=(old_send-new_send)/1024,
                cold_HBM_us=extra/HBM*1e6,
                equal_time_comparison_one_direction_link_GB_s=HBM*(old_send-new_send)/extra/1e9))
        probs=[(k,LOCAL_E*math.comb(rows,k)*(1/32)**k*(31/32)**(rows-k)) for k in range(rows+1)]
        item=dict(backend_h200_raw_pooler_us=pool_us,backend_h200_raw_unpooler_us=unpool_us,
                  users=b,rows=rows,approx_v64_period_ms=period_ms,isolated_projection_budget_ms=two_proj_ms,
                  proxy_ratio_if_both_projections_free=period_ms/(period_ms-two_proj_ms),
                  cohorts=saved,uniform_microbatch=geometry(rows,probs,unpool_us))
        observed=next((c for c in standalone if c["M"]==rows),None)
        if observed:
            counts=observed["routing"]["local_counts"]
            item["standalone_H200_histogram_conditional_random_row_partition"]=geometry(rows,[(k,1) for k in counts],unpool_us)
        results.append(item)
    return dict(parent_source_sha256=pinned_path(),v64_tree=TREE,hardware_score_id="abd217baf0ed49fd",projection_job="b5538a0dd25d46be",
                exhaustive_small_combinations=exhaustive_test(),peak_HBM_bytes_per_second_assumption=HBM,release_physical_gpu_run=False,
                per_rung=results,archived_H200_half_size_proxies=physical_shape_proxies(),caveats=["Isolated projections are not an instrumented end-to-end serving profile.",
                "Expected half overlap for the standalone measured histogram assumes a uniformly randomized token permutation.",
                "Extra packed weight requests may hit L2; HBM times assume these requests miss, with peak HBM bandwidth.",
                "Logical ring send estimates do not encode a native multicast implementation or startup latency."])
if __name__=="__main__":
    print(json.dumps(main(),indent=2))
