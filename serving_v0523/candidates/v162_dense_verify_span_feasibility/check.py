"""Native FA3 geometry, captured-pointer lifetime, H200 rows and one-user sensitivity."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GIT = ROOT / "serving_v0523/sglang"
SG = "297e54fd105e974cdf401e5b4d20253a22c16f90"
FA = "f89bc2306632d1ec5f97b014dded4254f5b4a907"
SRT = "python/sglang/srt/"
BASE = f"https://raw.githubusercontent.com/sgl-project/sgl-attn/{FA}/hopper/"


def method(relpath, cls, name):
    raw = subprocess.check_output(["git", "-C", GIT, "show", f"{SG}:{SRT}{relpath}"], text=True)
    nodes = ast.parse(raw).body
    if cls is not None:
        nodes = next(n.body for n in nodes if getattr(n, "name", "") == cls)
    return next(n for n in nodes if getattr(n, "name", "") == name)


def cpp_function(text, name):
    i = text.index(name)
    start = text.rindex("\n", 0, i) + 1
    pos = text.index("{", i)
    balance = 1
    for j in range(pos + 1, len(text)):
        balance += (text[j] == "{") - (text[j] == "}")
        if balance == 0:
            return text[start:j+1]
    raise ValueError(name)


def native():
    names = ("flash_api.cpp","tile_size.h","heuristics.h","seqlen.h","mainloop_fwd_sm90_tma_gmma_ws.hpp","flash_prepare_scheduler.cu")
    files = {}
    for name in names:
        with urlopen(BASE+name,timeout=15) as response:
            files[name] = response.read().decode()
    api = files["flash_api.cpp"]
    loop = files["mainloop_fwd_sm90_tma_gmma_ws.hpp"]
    assert "PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma" in api
    assert "static constexpr bool Use_TMA_KV = !PagedKVNonTMA;" in loop
    assert "if (!PagedKVNonTMA && args.ptr_pagetable != nullptr)" in loop
    assert "copy(params.tma_load_V.with(" in loop
    assert "seqlen = seqlen - leftpad_k + seqlen_new;" in files["flash_prepare_scheduler.cu"]
    c = """
#include <algorithm>
#include <cassert>
#include <iostream>
#include <tuple>
#define CUTLASS_DEVICE
struct Flash_fwd_params {
    int arch=90, d=64, d_rounded=64, dv_rounded=64, page_size=1, seqlen_q=8, h=16, h_k=2;
    bool page_table=true, leftpad_k=false, knew_ptr=false, is_local=false, is_causal=true, is_e4m3=true;
    float softcap=0;
};
"""
    c += cpp_function(files["tile_size.h"],"tile_size_fwd_sm90(")
    c += "\n" + cpp_function(files["heuristics.h"],"use_one_mma_wg(")
    c += "\n" + cpp_function(api,"get_pagedkv_tma(")
    c += "\ntemplate <bool Varlen, bool AppendKV>\n" + cpp_function(files["seqlen.h"],"struct SeqlenInfoQKNewK") + ";"
    c += """
int main(){
    Flash_fwd_params p;
    int page_cases=0,count=0;
    for(int page:{1,16,32,64,128,160,320,640,1280}){
        p.page_size=page;
        for(bool local:{false,true}){
            p.is_local=local;
            assert(!get_pagedkv_tma(p));
            ++page_cases;
        }
    }
    p.is_local=false;
    auto tile=tile_size_fwd_sm90(64,64,true,false,1,false,false,false,use_one_mma_wg(p));
    for(int first:{0,1,31,160,8369,3410000}){
        for(int live:{8,73,159,160,161,2049,74160,80000}){
            if(first+live>3419325)continue;
            int used=first+live;
            int cu_q[2]={0,8};
            SeqlenInfoQKNewK<true,false> s(0,8,3419325,0,cu_q,nullptr,nullptr,nullptr,&used,&first,nullptr);
            assert(s.offset_k==first && s.seqlen_k==live && s.seqlen_q==8);
            ++count;
        }
    }
    std::cout << page_cases << ' ' << count << ' ' << std::get<0>(tile) << ' ' << std::get<1>(tile);
}
"""
    with tempfile.TemporaryDirectory(prefix="v162-native-") as folder:
        tmp = Path(folder)
        (tmp / "ref.cpp").write_text(c)
        subprocess.run(["c++","-std=c++17","-O2",str(tmp/"ref.cpp"),"-o",str(tmp/"test")],check=True)
        pages, offsets, m, n = map(int,subprocess.check_output([tmp/"test"],text=True).split())
    return dict(paged_tma_rejections=pages, dense_device_offset_shapes=offsets,
                comparison_tile_m=m,comparison_tile_n=n,packed_q_rows=8*8,
                sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()})


def replay():
    node=method("model_executor/runner_backend/full_cuda_graph_backend.py","FullCudaGraphBackend","replay")
    env=dict(graph_pool_replay_scope=contextlib.nullcontext)
    exec("from __future__ import annotations\n"+ast.unparse(node),env)
    select=method("layers/attention/graph_variants.py",None,"create_attention_graph_variants")
    assert "is_hip() and is_deepseek_dsa" in ast.unparse(select)
    runner=method("model_executor/runner/decode_cuda_graph_runner.py","DecodeCudaGraphRunner","execute")
    assert "self.backend.replay(self._replay_graph_key, forward_batch)" in ast.unparse(runner)
    fixed_base=18
    capture_output=[None]
    state=NS(offset=fixed_base,used=fixed_base+9)
    class Graph:
        def replay(self):
            capture_output[0]=tuple(range(fixed_base,fixed_base+state.used-state.offset))
    backend=NS(_graphs={8:Graph()},_outputs={8:capture_output})
    count=stale=fragmented=0
    for start in (fixed_base,42,197,388):
        for width in (8,9,21,40):
            state.offset,state.used=start,start+width
            live=NS(expected_base=start,expected_span=width)
            actual=env["replay"](backend,8,live)[0]
            expected=tuple(range(start,start+width))
            assert (actual!=expected)==(start!=fixed_base)
            count+=1
            stale+=actual!=expected
            altered=list(expected)
            altered[width//2]+=1000
            assert altered[0]==start and altered[-1]==start+width-1
            assert altered != list(range(start,start+width))
            fragmented+=1
    return dict(replays=count,stale_sliced_pointers=stale,fragmented_spans_with_correct_endpoints=fragmented,
                native_graph_calls="FullCudaGraphBackend.replay and DecodeCudaGraphRunner.execute")


def benchmark(problem, job):
    path=next((ROOT/"experiments/candidates"/problem/"profiles").glob(f"*.{job}.json"))
    rows=[r for r in json.loads(path.read_text())["benchmarks"] if r["case"]["load"]=="verify 1"]
    assert rows and all(r["status"]=="pass" for r in rows)
    return dict(job=job,source=str(path.relative_to(ROOT)),
                rows=[dict(case=r["case"].get("layer","expert"),weight=r["case"]["weight"],
                           raw_stock_us=r["baseline_raw_ns"]/1000,raw_submission_us=r["median_raw_ns"]/1000)
                      for r in rows],
                stock_ms=sum(r["case"]["weight"]*r["baseline_raw_ns"] for r in rows)/1e6,
                submission_ms=sum(r["case"]["weight"]*r["median_raw_ns"] for r in rows)/1e6)


def measured():
    board="muse_attention_decode"
    attn=benchmark(board,"c21d391dbf604bb5")
    layer=benchmark("muse_decode_layer","c39d323f785c4e97")
    moe=benchmark("muse_moe_w4a16","3c15b408fdb04c44")
    estimate=json.loads((HERE.parent/"v140_decode_overlap_feasibility/check.json").read_text())["model"]["rungs"][0]
    period=estimate["estimated_period_ms"]
    full=next(r for r in attn["rows"] if r["case"]=="full")
    raw_full_ms=full["weight"]*full["raw_stock_us"]/1000
    samples=[]
    for f in (ROOT/"experiments/candidates"/board/"profiles").glob("*.json"):
        j=json.loads(f.read_text())
        samples.extend(r["baseline_raw_ns"]/1000 for r in j["benchmarks"]
                       if r['case']['load']=="verify 1" and r['case']['layer']=="full")
    gain=lambda saved:period/(period-saved)
    return dict(attention=attn,decoder_layer=layer,experts=moe,estimated_scored_v64_period_ms=period,
                full_only_stock_bundle_ms=raw_full_ms,full_profiles=len(samples),other_stock_full_raw_range_us=[min(samples),max(samples)],
                generous_full_only_decode_gain=gain(raw_full_ms),
                measured_v4_full_only_savings_ms=16*(full["raw_stock_us"]-full["raw_submission_us"])/1000,
                generous_all_attention_decode_gain=gain(attn["stock_ms"]),
                measured_v4_all_attention_savings_ms=attn["stock_ms"]-attn["submission_ms"])


if __name__=="__main__":
    output=dict(sglang_pin=SG,fa3_pin=FA,native=native(),capture=replay(),benchmarks=measured(),
                physical_serving_test=False,new_official_calls=0)
    print(json.dumps(output,indent=2))
