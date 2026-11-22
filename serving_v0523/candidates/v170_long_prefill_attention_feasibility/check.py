"""Run pinned FA3 host dispatch and the V64 layernorm-SP guard. No GPU required."""
from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
SERVING = HERE.parents[1]
V64 = "b5a37c2923f8d1ab24c1c5cf7920a21e2c086774"
PRIOR = HERE.parent / "v132_verify_scheduler_metadata"
sys.path.insert(0, str(PRIOR))
from independent_geometry import normalized_code
from splits import BASE, PIN, function


def pinned(path):
    return subprocess.check_output(["git", "-C", str(SERVING/"sglang"), "show", V64+":"+path], text=True)


def topology():
    module = ast.parse(pinned("python/sglang/srt/arg_groups/layernorm_sp_hook.py"))
    defs = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in
            {"validate_layernorm_sp", "handle_layernorm_sp"}]

    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, n):
            return ast.copy_location(ast.Pass(), n)
    module = RemoveImports().visit(ast.Module(body=defs, type_ignores=[]))
    architecture = "MuseSparkForConditionalGeneration"
    ns = dict(is_cuda=lambda:True, get_device_capability=lambda:(9, 0),
              envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False)),
              resolving_view=lambda c:c, model_config_of=lambda c:NS(hf_config=NS(architectures=[architecture])),
              attn_dp_enabled_of=lambda c:False, SP_SUPPORTED_ARCHITECTURES={architecture},
              is_muse_spark_architecture=lambda a:a==architecture)
    exec(compile(module, "<pinned V64 resolver>", "exec"), ns)
    rejected = []
    for cp in (1, 2, 4, 8):
        for dcp in (1, 2):
            for prefill_cp in (False,True):
                cfg = NS(tp_size=8, ep_size=8, enable_layernorm_sp=True, quantization="modelopt_fp4",
                         moe_a2a_backend="none", moe_runner_backend="humming", speculative_algorithm="DFLASH",
                         pp_size=1, attn_cp_size=cp, dcp_size=dcp, enable_prefill_cp=prefill_cp,
                         enable_attn_tp_input_scattered=False, enable_mixed_chunk=False,
                         enable_deterministic_inference=False)
                try:
                    ns["handle_layernorm_sp"](cfg)
                except ValueError as e:
                    assert str(e) == "Muse W4A16 LayerNorm SP requires no context parallelism.", str(e)
                    rejected.append([cp,dcp,prefill_cp])
                else:
                    assert (cp,dcp,prefill_cp)==(1,1,False)
    model_prefix = "python/sglang/srt/models/"
    muse, other = (pinned(model_prefix+name+".py") for name in ("muse_spark","qwen4_exp"))
    assert all(x not in muse and x in other for x in ("cp_split_before_forward", "cp_gather_after_forward"))
    assert 'tp_size = parallel.attn_tp_size' in muse and 'parallel_group="attn_tp"' in muse
    return dict(accepted=[[1,1,False]],rejected=rejected,reason="Muse W4A16 LayerNorm SP requires no context parallelism.")


def host_dispatch(cases, cold):
    names = ["flash_api.cpp", "heuristics.h", "tile_size.h", "block.h"]
    with ThreadPoolExecutor(3) as t:
        src = dict(zip(names,t.map(lambda name:urlopen(BASE+name,timeout=20).read(), names)))
    hashes = {name:hashlib.sha256(data).hexdigest() for name,data in src.items()}
    known = json.loads((PRIOR/"splits.json").read_text())["immutable_sha256"]
    known.update(json.loads((HERE.parent/"v146_zero_copy_window_prefill/independent_fa3.json").read_text())["source_sha256"])
    assert all(hashes[name]==known[name] for name in names)
    cpp, heur, tile = (src[n].decode() for n in names[:3])
    assert "if (scheduler_needs_semaphore && !use_prepare_varlen)" in cpp and "tile_count_semaphore.zero_()" in cpp
    unit = """
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
struct Flash_fwd_params{
 int arch=90,d=64,dv=64,d_rounded=64,dv_rounded=64,h=16,h_k=2,seqlen_q=8,seqlen_k=0,
     total_q=0,num_sm=132,window_size_left=0,window_size_right=0,b=1,page_size=1,attention_chunk=0,num_splits=0;
 bool cu_seqlens_q=true,cu_seqlens_k=false,seqused_q=false,seqused_k=true,leftpad_k=false,
      is_causal=false,is_local=false,is_e4m3=true,page_table=true,pagedkv_tma=false,knew_ptr=false;
 float softcap=0;
};
""" + tile
    for name in ("inline bool use_one_mma_wg","inline bool should_pack_gqa","inline int num_splits_heuristic"):
        unit += "\n"+function(heur,name)
    for name in ("inline bool get_pagedkv_tma","inline bool get_pack_gqa","inline int get_num_splits"):
        unit += "\n"+function(cpp,name)
    unit += """
int main(){
 int b,q,k,page,left,variable;
 while(std::cin>>b>>q>>k>>page>>left>>variable){
    Flash_fwd_params params;
    int seqlen_q=q,seqlen_k=k,window_size_left=left,window_size_right=-1,attention_chunk=0,head_size=64;
    bool is_causal=true,paged_KV=page>0;
    params.b=b; params.total_q=b*q;params.seqlen_q=q;params.seqlen_k=k;
    params.page_size=page;params.page_table=paged_KV;
    params.cu_seqlens_q=variable;params.seqused_k=variable;
"""
    unit += normalized_code(cpp,False)
    unit += """
    params.pagedkv_tma=get_pagedkv_tma(params);
    params.num_splits=get_num_splits(params);
    bool non_tma=paged_KV&&!params.pagedkv_tma;
    auto tile=tile_size_fwd_sm90(64,64,params.is_causal,params.is_local,1,false,
                                 non_tma,false,use_one_mma_wg(params));
    std::cout<<params.num_splits<<" "<<!non_tma<<" "<<get_pack_gqa(params)<<" "
             <<std::get<0>(tile)<<" "<<std::get<1>(tile)<<"\\n";
 }
}"""
    with tempfile.TemporaryDirectory(prefix="okc-v170-") as td:
        cp = Path(td)/"dispatch.cpp"
        exe = Path(td)/"dispatch"
        cp.write_text(unit)
        subprocess.run(["c++","-std=c++17","-w",str(cp),"-o",str(exe)],check=True)
        expanded = [(*row,1) for row in cases] + [(b,q,k,0,w,0) for b,q,k in cold for w in (-1,2047)]
        data = "".join(" ".join(map(str,row))+"\n" for row in expanded)
        rows = subprocess.check_output([str(exe)],input=data,text=True).splitlines()
        # Reuse the host declarations from the pinned block-range fixture; execute the real C++ inner bounds.
        source_text = (HERE.parent / "v146_zero_copy_window_prefill/independent_fa3.py").read_text()
        assigns = ast.parse(source_text).body
        lit = next(n.value.value for n in assigns if isinstance(n,ast.Assign) and
                   isinstance(n.targets[0],ast.Name) and n.targets[0].id=="program")
        header = lit[:lit.index("template<bool Packed,bool Split>")]
        header += """
template<bool Packed,bool Sliding>
long long count_tiles(int q,int k) {
    using Bounds=flash::BlockMN<Info,192,160,!Sliding,Sliding,Packed,false>;
    long long tiles=0;
    for(int m=0;m<cute::ceil_div(q*(Packed?8:1),192);++m) {
        auto [lo,hi,off]=Bounds::get_n_block_min_max({k,q},m,0,0,1,Sliding?2047:k-1,0,{0},{8});
        tiles+=std::max(hi-lo,0);
    }
    return tiles*(Packed?2:16);
}
int main(){ int q,k; while(std::cin>>q>>k) std::cout<<count_tiles<true,false>(q,k)<<" "
   <<count_tiles<false,false>(q,k)<<" "<<count_tiles<true,true>(q,k)<<" "<<count_tiles<false,true>(q,k)<<"\\n";}
"""
        (Path(td)/"block.h").write_bytes(src["block.h"])
        cp.write_text(header)
        subprocess.run(["c++","-O2","-std=c++17","-w",str(cp),"-o",str(exe)],check=True)
        raw = subprocess.check_output([str(exe)],input="".join(f"{q} {k}\n" for _,q,k in cold),text=True)
        tiles = [list(map(int,row.split())) for row in raw.splitlines()]
    assert len(rows)==len(expanded) and len(tiles)==len(cold)
    decoded = [list(map(int,row.split())) for row in rows]
    return decoded[:len(cases)],hashes,decoded[len(cases):],tiles


def main():
    result = json.loads((SERVING/"runs/abd217baf0ed49fd.json").read_text())["result"]
    log = (SERVING/"runs/abd217baf0ed49fd.log").read_text()
    assert "full_layer_tokens=3419325, swa_layer_tokens=854831" in log
    baseline = result["rungs"][0]
    assert result["passed"] and result["frontier"]["score"]==1.4119
    # In the targeted single-request cold-turn schedule, the final chunk is smaller.
    n = result["shape"]["first_turn"]
    chunk = 16384
    window = 2048
    cold = [(1,min(chunk,n-i),min(i+chunk,n)) for i in range(0,n,chunk)]
    scorelike = {(b,q,k) for b in (1,4,16,32) for q in (512,753,1024,2048,4096,8192,16384)
                 for k in (q,74160,131072) if b*q<=chunk}
    cases = sorted({(b,q,((k+159)//160)*160 if p==160 else k,p,left)
                    for b,q,k in (set(cold)|scorelike)
                    for p in (0,1,160) for left in (-1,window-1)})
    rows,hashes,static,tiles = host_dispatch(cases,cold)
    idx=dict(zip(cases,rows))
    different_splits=[dict(case=c,result=r) for c,r in idx.items() if r[0]!=1]
    families={}
    for page in (0,1,160):
        for left in (-1,2047):
            family = [r for c,r in idx.items() if c[-2:]==(page,left)]
            families[f"page{page}_left{left}"]={"cases":len(family),"dispatch_frequencies":{
                str(k):v for k,v in sorted(Counter(tuple(r) for r in family).items())}}
    cold_rows = []
    for ci,(b,q,k) in enumerate(cold):
        geo={}
        for page in (0,1,160):
            km=((k+159)//160)*160 if page==160 else k
            geo[str(page)]=[idx[b,q,km,page,left] for left in (-1,window-1)]
        cold_rows.append(dict(q=q,k=k,layout=geo,alternative_fixed_dense=static[ci*2:ci*2+2],
                              full_packed_unpacked_swa_packed_unpacked_tile_pairs=tiles[ci]))
    full=n*(n+1)//2
    swa=window*(window+1)//2+(n-window)*window
    gqa_heads=16
    return dict(fa3=PIN,source_sha256=hashes,v64_tree=V64,h200_execution=False,
        official_id="abd217baf0ed49fd",score=result["frontier"]["score"],memory=result["gpu_memory"],
        one_user={k:baseline[k] for k in ("ttft_first_p90_s","ttft_later_p90_s","cache_hit")},
        shape=result["shape"],cold=cold_rows,geometry_fields=["splits","uses_tma_kv","pack_gqa","tile_m","tile_n"],
        tested_geometries=len(cases),additional_uniform_fixed_geometries=len(static),families=families,multisplit_geometries=different_splits,
        layernorm_sp_resolver=topology(),math=dict(full_qk_pairs=full,swa_qk_pairs=swa,
            useful_qk_and_av_flops_per_rank=4*gqa_heads*64*(16*full+46*swa),
            static_full_plus_swa_kv_bytes_per_rank=(3419325*16+854831*46)*2*2*64,
            first_turn_kv_bytes_copied_per_rank_if_full_packed_all_chunks=16*sum(k for _,_,k in cold)*256,
            first_turn_kv_bytes_copied_per_rank_if_swa_packed_all_chunks=46*sum(min(k,q+window-1) for _,q,k in cold)*256))
if __name__=="__main__":
    print(json.dumps(main(),indent=2))
