"""Compile the exact pinned FA3 host heuristics and compare standalone builder vs paged forward."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile

from splits import function

HERE = Path(__file__).resolve().parent
SGL_ATTENTION = "f89bc2306632d1ec5f97b014dded4254f5b4a907"


def normalized_code(cpp, builder):
    if builder:
        body = cpp[cpp.index("mha_fwd_get_scheduler_metadata("):]
        return body[body.index("    if (window_size_left >= max_seqlen_k - 1)"):body.index("    params.arch =")]
    forward = cpp[cpp.index("    // This needs to go before kBlockM"): ]
    topology = forward[forward.index("    if (window_size_left >= seqlen_k - 1)"):
                       forward.index("    if (!is_varlen_q)")]
    params = cpp[cpp.index("    params.is_causal = window_size_left"):cpp.index("    params.arch =")]
    return topology + params


def build_oracle(vendor, directory):
    hashes = json.loads((HERE/"splits.json").read_text())["immutable_sha256"]
    for name, expected in hashes.items():
        assert hashlib.sha256((vendor/name).read_bytes()).hexdigest()==expected
    cpp = (vendor/"flash_api.cpp").read_text()
    heur = (vendor/"heuristics.h").read_text()
    unit = """
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>
struct Flash_fwd_params {
    int arch=90,d=64,dv=64,d_rounded=64,dv_rounded=64,h=16,h_k=2,seqlen_q=8,seqlen_k=0,
        total_q=0,num_sm=132,window_size_left=0,window_size_right=0,b=1,page_size=1,attention_chunk=0,num_splits=0;
    bool cu_seqlens_q=true,cu_seqlens_k=false,seqused_q=false,seqused_k=true,leftpad_k=false,
        is_causal=false,is_local=false,is_e4m3=true,page_table=true,pagedkv_tma=false,knew_ptr=false;
    float softcap=0;
};
"""
    unit += (vendor/"tile_size.h").read_text()
    for name in ("inline bool use_one_mma_wg","inline bool should_pack_gqa","inline int num_splits_heuristic"):
        unit += "\n"+function(heur,name)
    for name in ("inline bool get_pagedkv_tma","inline bool get_pack_gqa","inline int get_num_splits"):
        unit += "\n"+function(cpp,name)
    unit += """
std::array<int,7> run(bool builder,int b,int width,int page,int cols,int left,int sms,int explicit_splits) {
    Flash_fwd_params params;
    params.b=b;params.seqlen_q=width;params.seqlen_k=page*cols;params.page_size=page;
    params.num_sm=sms;params.total_q=builder ? 0 : b*width;
    int max_seqlen_q=width,max_seqlen_k=params.seqlen_k,seqlen_q=width,seqlen_k=params.seqlen_k;
    int window_size_left=left,window_size_right=-1,attention_chunk=0,headdim=64,head_size=64;
    bool is_causal=true,paged_KV=true;
    std::optional<int> page_size=page;
    if(builder) {
"""
    unit += normalized_code(cpp,True)+"\n    } else {"+normalized_code(cpp,False)+"""\n    }
    params.pagedkv_tma=get_pagedkv_tma(params);
    params.num_splits=explicit_splits>0 ? explicit_splits:get_num_splits(params);
    return {params.is_causal,params.is_local,params.window_size_left,params.window_size_right,
            params.pagedkv_tma,params.num_splits,get_pack_gqa(params)};
}
int main() {
    int b,w,p,c,l,sm,ns;
    while(std::cin>>b>>w>>p>>c>>l>>sm>>ns) {
        auto a=run(true,b,w,p,c,l,sm,ns),f=run(false,b,w,p,c,l,sm,ns);
        assert(a==f);
        for(auto i:a) std::cout<<i<<" ";
        std::cout<<"\\n";
    }
}
"""
    source = directory/"geometry.cpp"
    source.write_text(unit)
    exe = directory/"geometry"
    subprocess.run(["c++","-std=c++17","-w",str(source),"-o",str(exe)],check=True)
    return exe


def evaluate(exe, cases):
    inputs="".join(" ".join(map(str,row))+"\n" for row in cases)
    rows=subprocess.check_output([str(exe)],input=inputs,text=True).splitlines()
    assert len(rows)==len(cases)
    return [tuple(map(int,row.split())) for row in rows]


def main(vendor):
    with tempfile.TemporaryDirectory(prefix="okc-v132-abi-") as scratch:
        exe=build_oracle(vendor,Path(scratch))
        cases=list(itertools.product((1,2,4,8,16,32,48,64), (1,8,9,16,32), (1,128,160),
                                      (1,4,13,512,820), (-1,31,2047), (108,132), (0,1,4)))
        rows=evaluate(exe,cases)
        official=list(itertools.product((1,4,16,32),(8,),(160,),(820,),(-1,2047),(132,),(0,)))
        judge_rows=evaluate(exe,official)
        assert all(not r[4] and r[5]>1 and r[6] for r in judge_rows)
        assert [r[5] for r in judge_rows]==[57,15,15,14,4,4,2,2]
        # A shorter key table can normalize SWA to the full causal algorithm.
        # Single-split and TMA-positive controls exercise the inverse boundaries.
        assert {r[5] for r in rows if r[4]} and any(r[5]==1 for r in rows)
        return dict(fa3_pin=SGL_ATTENTION,matched_builder_forward_geometries=len(rows),
                    included_query_widths=[1,8,9,16,32],page_sizes=[1,128,160],sm_counts=[108,132],
                    verified_fields=["is_causal","is_local","window_left","window_right",
                                     "pagedkv_tma","static_num_splits","pack_gqa"],
                    v120b_page_capacity_tokens=820*160,
                    target_full_swa_splits={str(b):[judge_rows[i*2][5],judge_rows[i*2+1][5]]
                                            for i,b in enumerate((1,4,16,32))},
                    h200_execution=False)


if __name__=="__main__":
    import sys
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
