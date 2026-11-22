"""Compile and execute the released FA3 bottom-right window and split-tile geometry."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen


SHA="f89bc2306632d1ec5f97b014dded4254f5b4a907"
sources={}
for name in ("block.h","flash_api.cpp","flash_attn_interface.py","mainloop_fwd_sm90_tma_gmma_ws.hpp"):
    with urlopen(f"https://raw.githubusercontent.com/sgl-project/sgl-attn/{SHA}/hopper/{name}", timeout=15) as response:
        sources[name]=response.read()

api=sources["flash_api.cpp"].decode()
appended=api[api.index("if (k_new_.has_value()) {  // This needs to be set"):api.index("params.pagedkv_tma = get_pagedkv_tma",api.index("if (k_new_.has_value()) {  // This needs to be set"))]
assert "params.cu_seqlens_knew =" in appended
assert "bool const is_varlen_k = cu_seqlens_k_.has_value()" in api
mainloop=sources["mainloop_fwd_sm90_tma_gmma_ws.hpp"].decode()
assert "Use_TMA_KV = !PagedKVNonTMA" in mainloop
assert "seqlen_info.offset_k + n_offset, _0{}, _0{}), mK_TMA" in mainloop
assert "_0{}, seqlen_info.offset_k + n_offset, _0{}), mVt_TMA" in mainloop

program=r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <tuple>
#define CUTLASS_DEVICE
namespace cute {
using std::tuple;
int ceil_div(int a,int b) { return (a+b-1)/b; }
}
namespace cutlass {
struct FastDivmod {
    int divisor;
    int divide(int x) const { return x/divisor; }
};
}
namespace flash {
int round_down(cutlass::FastDivmod x,int n){ return (n/x.divisor)*x.divisor; }
int round_up(cutlass::FastDivmod x,int n){ return cute::ceil_div(n,x.divisor)*x.divisor; }
}
#include "block.h"
struct Info { int seqlen_k,seqlen_q; };
template<bool Packed,bool Split>
int64_t test() {
    using Native=flash::BlockMN<Info,192,160,false,true,Packed,Split>;
    int64_t cases=0;
    for (int q : {1,24,159,160,161,191,192,193,513,753,1024,16384})
    for (int prefix : {0,1,159,2045,2047,2048,65536,74160})
    for (int left : {0,31,127,2047,4095})
    for (int ratio : {1,8})
    for (int raw_splits : {1,2,4}) {
        const int k=prefix+q, omit=std::max(0,k-q-left);
        for (int m=0;m<cute::ceil_div(q*(Packed?ratio:1),192);++m) {
            for (int s=0; s<(Split?raw_splits:1); ++s) {
                auto [old_min, old_max, old_offset]=Native::get_n_block_min_max(
                    {k,q},m,0,s,raw_splits,left,0,{0},{ratio});
                auto [new_min, new_max, new_offset]=Native::get_n_block_min_max(
                    {k-omit,q},m,0,s,raw_splits,left,0,{0},{ratio});
                assert(old_min==new_min && old_max==new_max && old_offset==omit+new_offset);
                ++cases;
            }
        }
    }
    return cases;
}
int main() { std::cout << test<false,false>()+test<true,false>()+test<false,true>()+test<true,true>(); }
"""
with tempfile.TemporaryDirectory(prefix="okc-v146-fa3-") as tmp:
    folder=Path(tmp)
    (folder/"block.h").write_bytes(sources["block.h"])
    (folder/"native.cpp").write_text(program)
    subprocess.run(["c++","-std=c++17","-Wall","-Wextra","-Wno-unused-parameter","-Werror","-O2",
                    str(folder/"native.cpp"),"-o",str(folder/"native")],check=True)
    cases=int(subprocess.check_output([str(folder/"native")],text=True))
print(json.dumps(dict(sgl_attn_commit=SHA,compiled_native_tile_comparisons=cases,physical_gpu=False,
                      source_sha256={n:hashlib.sha256(v).hexdigest() for n,v in sources.items()}),indent=2))
