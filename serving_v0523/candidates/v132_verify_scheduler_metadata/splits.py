"""Run the exact pinned host split heuristic for native V64 DFLASH verification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen


PIN = "f89bc2306632d1ec5f97b014dded4254f5b4a907"
BASE = f"https://raw.githubusercontent.com/sgl-project/sgl-attn/{PIN}/hopper/"
FILES = ("flash_api.cpp", "flash_prepare_scheduler.cu", "heuristics.h", "tile_size.h",
         "flash_fwd_launch_template.h", "flash_fwd_combine_launch_template.h", "flash_fwd_combine_kernel.h")


def function(source, declaration):
    start = source.index(declaration)
    depth = 0
    for offset in range(source.index("{", start), len(source)):
        depth += source[offset] == "{"
        depth -= source[offset] == "}"
        if not depth:
            return source[start:offset+1]
    raise ValueError(declaration)


def check(source_dir):
    data = {f: (source_dir/f).read_bytes() if source_dir else urlopen(BASE+f, timeout=30).read()
            for f in FILES}
    src = {f:b.decode() for f,b in data.items()}
    cpp = src["flash_api.cpp"]
    assert "else if (scheduler_needs_semaphore && params.skip_scheduler_metadata_computation)" in cpp
    assert "tile_count_semaphore_offset + 1)}).zero_()" in cpp
    assert "Varlen && !params.skip_scheduler_metadata_computation" in src["flash_fwd_launch_template.h"]
    assert "*params.semaphore_to_reset = 0;" in src["flash_fwd_combine_kernel.h"]
    assert "*tile_count_semaphore = 0;" in src["flash_prepare_scheduler.cu"]
    assert "max_num_pages_per_seq * page_size" in cpp
    unit = """
#include <algorithm>
#include <cmath>
#include <iostream>
struct Flash_fwd_params {
    int page_size=1,arch=90,d=64,dv=64,d_rounded=64,dv_rounded=64,h=16,h_k=2,seqlen_q=8,
        seqlen_k=0,num_sm=132,window_size_left=0,window_size_right=0,b=0,total_q=0;
    bool cu_seqlens_q=true,cu_seqlens_k=false,seqused_q=false,seqused_k=true,leftpad_k=false,
        is_causal=true,is_local=false,is_e4m3=true,page_table=true,pagedkv_tma=false,knew_ptr=false;
    float softcap=0;
};
"""
    unit += src["tile_size.h"]
    unit += "\n" + function(src["heuristics.h"], "inline bool use_one_mma_wg")
    unit += "\n" + function(src["heuristics.h"], "inline int num_splits_heuristic")
    unit += "\n" + function(cpp, "inline bool get_pagedkv_tma")
    unit += "\n" + function(cpp, "inline int get_num_splits")
    unit += """
int main() {
    int accepted=0, declined=0;
    for (int page : {1, 16, 128, 160})
    for (int cols : {1, 4, 32, 512, 513, 1024})
    for (int width : {2, 3, 4, 7, 8, 12, 16})
    for (int gqa : {1, 2, 4, 8})
    for (int kv : {1, 2, 8})
    for (int b : {1, 4, 11, 16, 31, 32, 64})
    for (int sm : {32, 108, 132})
    for (int window : {-1, 255, 447, 448, 2047}) {
        Flash_fwd_params p;
        p.page_size=page; p.seqlen_k=page*cols; p.seqlen_q=width;
        p.b=b; p.h_k=kv; p.h=kv*gqa; p.num_sm=sm;
        p.window_size_left=window >= p.seqlen_k-1 ? -1 : window;
        p.is_causal=p.window_size_left < 0; p.is_local=!p.is_causal;
        if (p.window_size_left < 0) p.window_size_left=p.seqlen_k-1;
        p.pagedkv_tma=get_pagedkv_tma(p);
        int loaded=window < 0 ? p.seqlen_k : std::min(p.seqlen_k,window+1+64);
        int mblocks=((b*width*gqa+63)/64)*kv;
        if (width*gqa > 64 || loaded <= 4*128 || 2*mblocks > sm) { ++declined; continue; }
        ++accepted;
        int builder=get_num_splits(p);
        p.total_q=b*width;
        if (builder<2 || builder != get_num_splits(p) || p.pagedkv_tma) return 1;
    }
    std::cout << "bounds " << accepted << " " << declined << "\\n";
    for (int cols : {128, 512, 640, 2048, 8192, 74160, 131072, 131200}) {
        for (int batch : {1, 4, 16, 32}) {
            for (int window : {-1, 2047}) {
                Flash_fwd_params p;
                p.b=batch; p.seqlen_k=cols; p.page_size=(cols==131200 ? 160 : 1);
                p.window_size_left=(window >= cols-1 ? -1 : window);
                p.is_causal=p.window_size_left == -1; p.is_local=!p.is_causal;
                // Full causal converts the absent left bound to the K extent.
                if (p.window_size_left < 0) p.window_size_left=cols-1;
                p.pagedkv_tma=get_pagedkv_tma(p);
                int prepare_splits = get_num_splits(p);
                p.total_q=batch * p.seqlen_q;
                int forward_splits = get_num_splits(p);
                std::cout << cols << " " << batch << " " << window << " " << prepare_splits
                          << " " << forward_splits << "\\n";
            }
        }
    }
}
"""
    with tempfile.TemporaryDirectory(prefix="okc-fa3-verify-host-") as tmp:
        root = Path(tmp)
        (root/"audit.cc").write_text(unit)
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-w", str(root/"audit.cc"),
                        "-o", str(root/"audit")], check=True)
        output = subprocess.check_output([root/"audit"], text=True)
    geometries = {}
    for line in output.splitlines():
        if line.startswith("bounds "):
            qualified, declined = map(int, line.split()[1:])
            continue
        cols, batch, window, prep, forward = map(int, line.split())
        assert prep == forward
        geometries.setdefault((cols,batch), {})["full" if window == -1 else "swa"] = forward
    rows = []
    for (cols,batch), modes in geometries.items():
        # Always request two independent templates in this comparison, even
        # when a short physical table normalizes both modes to full causal.
        standalone = 16 * (modes["full"] == 1) + 46 * (modes["swa"] == 1)
        page=160 if cols==131200 else 1
        rows.append(dict(physical_k_capacity=cols, page_columns=cols//page, page_size=page, batch=batch, **modes, extra_standalone_zero=standalone,
                         net_kernels_removed_before_copies=60-standalone))
    return dict(fa3_pin=PIN, sm_count=132, width=8, q_heads_per_rank=16,
                kv_heads_per_rank=2, head_dim=64, qkv_dtype="e4m3", target_layers=[16,46],
                immutable_sha256={f:hashlib.sha256(data[f]).hexdigest() for f in FILES},
                qualified_multi_split_pairs=qualified, outside_conservative_bound=declined,
                qualified_page_sizes=[1,16,128,160], verified_builder_forward_pairs=len(rows)*2, shapes=rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-dir",type=Path,help="Read pinned hopper sources locally instead of downloading them.")
    print(json.dumps(check(parser.parse_args().source_dir),indent=2))
