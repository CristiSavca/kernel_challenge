#!/usr/bin/env python3
"""Actual packed-byte inverse, address coverage, and extracted-C++ schedule proof.

Requires NumPy and clang++; never imports CUDA/Torch or the complete candidate.
This is a CPU proof of specific invariants, not GPU correctness/performance.
"""
import ast
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from test_wgs_kperm3_cpu import CpuTorch, Tensor, function, literal

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/candidates/muse_moe_w4a16/ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a.py"
CANDIDATE = ROOT / "muse_moe_w4a16/candidates/native_wgs_compact_v10.py"


def cpp_function(source, name):
    start = re.search(r"^__device__[^\n]*\b" + re.escape(name) + r"\(", source, re.M).start()
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__device__", "").replace("__forceinline__", "").replace("__noinline__", "")


def preprocess(source, defines):
    source = re.sub(r"^#include.*$", "", source, flags=re.M)
    run = subprocess.run(["clang++", "-E", "-P", "-x", "c++", "-D__CUDA_ARCH__=900",
                          *defines, "-"], input=source, capture_output=True, text=True, check=True)
    # Only the constexpr expression used in the launch bound differs outside WGS.
    return re.sub(r"0\s*\?\s*2\s*:\s*1", "1", run.stdout)


def actual_byte_proof(tree):
    allowed = {"_stored_pos", "_stored_pos_mul", "_pack_codes_nat", "_wgs_copy"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in allowed]
    assert {n.name for n in nodes} == allowed
    ns = {"torch": CpuTorch, "DQ7": True}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CANDIDATE), "exec"), ns)
    row = np.arange(8192, dtype=np.uint32)[:, None]
    col = np.arange(2048, dtype=np.uint32)[None, :]
    codes = ((row * 37 + col * 11 + ((row ^ col) >> 3)) & 255).astype(np.uint8)
    scales = ((row * 13 + np.arange(256, dtype=np.uint32)[None, :] * 7) & 255).astype(np.uint8)
    gate = np.arange(4096).reshape(-1, 8)
    rows_a = np.stack((gate, gate + 4096), axis=1).reshape(-1)
    total_codes = total_scales = 0
    for encoding in ("_stored_pos", "_stored_pos_mul"):
        pos = ns[encoding]()
        for label, rows in (("W13", rows_a), ("W2", np.arange(4096))):
            packed = ns["_wgs_copy"](Tensor(codes), Tensor(scales), Tensor(rows), pos).a
            units = len(rows) // 128
            pieces = packed.reshape(units, 32, 2, 4608)
            words = pieces[..., :4096].copy().view(np.uint32).reshape(units, 32, 2, 2, 4, 32, 4)
            u, st, wg, lg, wp, lane, j = np.indices(words.shape, sparse=True)
            g, t = lane // 4, lane % 4
            for reg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                for e in range(2):
                    got = np.zeros(words.shape, dtype=np.uint32)
                    for bit in range(4):
                        got |= ((words >> pos[(reg, e, bit)]) & 1) << bit
                    # Independent logical NVFP4 coordinates for the MMA A fragment.
                    r = rows[u * 128 + wg * 64 + wp * 16 + g + h * 8]
                    k = st * 128 + lg * 64 + j * 16 + q * 8 + t * 2 + e
                    expected = (codes[r, k // 2] >> (4 * (k % 2))) & 15
                    assert np.array_equal(got, np.broadcast_to(expected, got.shape)), (encoding, label, reg, e)
                    total_codes += got.size
            sc = pieces[..., 4096:].reshape(units, 32, 2, 2, 4, 8, 4, 2)
            u, st, wg, lg, wp, g, j, h = np.indices(sc.shape, sparse=True)
            r = rows[u * 128 + wg * 64 + wp * 16 + g + 8 * h]
            kblock = st * 8 + lg * 4 + j
            assert np.array_equal(sc, scales[r, kblock])
            total_scales += sc.size
            assert packed.nbytes == len(rows) * 4096 * 9 // 16
    return {"fp4_codes": total_codes, "scale_bytes": total_scales, "encodings": 2}


def activation_and_readiness_proof():
    checked = 0
    for nt in range(1, 9):
        rows = nt * 8
        # Every real and padded token coordinate; padding is not consumed by output.
        seen = set()
        for r in range(rows):
            for k in range(4096):
                stage, kin = divmod(k, 128)
                ref = stage * rows * 128 + (r // 8) * 1024 + (kin // 8) * 64 + (r % 8) * 8 + kin % 8
                kc, e = divmod(k, 8)
                gather = ((((kc >> 4) * nt + (r >> 3)) * 16 + (kc & 15)) * 8 + (r & 7)) * 8 + e
                epi = ((((k >> 7) * nt + (r >> 3)) * 16 + ((k & 127) >> 3)) * 64 + (r & 7) * 8) + (k & 7)
                assert ref == gather == epi
                assert ref not in seen
                seen.add(ref)
                # Descriptor addresses: N8 stride2048B, leading offset128B.
                desc_byte = (r // 8) * 2048 + (kin // 8) * 128 + (r % 8) * 16 + (kin % 8) * 2
                assert 0 <= desc_byte <= rows * 128 * 2 - 2
                assert ref == stage * rows * 128 + desc_byte // 2
                checked += 1
        assert len(seen) == rows * 4096 and max(seen) == rows * 4096 - 1
    readiness = [[] for _ in range(32)]
    for ub in range(64):
        for wg in range(2):
            cols = list(range(ub * 64 + wg * 32, ub * 64 + wg * 32 + 32))
            assert all(c // 128 == ub // 2 for c in cols)
            readiness[ub // 2].append(cols)
    for stage, writers in enumerate(readiness):
        assert len(writers) == 4
        assert sorted(c for w in writers for c in w) == list(range(stage * 128, (stage + 1) * 128))
    return {"activation_coordinates": checked, "doneAs_stages": 32, "writers_per_stage": 4}


def extracted_cpp_schedule_proof(source):
    names = ("wgs_cost", "wgs_prefix", "wgs_stage_at", "wgs_range", "wgs_cta_of", "wgs_split")
    code = "\n".join(cpp_function(source, name) for name in names)
    # Compile the actual candidate functions, then compare with a different flat
    # cost-interval oracle. No model of those functions substitutes for executing them.
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#define OKC_WGSREORD 1
struct Smem {int jC[88]{},wcp[89]{},wcpB[89]{};uint32_t wtot{},wtotB{};};
constexpr unsigned long long pack(const int (&v)[8]) { unsigned long long r=0;for(int i=0;i<8;++i)r|=(unsigned long long)v[i]<<(i*8);return r;}
constexpr int ca[8]={112,135,140,158,173,185,200,207}, cb[8]={100,104,115,123,132,144,156,174};
constexpr unsigned long long WGS_PA=pack(ca),WGS_PB=pack(cb);
''' + code + r'''
int main(){
  std::vector<std::vector<int>> tests={{},{1},{64},{1,64},{7,8,9,15,16,17,31,32,33,63,64}};
  for(int tokens:{1,8,16,17,32,33,128,256,257,384}){
    std::vector<int> concentrated;for(int n=tokens;n>0;n-=64)concentrated.push_back(std::min(64,n));tests.push_back(concentrated);
    std::vector<int> spread;for(int i=0;i<32;++i){int c=tokens*8/32+(i<tokens*8%32);for(;c>0;c-=64)spread.push_back(std::min(64,c));}tests.push_back(spread);
  }
  std::mt19937 gen(20261104);
  for(int trial=0;trial<120;++trial){std::vector<int> jobs;int hist[32]={};int m=1+gen()%384;for(int t=0;t<m;++t){std::set<int> chosen;for(int n=gen()%9;n>0;--n){int e=gen()%32;while(chosen.count(e))e=gen()%32;chosen.insert(e);++hist[e];}}for(int c:hist)for(;c>0;c-=64)jobs.push_back(std::min(64,c));tests.push_back(jobs);}
  // Capacity stress (not a claim all these padded counts arise from one input).
  tests.push_back(std::vector<int>(88,64));
  size_t ranges=0,stages=0,split_units=0;
  for(const auto&jobs:tests){assert(jobs.size()<=88);Smem s{};std::copy(jobs.begin(),jobs.end(),s.jC);wgs_prefix(s,jobs.size());
    for(int pb=0;pb<2;++pb)for(int G:{1,2,132,264,319}){
      const int unit_count=pb?32:64, stage_count=unit_count*32;
      std::vector<uint64_t> boundaries(1,0);
      for(int c:jobs)for(int k=0;k<stage_count;++k)boundaries.push_back(boundaries.back()+(pb?cb:ca)[(c+7)/8-1]);
      uint64_t total=boundaries.back();std::vector<int> owner(boundaries.size()-1,-1);std::vector<std::vector<int>> unit_owners((boundaries.size()-1)/32);
      std::vector<std::pair<int,int>> rg;
      for(int c=0;c<G;++c){int a=-1,b=-1;wgs_range(s,jobs.size(),pb,c,G,a,b);++ranges;
        int ra=0,rb=0;if(total){ra=std::upper_bound(boundaries.begin(),boundaries.end(),total*c/G)-boundaries.begin()-1;rb=c+1==G?int(boundaries.size()-1):int(std::upper_bound(boundaries.begin(),boundaries.end(),total*(c+1)/G)-boundaries.begin()-1);}
        assert(a==ra&&b==rb&&a<=b);rg.push_back({a,b});
        int split=wgs_split(pb,a,b);assert(a<=split&&split<=b);std::vector<int> order;for(int x=split;x<b;++x)order.push_back(x);for(int x=a;x<split;++x)order.push_back(x);std::sort(order.begin(),order.end());assert(int(order.size())==b-a);
        for(int i=a;i<b;++i){assert(owner[i]==-1);owner[i]=c;assert(order[i-a]==i);++stages;}
      }
      for(int u=0;u<int(unit_owners.size());++u){int first=owner[u*32];assert(first>=0&&first<G);assert(wgs_cta_of(s,pb,u*32,G)==first);int sum=0;std::set<int> contributors;int finisher=0;
        for(int c=0;c<G;++c){int n=std::max(0,std::min(rg[c].second,(u+1)*32)-std::max(rg[c].first,u*32));if(n){sum+=n;contributors.insert(c);if(rg[c].first<=u*32&&rg[c].second>u*32)++finisher;}}
        assert(sum==32&&finisher==1);if(contributors.size()>1){++split_units;assert(first<320);}
      }
      // Each CTA can own only its trailing split-unit slot (the parent's invariant).
      std::set<int> slots;for(int u=0;u<int(unit_owners.size());++u){if(owner[u*32]!=owner[u*32+31])assert(slots.insert(owner[u*32]).second);}
    }
  }
  std::cout<<"{\"histograms\":"<<tests.size()<<",\"ranges\":"<<ranges<<",\"stages\":"<<stages<<",\"split_units\":"<<split_units<<"}\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="okc-compact-schedule-") as td:
        exe = Path(td) / "proof"
        subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-O2", "-o", str(exe), "-"],
                       input=harness, text=True, capture_output=True, check=True)
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def resource_and_scratch_proof(source):
    a = source.index("struct Smem {")
    b = source.index("\n};", a) + 3
    native = ("#include <cstdint>\n#define OKC_WGS 1\n#define OKC_WS 0\n"
              "constexpr int NW=8,NRB=2,RED_W=4,JMAX=88,EMAX=3072;\n"
              + source[a:b] + "\nstatic_assert(sizeof(Smem)==21216);\n")
    subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-fsyntax-only", "-"],
                   input=native, text=True, capture_output=True, check=True)
    dynamic = 21248 + 3 * (9216 + 16384)
    assert dynamic == 98048 and 2 * (dynamic + 1024) <= 233472
    em, hd, slots, groups, fixw = 3072, 4096, 320, 2, 4096
    regions = []
    start = em + em * hd
    for name, size in (("fixA", slots * groups * fixw), ("fixB", slots * groups * fixw),
                       ("countA", slots * groups), ("countB", slots * groups), ("doneAs", 136 * 32)):
        assert start * 4 % 128 == 0
        regions.append({"name": name, "first_float": start, "end_float": start + size})
        start += size
    assert start == em + em * hd + 2 * 320 * 2 * 4096 + 2 * 320 * 2 + 136 * 32
    assert 88 * 32 <= 136 * 32 and 264 + 1 <= slots
    for wg in range(2):
        for nt in range(8):
            for lane in range(128):
                assert wg * fixw + nt * 512 + lane * 4 + 3 < groups * fixw
    return {"native_smem_size": 21216, "rounded_metadata": 21248, "dynamic_smem": dynamic,
            "two_cta_smem_including_driver": 2 * (dynamic + 1024), "float_regions": regions,
            "cuda_occupancy": "unverified; candidate requires actual occupancy>=2 before launching"}


def main():
    parent_bytes, candidate_bytes = PARENT.read_bytes(), CANDIDATE.read_bytes()
    assert hashlib.sha256(parent_bytes).hexdigest() == PARENT.stem
    p, c = ast.parse(parent_bytes), ast.parse(candidate_bytes)
    for name in ("_pack_codes", "_pack_scales", "_fat_order", "_unit_order", "_pack_codes_nat",
                 "_scales_prmt_exact", "_main_defines", "kernel_fn", "prepare", "_prepare_layout", "_make_scratch"):
        assert ast.dump(function(p, name), include_attributes=False) == ast.dump(function(c, name), include_attributes=False), name
    for name in ("PDL", "PRMT_SCALES", "TINY_DEFINES", "TINY_MAX_M", "TINY2_DEFINES", "TINY2_MAX_M",
                 "MID_DEFINES", "MID_MAX_M", "MAIN_FALLBACK_DEFINES", "MAIN_NS", "DQ7", "CPP_SRC"):
        assert literal(p, name) == literal(c, name), name
    parent_src, src = literal(p, "CUDA_SRC"), literal(c, "CUDA_SRC")
    for name in ("TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES", "MAIN_FALLBACK_DEFINES"):
        for mul in (0, 1):
            for prmt in (0, 1):
                defs = [*literal(c, name), f"-DOKC_PRMTSC={prmt}", f"-DOKC_MULDEC={mul}", "-DOKC_DQ7=1"]
                assert preprocess(parent_src, defs) == preprocess(src, defs), (name, mul, prmt)
    assert "if (nj == 0) { g0 = g1 = 0; return; }" in src
    assert "occ0 >= 2 && occ1 >= 2" in src and "grid + 1 <= okc::WGS_SLOTS" in src
    assert "const uint32_t bytes = (uint32_t)npad * 256u;" in src
    assert "k < BW_E ? 2 : k < BX_F ? NW" in src
    assert "WGS_SLOTS = 320" in src and "WGS_PIECE = 4608" in src
    assert "JMAX * 32" in src and "j * 32 + st) < 4" in src
    assert src.count('asm volatile("griddepcontrol.launch_dependents;') == 1
    assert src.index('asm volatile("griddepcontrol.launch_dependents;') < src.index("DBG(4);")
    result = {"source_sha256": hashlib.sha256(candidate_bytes).hexdigest(), "fallback_preprocessed_builds": 16,
              "packing": actual_byte_proof(c), "activation_readiness": activation_and_readiness_proof(),
              "schedule": extracted_cpp_schedule_proof(src), "resources_scratch": resource_and_scratch_proof(src)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
