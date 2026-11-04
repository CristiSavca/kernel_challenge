"""Source-bound V30 checks; no claim of CUDA compilation or device arithmetic."""
import ast
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v30_build", HERE / "build.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def source_checks():
    parent = (HERE / "parent.py").read_text()
    candidate = (HERE / "submission.py").read_text()
    assert b.sha(parent) == b.PARENT_SHA and b.build() == candidate
    pcu, cu = b.value(parent, "CUDA_SRC"), b.value(candidate, "CUDA_SRC")
    added = b.device_addition(pcu)
    host = "\n#if defined(OKC_V30) && OKC_V30\n" + (HERE / "new_host.cuh").read_text() + "\n#endif\n"
    restored = candidate.replace(added + "\n}  // namespace okc", "}  // namespace okc")
    restored = restored.replace(host + "\nvoid okc_set_pdl", "void okc_set_pdl")
    restored = restored.replace('''#if defined(OKC_V30) && OKC_V30
  TORCH_CHECK(!g_okc_pdl, "V30 standalone uses ordinary ordered stream launches");
  v30_launch(p, (int)grid, at::cuda::getCurrentCUDAStream());
#else
''', "")
    restored = restored.replace("\n#endif  // V30 dispatch\n", "\n")
    restored = restored.replace('    return MAIN_DEFINES + ("-DOKC_V30=1",)', '    return MAIN_DEFINES')
    restored = restored.replace("okc_moe_wholek_v30_", "okc_moe_kperm3_v9_")
    restored = restored.replace('verbose=("-DOKC_V30=1" in defs)', 'verbose=False')
    restored = restored.replace("# V30: whole-K MAIN tiles and GPU sparse/dense job partition over frozen 247888acca.\n# KPERM3 lineage below remains; public fallback/tiny source and all packed formats are preserved.",
                                 "# Isolated KPERM3 adaptation over frozen ef7b210b; no SC6/MFIN or pipeline changes.")
    assert restored == parent, "complete source restoration"
    final = b.extract(pcu, "#if OKC_WGS && OKC_WGSFINSS\n  // item", "#elif OKC_FINCOL")
    final = final[final.index("\n") + 1:]
    assert cu.count(final) == 2
    for start, end in [("    // codes and scales one 4-step group", "        // the previous segment's deferred work")]:
        stage = b.extract(pcu, start, end)
        assert cu.count(stage) == 2, "actual complete decode/MMA/commit/wait/release body is exact"
    assert added.count("wgs_flush<") == 0 and "red.global.add" not in added
    assert "ld_acquire" not in added and "griddepcontrol" not in added
    assert "v30_gather(p, S);" in added and "meta->pos[i] = S.pos[i]" in added
    assert "p.xready" not in added and "p.doneAs" not in added
    assert added.count("const int sl = seq % WGS_NS;") == 1
    assert "int seq = 0;\n  for (int q = blockIdx.x;" in added
    assert "int seq = 0;\n    for (int q = blockIdx.x;" in added
    assert "k < BW_E ? 2 : k < BX_F ? CW : k < BX_E ? 1 : CW" in added
    assert "constexpr int CW = NC / 32;" in added
    assert "constexpr int NC = COLD ? 256 : 512;" in added
    assert "__syncthreads(); // producers and every consuming warp finish" in added
    return {"parent": b.sha(parent), "source": b.sha(candidate), "full_restoration": True,
            "stage_arithmetic_source_identity": True, "finalizer_source_identity": True}


def preprocess_controls():
    parent = (HERE / "parent.py").read_text()
    candidate = (HERE / "submission.py").read_text()
    cases = []
    with tempfile.TemporaryDirectory(prefix="v30-preprocess-") as td:
        paths = []
        for name, s in [("p", parent), ("c", candidate)]:
            p = Path(td) / (name + ".cpp")
            p.write_text(re.sub(r"^#include.*$", "", b.value(s, "CUDA_SRC"), flags=re.M))
            paths.append(p)
        # Sixteen inherited builds, with OKC_V30 deliberately absent: they must remain byte-identical.
        for key in ["MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES"]:
            for prmt in [False, True]:
                for mul in [False, True]:
                    dkey = "MAIN_FALLBACK_DEFINES" if key == "MAIN_DEFINES" and not prmt else key
                    defs = list(b.value(parent, dkey)) + [f"-DOKC_MULDEC={int(mul)}", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3"]
                    if prmt:
                        defs += ["-DOKC_PRMTSC=1"]
                    outs = [subprocess.run(["clang++", "-E", "-P", "-x", "c++", *defs, str(p)],
                                           capture_output=True, text=True, check=True).stdout for p in paths]
                    assert outs[0] == outs[1], (key, prmt, mul)
                    cases.append([key, prmt, mul])
    return {"identical_preprocessed_builds": len(cases), "cases": cases}


def histograms():
    # Valid ragged M384 histogram reaches79jobs, beyond both32/64 directory boundaries.
    hs = [[0] * 32, [321] * 9 + [129] + [1] * 22]
    for m in [1, 4, 8, 9, 16, 17, 31, 32, 33, 64, 65, 128, 256, 257, 384]:
        hs += [[m] + [0] * 31, [m] * 8 + [0] * 24,
               [min(m, 8)] * 32 if m >= 4 else [m] * 8 + [0] * 24]
    rng = random.Random(30)
    for m in [17, 32, 128, 256, 384]:
        for _ in range(4):
            h = [0] * 32
            for _ in range(m):
                for e in rng.sample(range(256), 8):
                    if e < 32: h[e] += 1
            hs.append(h)
    return hs


def jobs(h):
    off = 0
    fat, small = [], []
    for e, c in enumerate(h):
        if c > 8:
            fat += [(e, off + s, min(64, c - s)) for s in range(0, c, 64)]
        elif c:
            small.append((e, off, c))
        off += c
    return fat + small


def native_producer(mutant=None):
    source = (HERE / "submission.py").read_text()
    cu = b.value(source, "CUDA_SRC")
    fn = b.extract(cu, "template <bool PB, bool COLD>\n__device__ __forceinline__ void v30_producer(",
                   "// @V30_SEGMENT@" if "// @V30_SEGMENT@" in cu else "template <bool PB, bool COLD, int NTT>")
    # Three actual PTX invocations become CPU event recorders; all address/loop/partition code stays exact.
    matches = list(re.finditer(r'asm volatile\(.*? : "memory"\);', fn, re.S))
    assert len(matches) == 3
    replacements = ["emit_expect(S, bar, bytes, sl);",
                    "emit_copy(S, j, ub, st, seq, PB, COLD, true, src, dst, bytes, ring);",
                    "emit_copy(S, j, ub, st, seq, PB, COLD, false, src, dst, bytes, ring);"]
    for m, replacement in reversed(list(zip(matches, replacements))):
        fn = fn[:m.start()] + replacement + fn[m.end():]
    if mutant == "half":
        fn = fn.replace("(ub & 1) * 18432", "(ub & 1) * 9216")
    elif mutant == "stage":
        fn = fn.replace("st < 16", "st < 15")
    elif mutant == "directory":
        fn = fn.replace("const int offset = COLD ? 0 : S.njF;", "const int offset = 0;")
    pre = r'''
#include <cstdint>
#include <cstddef>
#include <vector>
#include <map>
#include <tuple>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#define __device__
#define __forceinline__ inline
using __nv_bfloat16=uint16_t;
struct Dim{int x;}; Dim threadIdx,blockIdx,gridDim;
constexpr int WGS_NS=3,WGS_XS=32768,UNIT_BYTES=36864,HD=4096,BW_F=0,BW_E=3,BX_F=6,BX_E=9;
struct Smem{int njF,njS,jE[88],jS[88],jC[88],jrow[89],forder[88]; uint64_t bars[24];};
struct Params{const unsigned char *wAs,*wBs; const uint16_t *xA,*xB;};
constexpr uintptr_t WA=0x100000000ull,WB=0x200000000ull,XA=0x300000000ull,XB=0x400000000ull;
using Key=std::tuple<int,int,int,int,int>;
std::map<Key,int> seen;
long long copies=0, bytesW=0, bytesX=0, waits=0;
void check(bool v){if(!v)throw std::runtime_error("native producer contract");}
void wgs_wait(uint64_t*,int phase,int){check(phase==0||phase==1);++waits;}
void emit_expect(Smem& s,uint64_t* bar,uint32_t bytes,int sl){check(bar==s.bars+sl);check(bytes>0&&bytes%16==0);}
void emit_copy(Smem& s,int j,int ub,int st,int seq,bool pb,bool cold,bool x,
               const void* src,const void* dst,uint32_t bytes,unsigned char* ring){
  check(s.jC[j]>0 && (cold ? s.jC[j]<=8:s.jC[j]>8));
  const int oldUnits=pb?16:32;
  int n=(s.jC[j]+7)/8*8,slot=seq%3;
  uintptr_t want;
  if(x){want=(pb?XB:XA)+(uintptr_t(s.jrow[j])*4096+st*n*256)*2;
    check(bytes==uint32_t(n*512));bytesX+=bytes;
    check(uintptr_t(dst)-uintptr_t(ring)==uintptr_t(3*(cold?18432:36864)+slot*(cold?4096:32768)));
  }else{int ou=cold?ub/2:ub;
    want=(pb?WB:WA)+uintptr_t((s.jE[j]*oldUnits+ou)*16+st)*36864+(cold?ub%2*18432:0);
    check(bytes==uint32_t(cold?18432:36864));bytesW+=bytes;
    check(uintptr_t(dst)-uintptr_t(ring)==uintptr_t(slot*(cold?18432:36864)));
  }
  check(uintptr_t(src)==want && want%16==0);check(st>=0&&st<16);
  check(++seen[Key(pb,j,ub,st,x)]==1);++copies;
}
'''
    main = r'''
int main(){try{
  int num;std::cin>>num;long long expectedW=0;int zero=0,cases=0;
  Params p{(const unsigned char*)WA,(const unsigned char*)WB,(const uint16_t*)XA,(const uint16_t*)XB};
  std::vector<unsigned char> ring(209000);
  while(num--){Smem s{};int nj;std::cin>>nj;int pad=0;
    for(int j=0;j<nj;++j){std::cin>>s.jE[j]>>s.jS[j]>>s.jC[j];s.jrow[j]=pad;pad+=(s.jC[j]+7)/8*8;}
    s.jrow[nj]=pad;check(pad<=3776);
    for(int j=0;j<nj;++j)if(s.jC[j]<=8)s.forder[s.njF++]=j;
    for(int j=0;j<nj;++j)if(s.jC[j]>8)s.forder[s.njF+s.njS++]=j;
    if(!nj)++zero;
    for(int g: {1,7,132,264}){
      seen.clear();gridDim.x=g;
      for(int c=0;c<g;++c){blockIdx.x=c;
        for(int xp=0;xp<2;++xp){threadIdx.x=256+xp*32;
          v30_producer<false,true>(p,s,ring.data(),xp);
          v30_producer<true,true>(p,s,ring.data(),xp);
          threadIdx.x=512+xp*32;
          v30_producer<false,false>(p,s,ring.data(),xp);
          v30_producer<true,false>(p,s,ring.data(),xp);
        }
      }
      for(int pb=0;pb<2;++pb)for(int j=0;j<nj;++j){int units=(pb?16:32)*(s.jC[j]<=8?2:1);
        for(int ub=0;ub<units;++ub)for(int st=0;st<16;++st)for(int x=0;x<2;++x)
          check(seen[Key(pb,j,ub,st,x)]==1);
      }
      expectedW+=long long(nj)*27*1024*1024; ++cases;
    }
  }
  check(bytesW==expectedW);
  std::cout<<"{\"cases\":"<<cases<<",\"copies\":"<<copies<<",\"weight_bytes\":"<<bytesW
    <<",\"activation_bytes\":"<<bytesX<<",\"ring_waits\":"<<waits<<",\"zero_histograms\":"<<zero<<"}";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 2;}}
'''.replace("long long(nj)", "static_cast<long long>(nj)")
    hs = histograms()
    stdin = str(len(hs)) + "\n" + "".join(str(len(jobs(h))) + "\n" +
              "".join("%d %d %d\n" % row for row in jobs(h)) for h in hs)
    with tempfile.TemporaryDirectory(prefix="v30-native-") as td:
        cpp, exe = Path(td)/"test.cpp", Path(td)/"test"
        cpp.write_text(pre + fn + main)
        subprocess.run(["clang++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)],
                       check=True, capture_output=True, text=True)
        proc = subprocess.run([str(exe)], input=stdin, text=True, capture_output=True, timeout=45)
        if mutant:
            assert proc.returncode == 2, (mutant, proc.stdout, proc.stderr)
            return {"mutant": mutant, "caught": True}
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)


def coverage():
    # Actual queue metadata is overwritten on every call. The tests deliberately alternate zero/live/skewed.
    maxj = maxpad = 0
    for h in histograms():
        js = jobs(h)
        cold = [j for j, (_, _, c) in enumerate(js) if c <= 8]
        hot = [j for j, (_, _, c) in enumerate(js) if c > 8]
        assert sorted(cold + hot) == list(range(len(js))) and not set(cold) & set(hot)
        assert sum(c for _, _, c in js) == sum(h)
        maxj = max(maxj, len(js)); maxpad = max(maxpad, sum((c+7)//8*8 for _, _, c in js))
        for e, c in enumerate(h):
            chunks = [(s, n) for ex, s, n in js if ex == e]
            assert sum(n for _, n in chunks) == c
    # Cold output rows coincide with the original two halves, including paired gate/up ordering.
    coords = 0
    for pb in [False, True]:
        for ub in range(32 if pb else 64):
            for wg in range(2):
                for wp in range(4):
                    assert ub*8+wg*4+wp == (ub//2)*16+((ub%2)*2+wg)*4+wp
                    coords += 1
    return {"histograms": len(histograms()), "max_jobs_tested": maxj, "max_padded_rows": maxpad,
            "cold_epilogue_coordinate_equalities": coords,
            "limits": "CPU job/count/coordinate evidence; actual warp routing and GPU memory order require official check"}


def run():
    result = {"status": "CPU_PASS", "source": source_checks(), "controls": preprocess_controls(),
              "coverage": coverage(), "native_actual_producer": native_producer(),
              "negative_controls": [native_producer(x) for x in ["half", "stage", "directory"]],
              "limits": ["No CUDA compiler/device used", "Native test replaces only producer PTX with recording callbacks",
                         "Per-unit FP32 association differs from Stream-K and is not certified numerically",
                         "Actual2CTA capacity, WGMMA scheduling and graph performance remain untested"]}
    return result


if __name__ == "__main__":
    result = run()
    (HERE / "cpu_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
