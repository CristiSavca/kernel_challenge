#!/usr/bin/env python3
"""Execute real parent/candidate table builders and extracted C++ escape loops.

CPU invariants only: no CUDA compiler, device race, numerical oracle, or speed proof.
"""
import ast
import contextlib
import hashlib
import io
import json
import operator
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from test_dense_escape_global_cpu import ring_model

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/candidates/muse_decode_layer/80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c.py"
CANDIDATE = ROOT / "muse_decode_layer/candidates/native_dense_stage_directory_v11.py"


def raw(v):
    return v.a if isinstance(v, Tensor) else v


class Tensor:
    def __init__(self, a): self.a = np.asarray(a)
    @property
    def shape(self): return self.a.shape
    @property
    def device(self): return "cpu"
    def numel(self): return self.a.size
    def long(self): return Tensor(self.a.astype(np.int64))
    def to(self, dtype): return Tensor(self.a.astype(dtype))
    def item(self): return self.a.item()
    def max(self): return Tensor(self.a.max())
    def contiguous(self): return Tensor(np.ascontiguousarray(self.a))
    def view(self, *args):
        if len(args) == 1 and args[0] in (np.int32, np.int64, np.float32, np.uint8):
            assert self.a.flags.c_contiguous
            return Tensor(self.a.view(args[0]))
        return Tensor(self.a.reshape(*args))
    def __getitem__(self, key):
        if isinstance(key, tuple): key = tuple(raw(k) for k in key)
        return Tensor(self.a[raw(key)])
    def __setitem__(self, key, value):
        if isinstance(key, tuple): key = tuple(raw(k) for k in key)
        self.a[raw(key)] = raw(value)


for name, fn in {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
                 "floordiv": operator.floordiv, "mod": operator.mod, "lt": operator.lt,
                 "lshift": operator.lshift, "or": operator.or_}.items():
    setattr(Tensor, "__" + name + "__", lambda self, other, fn=fn: Tensor(fn(self.a, raw(other))))
    setattr(Tensor, "__r" + name + "__", lambda self, other, fn=fn: Tensor(fn(raw(other), self.a)))


class Torch:
    int32, int64, long = np.int32, np.int64, np.int64
    @staticmethod
    def arange(*args, device=None): return Tensor(np.arange(*args, dtype=np.int64))
    @staticmethod
    def zeros(*shape, dtype=None, device=None):
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        return Tensor(np.zeros(shape, dtype=dtype))
    @staticmethod
    def bincount(x, minlength=0): return Tensor(np.bincount(x.a, minlength=minlength))
    @staticmethod
    def argsort(x): return Tensor(np.argsort(x.a, kind="quicksort"))
    @staticmethod
    def cumsum(x, dim): return Tensor(np.cumsum(x.a, axis=dim))
    @staticmethod
    def searchsorted(x, values): return Tensor(np.searchsorted(x.a, values.a, side="left"))
    @staticmethod
    def cat(values, dim=0): return Tensor(np.concatenate([v.a for v in values], axis=dim))


def key(node):
    if isinstance(node, ast.FunctionDef): return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id


def literal(tree, name):
    return ast.literal_eval(next(n.value for n in tree.body if key(n) == name))


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def namespace(tree):
    wanted = {"bfp_SK2_CONFIGS", "bfp_SK2_PICK", "bfp_SK2_K", "bfp_SK2_SRC",
              "SK2B", "SK2B_UN_TOKR", "SK2B1", "SK2B_O_TOKR"}
    funcs = {"bfp__sk2_geom", "bfp__sk2_directory_specs", "bfp_sk2_table", "bfp__sk2_src"}
    nodes = [n for n in tree.body if key(n) in wanted | funcs
             or key(n) == "_bfp__sk2_src_unfused"]
    env = {"torch": Torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual-helpers>", "exec"), env)
    return env


def block(source, start):
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def escape_loop(source):
    start = source.index("      const int cnt = ptab_s[0].x;")
    loop = source.index("      for (int st = 0; st < nstt; ++st)", start)
    end = loop + len(block(source, loop))
    result = source[start:end].replace("constexpr int KST = UPS * 64;", "const int KST = UPS * 64;")
    # Instrument the exact source input-address expression; keep traversal and
    # arithmetic untouched. The helper records entry and ring-byte address.
    result = result.replace("const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(",
                            "const __nv_bfloat16 v = record_read(pe, reinterpret_cast<const __nv_bfloat16*>(")
    result = result.replace("+ (kc & 7) * 2);", "+ (kc & 7) * 2));")
    return result


def cpp_trace_proof(parent_src, new_src, cases):
    base = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>
struct int2 {int x,y;}; using __nv_bfloat16=uint16_t;
struct Read {int entry;int64_t addr;bool operator==(const Read& o)const{return entry==o.entry&&addr==o.addr;}};
std::vector<Read> reads; std::vector<std::tuple<int,int,int>> events;
const unsigned char* baseptr;
float __int_as_float(int a){float f;std::memcpy(&f,&a,4);return f;}
float __bfloat162float(__nv_bfloat16 a){return float(a);}
__nv_bfloat16 record_read(int pe,const __nv_bfloat16* p){reads.push_back({pe,reinterpret_cast<const unsigned char*>(p)-baseptr});return *p;}
void __syncwarp(){events.push_back({2,0,0});}
void SK2_WAIT(int bar,int phase){events.push_back({0,bar,phase});}
void mbar_arrive(int bar){events.push_back({1,bar,0});}
'''
    args = "const int2* ptab_s,int UPS,int NESC,int NSTAGE,int nst,int nstt,bool FUSE,int gs0,int CL,int ew,int lane,int M,int CP,int XU,int Mb,int SST,const unsigned char* gbase,float* cr"
    for name, src in (("old_loop", parent_src), ("new_loop", new_src)):
        base += "void " + name + "(" + args + "){int full0=0,empty0=1024;\n" + escape_loop(src) + "\n}\n"
    base += r'''
int main(int argc,char**argv){
  std::ifstream in(argv[1],std::ios::binary);int nc;in.read((char*)&nc,4);size_t checked=0,evchecked=0;
  for(int test=0;test<nc;++test){
    int meta[9];in.read((char*)meta,sizeof(meta));int K=meta[0],R=meta[1],UPS=meta[2],NESC=meta[3],depth=meta[4],M=meta[5],pa=meta[6],pb=meta[7],fuse=meta[8];
    std::vector<int2>a(pa),b(pb);in.read((char*)a.data(),pa*8);in.read((char*)b.data(),pb*8);
    int nst=K/(UPS*64),nall=nst+(fuse?4:0),sst=UPS*M*128+4096;
    std::vector<uint64_t> mem((depth*sst+7)/8,0);baseptr=(unsigned char*)mem.data();
    // Valid first/last physical token and one padded lane. UV uses the same
    // physical M; its provenance does not change the shared byte layout.
    for(int lane:std::vector<int>{0,M-1,M})for(int ew=0;ew<NESC;++ew){
      if(lane>=32)continue;std::vector<float>corr(32*(R+1),0.f);
      reads.clear();events.clear();old_loop(a.data(),UPS,NESC,depth,nst,nall,fuse,0,1,ew,lane,M,R+1,M*128,M,sst,baseptr,corr.data());auto rd=reads;auto ev=events;
      reads.clear();events.clear();new_loop(b.data(),UPS,NESC,depth,nst,nall,fuse,0,1,ew,lane,M,R+1,M*128,M,sst,baseptr,corr.data());
      assert(reads==rd&&events==ev);checked+=rd.size();evchecked+=ev.size();
      assert(ev.size()==size_t(nall*(2+(lane==0))));
    }
    // If an inherited clustered caller receives a decorated table, CL>1 must
    // retain the old local-stage ownership and skip pre-gs0 entries identically.
    if(nst>=2){int gs0=1,part=nst-1;for(int ew=0;ew<NESC;++ew){std::vector<float>corr(32*(R+1),0.f);
      reads.clear();events.clear();old_loop(a.data(),UPS,NESC,depth,part,part,false,gs0,2,ew,0,M,R+1,M*128,M,sst,baseptr,corr.data());auto rd=reads;auto ev=events;
      reads.clear();events.clear();new_loop(b.data(),UPS,NESC,depth,part,part,false,gs0,2,ew,0,M,R+1,M*128,M,sst,baseptr,corr.data());assert(reads==rd&&events==ev);checked+=rd.size();evchecked+=ev.size();
    }}
  }
  std::cout<<"{\"trace_entry_address_pairs\":"<<checked<<",\"barrier_events\":"<<evchecked<<",\"table_geometry_cases\":"<<nc<<"}\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="okc-stage-dir-") as td:
        exe, data = Path(td) / "trace", Path(td) / "cases.bin"
        with data.open("wb") as f:
            f.write(np.asarray([len(cases)], np.int32).tobytes())
            for meta, old, new in cases:
                f.write(np.asarray(meta, np.int32).tobytes())
                f.write(old.astype(np.int32).tobytes()); f.write(new.astype(np.int32).tobytes())
        subprocess.run(["clang++", "-std=c++17", "-O2", "-x", "c++", "-o", str(exe), "-"],
                       input=base, text=True, capture_output=True, check=True)
        run = subprocess.run([str(exe), str(data)], text=True, capture_output=True, check=True)
    return json.loads(run.stdout)


def main():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT.stem
    candidate_sha = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    pt, ct = ast.parse(PARENT.read_text()), ast.parse(CANDIDATE.read_text())
    changed = {"bfp_SK2_SRC", "bfp__get_ext", "bfp_sk2_table", "bfp_sk2_prepare"}
    added = {"bfp__sk2_directory_specs"}
    dump = lambda n: ast.dump(n, include_attributes=False)
    assert [dump(n) for n in pt.body if key(n) not in changed] == [dump(n) for n in ct.body if key(n) not in changed | added]
    ps, cs = literal(pt, "bfp_SK2_SRC"), literal(ct, "bfp_SK2_SRC")
    # Every byte outside the exact escape traversal and kernel namespace is old.
    old_loop, new_loop = escape_loop(ps), escape_loop(cs)
    # Perform equality on uninstrumented source regions surrounding the loop.
    def strip_loop(src):
        start = src.index("      const int cnt = ptab_s[0].x;")
        pos = src.index("      for (int st = 0; st < nstt; ++st)", start)
        end = pos + len(block(src, pos))
        return src[:start] + "<reviewed traversal>" + src[end:]
    assert strip_loop(cs).replace("sk2_stage_directory_kernel", "sk2_kernel") == strip_loop(ps)
    assert "const __nv_bfloat16 v = x[" not in new_loop
    assert cs.count("mbar_init(cbar, 32 * NESC)") == ps.count("mbar_init(cbar, 32 * NESC)") == 2
    kernel = block(cs, cs.index("sk2_stage_directory_kernel("))
    static_declarations = re.findall(r"^\s*(?:extern )?__shared__[^;]*;", kernel, re.M)
    assert len(static_declarations) == 3, static_declarations
    assert any("uv_ra[32], uv_rs[32]" in x for x in static_declarations)
    assert any("int s_rlast" in x for x in static_declarations)
    assert any("extern __shared__ __align__(1024)" in x for x in static_declarations)
    assert 32 * 4 * 2 + 4 + 1023 <= 2048
    prep_text = ast.unparse(function(ct, "bfp_sk2_prepare"))
    assert "shared_memory_per_block_optin" in prep_text and "- 2048" in prep_text
    pe, ce = namespace(pt), namespace(ct)
    specs = ce["bfp__sk2_directory_specs"]()
    emitted = ce["bfp__sk2_src"](True)
    prefixes = re.findall(r"launch_sk2<(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), true, false, (\d+), (\d+),", emitted)
    emitted_geometries = {tuple(map(int, p)) for p in prefixes}
    # Independently enumerate actual generated dispatch lines, including the
    # forced fused pooler, UV wrappers, and BND wrappers. Do not rely solely on
    # the new helper's own consumer list for its min-NESC/resource proof.
    actual_consumers = {skey: set() for skey in specs}
    for line in emitted.splitlines():
        hit = re.search(r"launch_sk2<(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), true, false, (\d+), (\d+),", line)
        if not hit:
            continue
        mt, tokr, nt, nwr, wsk, upw, depth, nacc, cl = map(int, hit.groups())
        cfg_id = re.search(r"cfg == (\d+)", line)
        boundary = re.search(r"layer == (\d+)", line)
        if cfg_id:
            cfg = ce["bfp_SK2_CONFIGS"][int(cfg_id[1])]
        elif boundary:
            layer = "unpooler" if int(boundary[1]) == 0 else "o"
            name = ce["bfp_SK2_PICK"][(layer, tokr)]
            cfg = next(c for c in ce["bfp_SK2_CONFIGS"] if c[0] == name)
        else:
            cfg = next(c for c in ce["bfp_SK2_CONFIGS"] if c[0] == "pooler")
        name, layer, nrows, R, *_ = cfg
        skey = (layer, nrows, R, nt, nwr, wsk, upw)
        actual_consumers[skey].add((mt, tokr, depth, cl))
    for skey, consumers in actual_consumers.items():
        assert consumers
        layer, _, R, nt, nwr, wsk, upw = skey
        assert specs[skey]["kst"] == 64 * wsk * upw
        assert specs[skey]["min_nesc"] == min(min(4, d) for _, _, d, _ in consumers)
        fixed_sizes = []
        for mt, tokr, depth, cl in consumers:
            assert cl == 1
            # Independent formula in terms of R and UPS, rather than the helper's
            # expanded warp/tile product. Matches the unchanged launch_sk2.
            ups, toks, ne = wsk * upw, 16 * mt, min(4, depth)
            slot = ((ups * tokr * 128 + R * ups * 96 + 1023) // 1024) * 1024
            fixed_sizes.append(1024 + depth * slot + 16 * depth + 32
                               + ne * (((toks * (R + 1) + 3) // 4) * 4) * 4 + 128)
        assert specs[skey]["fixed_max"] == max(fixed_sizes)
    covered = set()
    cases, selections, entry_bytes = [], [], 0
    rng = np.random.default_rng(112604)
    for skey, spec in specs.items():
        layer, nrows, R, nt, nwr, wsk, upw = skey
        K, S = ce["bfp_SK2_K"][layer], spec["stages"]
        # Two CTAs expose unequal counts and empty individual CTAs in a selected table.
        ncfg = R * 2
        kk_cases = [np.array([], np.int64), np.array([0], np.int64),
                    np.clip(np.repeat(np.arange(S + 1) * spec["kst"], 3) + np.tile([-1, 0, 1], S + 1), 0, K - 1),
                    rng.integers(0, K, max(256, S * 32)), np.repeat(np.arange(K), 2)]
        for ci, kk in enumerate(kk_cases):
            rr = rng.integers(0, ncfg, len(kk)) if ci != 3 else rng.integers(0, R, len(kk))
            delta = (np.arange(len(kk), dtype=np.float32) % 17 - 8) * .125
            comp = {"rows": Tensor(rr), "cols": Tensor(kk), "delta": Tensor(delta),
                    "packed": Tensor(np.empty((ncfg // 8, K // 64, 0), np.uint8))}
            old, pold = pe["bfp_sk2_table"](comp, ncfg, R)
            with contextlib.redirect_stdout(io.StringIO()):
                off, poff = ce["bfp_sk2_table"](comp, ncfg, R, spec, 0)
                normal, pnormal = ce["bfp_sk2_table"](comp, ncfg, R, spec, 230400)
                # Capacity stress permits a decorated dense table too large for
                # any real launch; normal calls above enforce the real bound.
                new, pnew = ce["bfp_sk2_table"](comp, ncfg, R, spec, 1 << 30)
            dbytes = ((4 * (S + 1) + 15) // 16) * 16
            work_saved = 4 * (spec["min_nesc"] - 1) * len(kk)
            work_added = 2 * (dbytes + 8 * S)
            needed = spec["fixed_max"] + 8 * (pold + dbytes // 8)
            expected = work_saved > work_added and needed <= 230400
            assert bool(normal.a[0, 0, 1]) == expected
            if work_saved > work_added:
                with contextlib.redirect_stdout(io.StringIO()):
                    below, below_p = ce["bfp_sk2_table"](comp, ncfg, R, spec, needed - 1)
                    exact, exact_p = ce["bfp_sk2_table"](comp, ncfg, R, spec, needed)
                assert below_p == pold and np.array_equal(below.a, old.a)
                assert exact_p == pold + dbytes // 8 and int(exact.a[0, 0, 1]) == pold
            assert poff == pold and np.array_equal(off.a, old.a)
            for candidate, pn in ((normal, pnormal), (new, pnew)):
                offset = int(candidate.a[0, 0, 1])
                if not offset:
                    assert pn == pold and np.array_equal(candidate.a, old.a)
                else:
                    assert offset == pold and pn % 2 == 0 and pn * 8 % 16 == 0
                    assert np.array_equal(candidate.a[:, 0, 0], old.a[:, 0, 0])
                    assert np.array_equal(candidate.a[:, 1:pold, :], old.a[:, 1:, :])
                    for cta in range(2):
                        count = int(old.a[cta, 0, 0]); keys = old.a[cta, 1:count + 1, 0] & 65535
                        d = candidate.a[cta, offset:, :].reshape(-1)[:S + 1]
                        assert np.array_equal(d, 1 + np.searchsorted(keys, np.arange(S + 1) * spec["kst"]))
                        assert d[0] == 1 and d[-1] == count + 1 and np.all(d[:-1] <= d[1:])
                    entry_bytes += 2 * (pold - 1) * 8
            selections.append({"table": skey, "case": ci, "escapes": len(kk), "selected_at_real_budget": bool(normal.a[0, 0, 1]),
                               "old_pmax": pold, "new_pmax": pnormal})
            for name, tokr, depth, ne, ordinary, fused in spec["consumers"]:
                cfg = next(c for c in ce["bfp_SK2_CONFIGS"] if c[0] == name)
                nacc = 1 if tokr == 32 and depth >= 4 and nt == 4 else cfg[8]
                covered.add((2 if tokr == 32 else 1, tokr, nt, nwr, wsk, upw, depth, nacc, cfg[9]))
                for M in range(1, tokr + 1):
                    for cta in range(2):
                        cases.append(([K, R, wsk * upw, ne, depth, M, pold, pnew, int(fused)], old.a[cta], new.a[cta]))
    assert emitted_geometries <= covered, (emitted_geometries - covered)
    assert len(emitted_geometries) >= 14
    assert any(x["selected_at_real_budget"] for x in selections)
    assert any(not x["selected_at_real_budget"] and x["escapes"] for x in selections)
    # Directory does not remove any ring consumer; model NW+NESC participants,
    # including intentionally delayed correction warps and fused-router stages.
    rings = 0
    for skey, spec in specs.items():
        _, _, _, _, nwr, wsk, _ = skey
        for _, _, depth, ne, _, fused in spec["consumers"]:
            for seed in range(32):
                assert ring_model(nwr * wsk + ne, depth, spec["stages"] + (4 if fused else 0), seed)
                rings += 1
    assert any(not ring_model(8, 3, 16, seed, -1) for seed in range(32))
    assert not ring_model(8, 3, 16, 7, 1)
    traces = cpp_trace_proof(ps, cs, cases)
    assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() == candidate_sha
    print(json.dumps({"status": "pass", "candidate_sha256": candidate_sha, "parent_sha256": PARENT.stem,
                      "changed_nodes": sorted(changed), "added_nodes": sorted(added),
                      "emitted_geometries": len(emitted_geometries), "immutable_shared_tables": len(specs),
                      "static_shared_reserve_bytes": 2048, "tested_dynamic_budget": 230400,
                      "table_cases": len(selections), "preserved_entry_bytes": entry_bytes,
                      "ring_interleavings": rings, "negative_controls": 2,
                      "extracted_cpp": traces, "selection_cases": selections,
                      "limits": "Actual AST table helpers through NumPy adapter and actual extracted C++ traversal with instrumented reads; not Torch/CUDA ordering, device compiler, GPU numerical output or performance."}, indent=2))


if __name__ == "__main__":
    main()
