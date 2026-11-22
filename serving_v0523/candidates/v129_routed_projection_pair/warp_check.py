"""Exercise the shipped ascending-K projections, including their original-slot memory accesses."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v122_clustered_expert_mlp"))
import independent_cpu as base

v86 = base.v86

DRIVER = r"""
using namespace sglang::muse_route_pair;
extern "C" void simulate(const bf16_t* x, const uint32_t* gate, const uint8_t* gs, const bf16_t* gscale,
                         const uint32_t* down, const uint8_t* ds, const bf16_t* dscale, bf16_t* act, bf16_t* out,
                         const int32_t* plan, uint32_t h, uint32_t inner, uint32_t top, uint32_t groups) {
  Projection up{x, act, {gate,gs}, gscale, plan, h, inner, top};
  Projection dn{act, out, {down,ds}, dscale, plan, inner, h, 1};
  for (unsigned phase=0;phase<2;++phase) {
    const auto& p=phase?dn:up;
    for (unsigned cohort=0;cohort<groups;++cohort) {
      for (unsigned j=cohort;j<static_cast<unsigned>(plan[0]);j+=groups) {
        for(unsigned band=0;band<p.output_width;band+=64) {
          Routine lanes[32];
          for (auto& lane:lanes) lane=phase?sim<false>(p,plan+1+j*9,band):sim<true>(p,plan+1+j*9,band);
          for (;;) {
            for(unsigned n=0;n<32;++n){threadIdx.x=n;lanes[n].h.resume();}
            bool done=lanes[0].h.done();
            for(auto& lane:lanes) if(lane.h.done()!=done)std::abort();
            if(done) break;
            warp_mma(lanes);
          }
          for(auto& lane:lanes)lane.h.destroy();
        }
      }
    }
  }
}
"""


def compile_device(stage, tmp):
    path = stage / "python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh"
    src = path.read_text()
    pre = src[src.index("namespace sglang {"):src.index("SGL_DEVICE void contract(")]
    pre, count = re.subn(
        r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
        r"\1 = pair_mul(\2, \3);", pre)
    assert count == 2
    contract = base.body(src, "void contract(")
    contract, count = re.subn(r"asm volatile\(.*?\)\);", "co_yield Op{&d,{a0,a1,a2,a3},{lo,hi}};", contract, flags=re.S)
    assert count == 1
    body = base.body(src, "void project(")

    def inline(call):
        out, addr = call.groups()
        inner = re.sub(r"\bbase\b", "address", contract)
        return f"{{auto& out={out}; const Bank bank=p.bank; size_t address={addr};" + inner + "}"

    body, count = re.subn(r"contract\((\w+), p.bank, ([^,]+), lo, hi\);", inline, body)
    assert count == 2
    types = src[src.index("SGL_DEVICE float scaled_bf16"):src.index("template <bool Gate>")]
    fn = "template<bool Gate> Routine sim(const Projection& p,const int32_t* job,unsigned column){" + body + "}"
    code = v86.PRELUDE + base.v89.HOST_MUL + pre + types + fn + "\n}}\n"
    code += v86.POSTLUDE.split('extern "C" void simulate')[0] + DRIVER
    cpp, lib = tmp / "warp.cpp", tmp / "warp.so"
    cpp.write_text(code)
    subprocess.run(["c++", "-O2", "-std=c++20", "-fno-strict-aliasing", "-shared", "-fPIC", str(cpp), "-o", str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.simulate.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_uint32] * 4
    dll.simulate.restype = None
    return dll, hashlib.sha256(src.encode()).hexdigest()


def plan_jobs(ids, experts, rng):
    # Randomize inter-expert order exactly as independent warp atomic allocation may.
    jobs = []
    for e in rng.permutation(experts):
        alive = np.flatnonzero(ids.ravel() == e)
        for start in range(0, len(alive), 8):
            tile = alive[start:start + 8].tolist()
            jobs.append([e] + tile + [-1] * (8 - len(tile)))
    return np.array([len(jobs), *(i for j in jobs for i in j)], dtype=np.int32)


def run_case(dll, rng, h, inner, experts, ids, groups):
    tokens, top = ids.shape
    gate = base.weight_bank(rng, experts, 2 * inner, h)
    down = base.weight_bank(rng, experts, h, inner)
    x = v86.bf16(rng.integers(-8, 9, (tokens, h)).astype(np.float32) / 16)
    before = x.copy()
    choices = np.array([0.0625, 0.75, 1, 1.5], np.float32)
    gscale, dscale = (v86.bf16(rng.choice(choices, experts)) for _ in range(2))
    active = np.full((ids.size, inner), 0x7FC1, dtype=np.uint16)
    outputs = np.full((ids.size, h), 0x7FC2, dtype=np.uint16)
    exp_act, exp_out = active.copy(), outputs.copy()
    for e in range(experts):
        slots = np.flatnonzero(ids.ravel() == e)
        if not slots.size:
            continue
        src = v86.floats(x[slots // top])
        accum = base.reference(src, gate[0][e], gate[1][e])
        scaled = v86.floats(v86.bf16(v86.floats(accum) * v86.floats(gscale[e])))
        g, u = np.split(scaled, 2, axis=1)
        with np.errstate(over="ignore"):
            exp_act[slots] = v86.bf16(g / (1 + np.exp(-g)) * u)
        accum = base.reference(v86.floats(exp_act[slots]), down[0][e], down[1][e])
        exp_out[slots] = v86.bf16(v86.floats(accum) * v86.floats(dscale[e]))
    plan = plan_jobs(ids, experts, rng)
    buffers = [x, gate[2], gate[3], gscale, down[2], down[3], dscale, active, outputs, plan]
    dll.simulate(*(a.ctypes.data for a in buffers), h, inner, top, groups)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(active, exp_act)
    np.testing.assert_array_equal(outputs, exp_out)
    live = int(np.count_nonzero((ids >= 0) & (ids < experts)))
    return dict(hidden=h, intermediate=inner, tokens=tokens, topk=top, experts=experts,
                groups=groups, tiles=int(plan[0]), live_routes=live, active_bf16=live*inner,
                output_bf16=live*h, preserved_inactive=(ids.size-live)*(inner+h), preserved_input=x.size)


if __name__ == "__main__":
    rng = np.random.default_rng(129)
    cases = [(128,128,2,np.full((7,3),-1),1),
             (256,128,4,rng.integers(-1,5,(11,3)),1),
             (128,512,3,np.zeros((19,2),dtype=np.int32),2),
             (512,256,3,rng.integers(-2,4,(8,8)),2),
             (4096,4096,1,np.array([[-1,0],[-1,-1]]),1)]
    with tempfile.TemporaryDirectory(prefix="v129-warp-") as tmp:
        dll, digest = compile_device(Path(sys.argv[1]), Path(tmp))
        results = [run_case(dll,rng,*args) for args in cases]
    print(json.dumps(dict(header_sha256=digest,cases=results,cuda_execution=False),indent=2))
