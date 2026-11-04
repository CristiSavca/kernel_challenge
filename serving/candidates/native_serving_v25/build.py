"""Generate the isolated V25 WGS-only serving component from exact native247888."""
from pathlib import Path
import ast, hashlib, json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT=ROOT/'muse_moe_w4a16/submission.py'
PARENT_SHA='247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0'
EDITS=[]
def change(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a))
    EDITS.append((a,b));return s.replace(a,b)
def literal(tree,name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def build():
    EDITS.clear()
    parent=PARENT.read_text();assert hashlib.sha256(parent.encode()).hexdigest()==PARENT_SHA
    tree=ast.parse(parent);cuda=literal(tree,'CUDA_SRC');oldcuda=cuda
    cuda=change(cuda,'#if OKC_WGS && !(OKC_NW == 16 && !OKC_FAST1 && OKC_PRMTSC && !OKC_WS && OKC_FINCOL)\n#error "OKC_WGS: 16 warps, PRMT scales, column final pass, no FAST1 / WS"',
        '#if !OKC_WGS || !(OKC_NW == 16 && !OKC_FAST1 && !OKC_WS)\n#error "V25 requires the fixed WGS-only geometry"')
    cuda=change(cuda,'  const uint32_t* lut; const __nv_bfloat16* normw;',
        '  const __nv_bfloat16* marlin_gsA; const __nv_bfloat16* marlin_gsB;\n  int* local_out;\n  float* fixbuf;\n  const uint32_t* lut; const __nv_bfloat16* normw;')
    cuda=change(cuda,'  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;',
        '  return p.fixbuf + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;')
    cuda=change(cuda,'  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;',
        '  return reinterpret_cast<int*>(p.fixbuf + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;')
    a=cuda.index('template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile(')
    z=cuda.index('\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_flush(',a)
    old=cuda[a:z]
    new='''template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  // Match Marlin write_result: BF16(accumulator), then a BF16 global product.
  const __nv_bfloat16 gs = __ldg((PB ? p.marlin_gsB : p.marlin_gsA) + e);
  const __nv_bfloat162 gsp = __halves2bfloat162(gs, gs);
  if (!PB) {
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const __nv_bfloat162 v = __hmul2(__floats2bfloat162_rn(d4[h], d4[2 + h]), gsp);
        const float gt = __bfloat162float(v.x), up = __bfloat162float(v.y);
        // Stock activation is built with fast math; GEMM/fixups are not.
        const float silu = __fdividef(gt, 1.f + __expf(-gt));
        float act;
        asm volatile("mul.rn.ftz.f32 %0, %1, %2;\\n" : "=f"(act) : "f"(silu), "f"(up));
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(act));
      }
    }
  } else {
    const int n0 = r16 * 16 + g;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const int entry = S.ent[jst + tk];
        const __nv_bfloat162 v = __hmul2(__floats2bfloat162_rn(d4[h], d4[2 + h]), gsp);
        st_keep_bf16(p.out + (size_t)entry * HD + n0, v.x);
        st_keep_bf16(p.out + (size_t)entry * HD + n0 + 8, v.y);
      }
    }
  }
}
'''
    cuda=change(cuda,old,new.rstrip())
    cuda=change(cuda,'      scale_pairs[j] = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;',
'''#if OKC_PRMTSC
      scale_pairs[j] = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x3F803F80u;
#else
      const uint32_t low = S.lut[(sw >> (8u * sb)) & 255u];
      const uint32_t high = S.lut[(sw >> (8u * (sb + 1u))) & 255u];
      scale_pairs[j] = (low & 0xffffu) | (high << 16);
#endif''')
    cuda=change(cuda, '// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)',
'''// Integer CAS preserves the FP32 add's subnormal domain. red.global.add.f32
// flushes subnormals even without --use_fast_math, unlike stock Marlin's reducer.
__device__ __forceinline__ void v25_add_f32(float* address, float value) {
  unsigned* bits = reinterpret_cast<unsigned*>(address);
  unsigned old = atomicCAS(bits, 0u, 0u);
  for (;;) {
    const unsigned expected = old;
    float sum;
    asm volatile("add.rn.f32 %0, %1, %2;\\n" : "=f"(sum) : "f"(__uint_as_float(expected)), "f"(value));
    old = atomicCAS(bits, expected, __float_as_uint(sum));
    if (old == expected) return;
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)''')
    cuda=change(cuda, '''    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");'''.replace('::::', '::'),
'''    for (int nt = 0; nt < NTT; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) v25_add_f32(fb + nt * 512 + i, d[nt][i]);
    }''')
    cuda=change(cuda,'  wgs_prefix(S, nj);\n  wgs_range(S, nj, 0,',
'''  wgs_prefix(S, nj);
  if (nj == 0) {
    S.wgr[0] = S.wgr[1] = S.wgr[2] = S.wgr[3] = 0;
    return;  // all producer/consumer participants continue through their joins
  }
  wgs_range(S, nj, 0,''')
    cuda=change(cuda,'  asm volatile("griddepcontrol.wait;\\n" ::: "memory");','  // V25 uses a cooperative, ordinary ordered launch; PDL is disabled.')
    cuda=change(cuda,'  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");','  // No PDL trigger: the stock finalizer follows normal stream ordering.')
    a=cuda.index('  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {')
    z=cuda.index('\n  if (tid == 0) {',a)
    cuda=change(cuda,cuda[a:z],'''  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT)
    S.lut[i] = __ldg(p.lut + i);  // exact stock-Marlin transformed scale bits''')
    cuda=change(cuda, '    S.ssp = (gen & 1) ? p.ss1 : p.ss;', '    S.ssp = nullptr;  // V25 has no native norm accumulator')
    cuda=change(cuda,'  route(p, S);\n#if OKC_WGS && OKC_WGSPR',
'''  route(p, S);
  if (blockIdx.x == 0) {
    for (int i = tid; i < p.M * 8; i += NT) p.local_out[i] = (int)S.eid[i];
  }
#if OKC_WGS && OKC_WGSPR''')
    cuda=change(cuda,'''    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
''','    // V25 has no native norm-sum scratch to clear.\n')
    a=cuda.index('  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]')
    z=cuda.index('  DBG(5);',a)
    cuda=change(cuda,cuda[a:z],'''  // V25 stops at the BF16 expert-output boundary. The original stock
  // finalizer consumes p.out and the complete local-ID mask after this grid.
''')
    cuda=change(cuda, '__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p)', '__global__ void __launch_bounds__(NTL, 1) native_serving_v25(Params p)')
    a=cuda.index('static int g_okc_pdl = 0;')
    cuda=change(cuda,cuda[a:],(HERE/'host.cpp.inc').read_text())
    # Keep the original attribution and imports; replace the obsolete overview.
    prefix=parent[:parent.index('CUDA_SRC = r')]
    prefix=prefix.replace('# Isolated KPERM3 adaptation over frozen ef7b210b; no SC6/MFIN or pipeline changes.',
'''# V25 serving component: one WGS bank, exact Marlin operand/global boundaries,
# BF16 expert outputs, stock finalizer, and cooperative launch. Isolated prototype.
# Base247888 attribution/history follows; current modifications are recorded in manifest.json.''')
    helpers=[]
    names={'_stored_pos_mul','_pack_codes_nat','_wgs_kperm_index','_wgs_kperm_codes','_wgs_kperm_rows','_wgs_copy'}
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:helpers.append(ast.get_source_segment(parent,n))
    cpp='''#include <torch/extension.h>
int64_t okc_init();
void okc_run(std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
'''
    component=prefix+'CUDA_SRC = r\'\'\'\n'+cuda+"\n'''\n\nCPP_SRC = r'''\n"+cpp+"'''\n\n"+'\n\n'.join(helpers)+ '\n\n_POS_MUL = _stored_pos_mul()\n_POS = _POS_MUL\n\n'+(HERE/'adapter.py.inc').read_text()
    ast.parse(component)
    (HERE/'component.py').write_text(component)
    (HERE/'parent.py').write_text(parent)
    restore=cuda
    for old,new in reversed(EDITS):
        if new:
            assert restore.count(new)==1,(new[:80],restore.count(new));restore=restore.replace(new,old)
        else:
            # Empty deletion is checked by reconstructing the whole transform,
            # not an ambiguous global replacement of the empty string.
            pass
    assert restore == oldcuda, 'complete CUDA restoration'
    manifest={'parent_sha256':PARENT_SHA,'component_sha256':hashlib.sha256(component.encode()).hexdigest(),
              'parent_cuda_sha256':hashlib.sha256(oldcuda.encode()).hexdigest(),'candidate_cuda_sha256':hashlib.sha256(cuda.encode()).hexdigest(),
              'scope':'WGS-only serving component, not a production checkout integration',
              'weight_banks':1,'full_bank_bytes':905969664,'functions_retained_exact':sorted(names),
              'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'build.py',HERE/'host.cpp.inc',HERE/'adapter.py.inc']}}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':build()
