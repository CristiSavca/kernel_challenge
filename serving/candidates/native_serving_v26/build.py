"""Isolated V26 private boundary reduction, from immutable V25 source."""
from pathlib import Path
import ast, hashlib, json
HERE=Path(__file__).resolve().parent
PARENT_SHA='3a8c973b737dd074bcf5115f7418f5f0b34139fd4f9f637410ad1765ccbe050d'
EDITS=[]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def literal(source,name):
    t=ast.parse(source)
    n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets))
    return ast.literal_eval(n.value)
def change(s,a,b):
    assert s.count(a)==1,(a[:80],s.count(a))
    EDITS.append((a,b));return s.replace(a,b)
def build():
    EDITS.clear()
    parent=(HERE/'parent.py').read_text();assert sha(parent)==PARENT_SHA
    source=parent
    source=change(source,'constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW',
                        'constexpr size_t WGS_FIXF = (size_t)4 * WGS_SLOTS * 4 * WGS_FIXW')
    source=change(source,'''__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.fixbuf + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}''','''// Two immutable boundary payloads per CTA/phase; full units need no payload.
__device__ __forceinline__ float* wgs_partial(const Params& p, int pb, int c, int boundary, int wg) {
  return p.fixbuf + (size_t)(((pb * WGS_SLOTS + c) * 2 + boundary) * 4 + wg) * WGS_FIXW;
}
__device__ __forceinline__ float v26_sum(float a, float b) {
  float value;
  asm volatile("add.rn.f32 %0, %1, %2;\\n" : "=f"(value) : "f"(a), "f"(b));
  return value;
}
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x);
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);''')
    source=change(source,'return reinterpret_cast<int*>(p.fixbuf + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW)',
                        'return reinterpret_cast<int*>(p.fixbuf + (size_t)4 * WGS_SLOTS * 4 * WGS_FIXW)')
    a=source.index('  float* fb = wgs_fix(p, PB)');z=source.index('  if (wt == 0) *fc = 0;',a)
    source=change(source,source[a:z],'''  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  const int nj = S.njF + S.njS, G = (int)gridDim.x;
  const int begin = pd.u << 4, end = begin + 16;
  const int first = pd.slot, last = wgs_cta_of(S, PB, end - 1, G);
  const uint32_t total = PB ? S.wtotB : S.wtot;
  // Preserve the inherited publication protocol above. Every contributing WG
  // has finished its immutable stores before the owner reads these payloads.
  // Increasing CTA ranges are increasing original K-stage ranges, even when
  // phase A executes its trailing boundary first.
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float sum[2][4] = {};
    int g0 = wgs_stage_at(S, nj, PB, total * (uint32_t)first / (uint32_t)G);
#pragma unroll 1
    for (int c = first; c <= last; ++c) {
      const int g1 = c == G - 1 ? (nj << (9 - PB)) :
          wgs_stage_at(S, nj, PB, total * (uint32_t)(c + 1) / (uint32_t)G);
      if (g0 < end && g1 > begin && g1 > g0) {
        const int boundary = (g0 >> 4) == pd.u ? 0 : 1;
        const float* fb = wgs_partial(p, PB, c, boundary, wg) + wt * 4;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          if (nt0 + k < ntt) {
            const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
            sum[k][0] = v26_sum(sum[k][0], v.x);
            sum[k][1] = v26_sum(sum[k][1], v.y);
            sum[k][2] = v26_sum(sum[k][2], v.z);
            sum[k][3] = v26_sum(sum[k][3], v.w);
          }
        }
      }
      g0 = g1;
    }
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, sum[k]);
  }
''')
    a=source.index('// Integer CAS preserves the FP32');z=source.index('// one segment (stages',a)
    source=change(source,source[a:z],'// V26 uses immutable private partials and the ordered non-FTZ sum above.\n')
    source=change(source,'''    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) v25_add_f32(fb + nt * 512 + i, d[nt][i]);
    }''','''    const int boundary = u == (S.wgr[2 * PB] >> 4) ? 0 : 1;
    float* fb = wgs_partial(p, PB, (int)blockIdx.x, boundary, wg) + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(d[nt][0], d[nt][1], d[nt][2], d[nt][3]);''')
    source=change(source,'p.doneAs = reinterpret_cast<int*>(p.fixbuf + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW)',
                        'p.doneAs = reinterpret_cast<int*>(p.fixbuf + (size_t)4 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW)')
    source=change(source,'WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16',
                        'WGS_FIXF = 4 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16')
    for old,new in [('native_serving_v25','native_serving_v26'),('okc_serving_wgs_v25_','okc_serving_wgs_v26_'),('V25R big=','V26R big=')]:
        # Explicit complete identifier rename; retain all other inherited text.
        assert old in source
        EDITS.append((old,new));source=source.replace(old,new)
    marker='# V25 serving component: one WGS bank, exact Marlin operand/global boundaries,'
    source=change(source,marker,'# V26 private split reduction over frozen V25; one WGS bank and serving precision unchanged.\n'+marker)
    ast.parse(source)
    restored=source
    for old,new in reversed(EDITS):restored=restored.replace(new,old)
    assert restored==parent
    (HERE/'component.py').write_text(source)
    (HERE/'edits.json').write_text(json.dumps(EDITS,indent=2)+'\n')
    manifest={'parent_sha256':PARENT_SHA,'component_sha256':sha(source),'parent_cuda_sha256':sha(literal(parent,'CUDA_SRC')),
              'candidate_cuda_sha256':sha(literal(source,'CUDA_SRC')),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'edits_sha256':hashlib.sha256((HERE/'edits.json').read_bytes()).hexdigest(),
              'private_partial_bytes':41943040,'workspace_bytes':129005088,'weight_bank_bytes':905969664,
              'scope':'All-shape private boundary reduction; unchanged V25 math boundaries and serial workspace contract.'}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':build()
