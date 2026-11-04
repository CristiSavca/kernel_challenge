#!/usr/bin/env python3
"""Build one fixed shared-A experiment from the immutable scored MoE parent."""
import ast
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"
PARENT = ROOT / f"experiments/candidates/muse_moe_w4a16/{PARENT_SHA}.py"
CANDIDATE = ROOT / "muse_moe_w4a16/candidates/native_wgs_shared_a_v15.py"


def once(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new)


def wrappers():
    out = ["template <int NTT> struct WGSMMA;\n"]
    for n in range(1, 9):
        out.append(f"template <> struct WGSMMA<{n}> {{\n")
        for name, zero, constraint in (("mma0", 0, "=f"), ("mma", 1, "+f")):
            regs = ", ".join(f"%{i}" for i in range(4*n))
            outputs = ", ".join(f'"{constraint}"(d[{i//4}][{i%4}])' for i in range(4*n))
            out.append(f'''  static __device__ __forceinline__ void {name}(float (&d)[WGS_NT][4], uint64_t adesc, uint64_t bdesc) {{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n{8*n}k16.f32.bf16.bf16 {{{regs}}}, %{4*n}, %{4*n+1}, {zero}, 1, 1, 0, 0;\\n"
                 : {outputs} : "l"(adesc), "l"(bdesc) : "memory");
#endif
  }}
''')
        out.append("};\n")
    return "".join(out).rstrip("\n")


A_HELPER = r'''
// Exact original RS fragment pairs, scattered to a K-major m64k16 BF16 tile.
// r0/r1 differ by row+8; r0/r2 differ by K+8. Each store keeps both BF16 halves.
__device__ __forceinline__ void wgs_a_store(uint32_t tile, int wp, int lane,
                                           uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  const uint32_t off = (uint32_t)(wp * 512 + (lane >> 2) * 16 + (lane & 3) * 4);
  asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(tile + off), "r"(r0) : "memory");
  asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(tile + off + 256), "r"(r1) : "memory");
  asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(tile + off + 128), "r"(r2) : "memory");
  asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(tile + off + 384), "r"(r3) : "memory");
}
'''


HOST = r'''
#if OKC_WGS
// Once per compiled WGS module, outside graph replay. Valid admission only:
// register/local allocation is evidence, not an automatic performance verdict.
static void okc_shared_a_resources(size_t sm, cudaError_t set0, cudaError_t set1) {
  cudaFuncAttributes a0 = {}, a1 = {};
  int occ0 = -1, occ1 = -1;
  cudaError_t f0 = cudaFuncGetAttributes(&a0, okc::moe_fused<false>);
  cudaError_t f1 = cudaFuncGetAttributes(&a1, okc::moe_fused<true>);
  cudaError_t o0 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ0, okc::moe_fused<false>, okc::NTL, sm);
  cudaError_t o1 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1, okc::moe_fused<true>, okc::NTL, sm);
  fprintf(stderr, "SSAR v=1 big=0 ntl=%d dyn=%zu regs=%d local=%zu static=%zu maxdyn=%d occ=%d set=%d attr=%d cap=%d\n",
          okc::NTL, sm, a0.numRegs, a0.localSizeBytes, a0.sharedSizeBytes, a0.maxDynamicSharedSizeBytes,
          occ0, (int)set0, (int)f0, (int)o0);
  fprintf(stderr, "SSAR v=1 big=1 ntl=%d dyn=%zu regs=%d local=%zu static=%zu maxdyn=%d occ=%d set=%d attr=%d cap=%d\n",
          okc::NTL, sm, a1.numRegs, a1.localSizeBytes, a1.sharedSizeBytes, a1.maxDynamicSharedSizeBytes,
          occ1, (int)set1, (int)f1, (int)o1);
  TORCH_CHECK(set0 == cudaSuccess && set1 == cudaSuccess && f0 == cudaSuccess && f1 == cudaSuccess &&
              o0 == cudaSuccess && o1 == cudaSuccess && occ0 >= 1 && occ1 >= 1,
              "shared-A WGS resource admission failed; inspect SSAR records");
}
#endif
'''


def modify_cuda(s):
    s = once(s, "// ---------------------------------------------------------------- WGS: RS-wgmma for every job", "// ---------------------------------------------------------------- WGS: SS-wgmma for every job")
    s = once(s, "// (A = dequantized codes in registers, N = 8 * n-tiles)", "// (A = exact dequantized codes in two small shared banks, N = 8 * n-tiles)")
    s = once(s, "#define OKC_WGSNX 3                             // x ring slots", "#define OKC_WGSNX 2                             // fixed shared-A experiment: two X slots")
    s = once(s, "constexpr int WGS_NS = 3;", "constexpr int WGS_NS = 2;")
    s = once(s, 'static_assert(WGS_NS == WGS_NX, "Joined completion requires equal ring depths");', 'static_assert(WGS_NS == 2 && WGS_NX == 2, "Shared-A uses exactly two joined ring slots");')
    s = once(s, "constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices", "constexpr int WGS_A_BANK_BYTES = 4 * 2 * 64 * 16 * 2;  // four WG-private GS2 tiles\nconstexpr int WGS_A_BYTES = 2 * WGS_A_BANK_BYTES;\nconstexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2 * WGS_NS, BX_E = 2 * WGS_NS + WGS_NX;  // barrier indices")
    a = s.index("template <int NTT> struct WGSMMA;")
    b = s.index("\n};", s.index("template <> struct WGSMMA<8>")) + len("\n};")
    # Stop at the eighth wrapper, preserving the following non-template producers.
    s = s[:a] + wrappers() + A_HELPER + s[b:]
    a = s.index("template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(")
    b = s.index("\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_phase(", a)
    seg = s[a:b]
    seg = once(seg, "  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas\n  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time", "  // Preserve the parent's GS2 issue schedule. Its historical RS register/serialization notes\n  // do not establish this 576-thread shared-A kernel's compiled allocation or scheduling.")
    seg = once(seg, "  uint32_t af[2][GS][4];", "  static_assert(GS == 2, \"Shared-A banks preserve the incumbent GS2 schedule\");\n  const uint32_t aring_s = smem_u32(ring + WGS_NS * UNIT_BYTES + WGS_NX * WGS_XS);")
    seg = once(seg, "        const int ci = lg * (4 / GS) + h;  // commit group index in the stage", "        const int ci = lg * (4 / GS) + h;  // commit group index in the stage\n        const uint32_t atile = aring_s + (ci & 1) * WGS_A_BANK_BYTES + wg * 4096;\n        // Every lane has retired this bank's preceding use before any lane overwrites it.\n        asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");")
    seg = once(seg, "          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);", "          wgs_a_store(atile + jj * 2048, wp, lane, bmul2l(r0, sgp), bmul2h(r1, sgp),\n                      bmul2l(r2, sgp), bmul2h(r3, sgp));")
    seg = once(seg, "        wg_fence();", "        // Publish generic shared stores to WGMMA's async proxy, then join all 128 writers.\n        asm volatile(\"fence.proxy.async.shared::cta;\\n\" ::: \"memory\");\n        asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");\n        const uint64_t ad = wg_desc(atile, 128, 256);\n        wg_fence();")
    assert seg.count("af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3]") == 4
    seg = seg.replace("af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3]", "ad + (uint64_t)(jj * 128)")
    assert seg.count("af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3]") == 2
    seg = seg.replace("af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3]", "ad")
    s = s[:a] + seg + s[b:]
    s = once(s, "return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;", "return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS + okc::WGS_A_BYTES;")
    s = once(s, "void okc_run(std::vector<torch::Tensor>", HOST + "\nvoid okc_run(std::vector<torch::Tensor>")
    s = once(s, "    attr = true;", "#if OKC_WGS\n    okc_shared_a_resources(sm, e0, e1);\n#endif\n    attr = true;")
    return s


def build():
    raw = PARENT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    text = raw.decode(); tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))
    old = ast.literal_eval(node.value)
    text = once(text, old, modify_cuda(old))
    text = once(text, 'name=f"okc_moe_kperm3_v9_{tag}_sm{arch}",', 'name=f"okc_moe_wgs_shared_a_v15_{tag}_sm{arch}",')
    text = once(text, "            verbose=False,", "            verbose=\"-DOKC_WGS=1\" in defs,  # preserve WGS compiler diagnostics")
    text = once(text, "# Isolated KPERM3 adaptation over frozen ef7b210b; no SC6/MFIN or pipeline changes.", "# Isolated shared-A WGS experiment over frozen scored 247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0.\n# Changes: two joined raw/X slots, two exact BF16 shared-A banks, SS WGMMA and publication/reuse barriers.\n# Parent KPERM3 provenance and all upstream notices are preserved below.")
    ast.parse(text)
    return text


if __name__ == "__main__":
    text = build(); CANDIDATE.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), CANDIDATE)
