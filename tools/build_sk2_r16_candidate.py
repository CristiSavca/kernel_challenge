#!/usr/bin/env python3
"""Build one fixed R16 SK2 family from exact scored97af; no candidate execution."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
PARENT = ROOT / "experiments/candidates/muse_decode_layer" / (PARENT_SHA + ".py")
CANDIDATE = ROOT / "muse_decode_layer/candidates/native_sk2_r16_v14.py"
REPLACEMENTS = []

HOST = r'''
// R16_ADMISSION_BEGIN: preparation only; exact pointers and actual packed pmax.
#include <cstdio>
#include <mutex>
namespace sk2_r16_admission {
static std::mutex mu;
static size_t bytes = 0;
static bool stopped = false;
constexpr size_t LIMIT = 8192, RESERVE = 96;
static void emit(const char* key, const cudaFuncAttributes& fa, int blocks, int minimum,
                 int ed, int ep, int ea, int es, int eb, int eo) {
  std::lock_guard<std::mutex> lock(mu);
  if (stopped) return;
  char row[640];
  int n = std::snprintf(row, sizeof(row),
    "R16R %s a=%d,%lld,%lld,%d,%d occ=%d min=%d err=%d,%d,%d,%d,%d,%d\n",
    key, eb == 0 ? fa.numRegs : -1, eb == 0 ? (long long)fa.localSizeBytes : -1LL,
    eb == 0 ? (long long)fa.sharedSizeBytes : -1LL, eb == 0 ? fa.maxDynamicSharedSizeBytes : -1,
    eb == 0 ? fa.maxThreadsPerBlock : -1, eo == 0 ? blocks : -1, minimum, ed, ep, ea, es, eb, eo);
  constexpr char schema[] = "R16R_SCHEMA phase=prepare_preflight t=MT,TOKR,NT,NWR,WSK,UPW,NS,COMP,HL,NACC,CL,OutBytes,PK,FUSE,BND x=M,N,K,pmax,threads,grid,dyn a=regs,local,static,maxdyn,maxthreads occ=blocks_per_SM err=device,optin,attrs_before,set,attrs_after,occ\n";
  const size_t header = bytes == 0 ? sizeof(schema) - 1 : 0;
  if (n <= 0 || (size_t)n >= sizeof(row) || bytes + header + (size_t)n + RESERVE > LIMIT) {
    constexpr char stop[] = "R16R_STOP coverage=incomplete reason=log_capacity\n";
    if (bytes + sizeof(stop) - 1 <= LIMIT) { std::fwrite(stop, 1, sizeof(stop) - 1, stderr); bytes += sizeof(stop) - 1; }
    stopped = true;
    return;
  }
  if (header) { std::fwrite(schema, 1, header, stderr); bytes += header; }
  std::fwrite(row, 1, (size_t)n, stderr); bytes += (size_t)n;
}
template<class Kernel>
static void check(Kernel kern, const char* key, int threads, size_t smem, int minimum) {
  int dev = -1, optin = -1, blocks = -1;
  cudaFuncAttributes before{}, after{};
  int ed = (int)cudaGetDevice(&dev);
  int ep = ed == 0 ? (int)cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) : -1;
  int ea = (int)cudaFuncGetAttributes(&before, kern);
  // Identical opt-in value to the original launch setup; no register cap or new launch geometry.
  int es = ep == 0 && ea == 0 ? (int)cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                        optin - (int)before.sharedSizeBytes) : -1;
  int eb = (int)cudaFuncGetAttributes(&after, kern);
  int eo = es == 0 && eb == 0 ? (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, threads, smem) : -1;
  emit(key, after, blocks, minimum, ed, ep, ea, es, eb, eo);
  // Logging can stop, but checking never stops: no resource failure becomes a tuner fallback.
  TORCH_CHECK(ed == 0 && ep == 0 && ea == 0 && es == 0 && eb == 0 && eo == 0,
              "R16 resource API failure: ", key);
  TORCH_CHECK(after.localSizeBytes == 0 && after.maxThreadsPerBlock >= threads &&
              after.maxDynamicSharedSizeBytes >= (int)smem && blocks >= minimum,
              "R16 resource admission failure: ", key, " blocks=", blocks, " minimum=", minimum,
              " local=", after.localSizeBytes);
}
}
// R16_ADMISSION_END
'''

PREFLIGHT_BRANCH = r'''
    if (r16_minimum > 0) {
      char key[320];
      int kn = std::snprintf(key, sizeof(key),
        "t=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d x=%d,%d,%d,%lld,%d,%d,%llu",
        MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, (int)COMP, (int)HL, NACC, CL, (int)sizeof(OutT),
        (int)PK, (int)FUSE, (int)BND, M, N, K, (long long)pmax, (NW + 1 + NESC) * 32, N / R * CL,
        (unsigned long long)smem);
      TORCH_CHECK(kn > 0 && (size_t)kn < sizeof(key), "R16 resource key overflow");
      sk2_r16_admission::check(kern, key, (NW + 1 + NESC) * 32, smem, r16_minimum);
      return;
    }
'''

GENERATE_PREFLIGHT = '''    # Query the actual generated pointers, including compatibility paths, before any self-test/tuner launch.
    plines = []
    changed = {"pooler", "pooler32_r144", "qkv8_r176", "qkv16_r176", "qkv32_r208"}
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if name not in changed:
            continue
        calls = []
        for fused, candidates in ((False, lines), (True, flines if name == "pooler" else [])):
            for line in candidates:
                if not fused and f"cfg == {ci} &&" not in line:
                    continue
                tokr = int(_re.search(r"tokr == (\\d+)", line).group(1))
                template = _re.search(r"launch_sk2<([^>]+)>", line).group(1)
                minimum = 1 if (name == "pooler" and (not fused or tokr == 32)) else 2
                for m in ((1,) if tokr == 8 else (1, tokr)):
                    calls.append(f"launch_sk2<{template}>(z, z, z, z, pmax, 0, {m}, {n_rows}, {bfp_SK2_K[layer]}, "
                                 f"0, nullptr, nullptr, nullptr, nullptr, nullptr, Sk2Bnd{{}}, nullptr, TArgs{{}}, {minimum});")
        plines.append(f"if (cfg == {ci}) {{ " + " ".join(calls) + f" return {len(calls)}; }}")
'''

PYTHON_PREFLIGHT = '''
def bfp_sk2_r16_preflight(state):
    """All14 changed resource records (12 pointers), outside self-test/tuner exception handlers."""
    if not state.get("sm90"):
        raise RuntimeError("R16 fixed candidate requires sm90 SK2")
    ext = bfp__get_ext()
    count = 0
    for name, expected in (("pooler", 7), ("pooler32_r144", 2), ("qkv16_r176", 2),
                           ("qkv32_r208", 2), ("qkv8_r176", 1)):
        if name not in state.get("sk2", {}):
            raise RuntimeError("R16 missing compressed configuration: " + name)
        ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
        if int(pmax) <= 0 or int(pmax) % 2 or int(tab.shape[1]) != int(pmax):
            raise RuntimeError("R16 invalid prepared escape table: " + name)
        actual = int(ext.sk2_r16_preflight(ci, pmax))
        if actual != expected:
            raise RuntimeError("R16 incomplete resource preflight: " + name)
        count += actual
    if count != 14:
        raise RuntimeError("R16 incomplete resource preflight")
    state["r16_preflight_complete"] = True

'''


def replace(source, old, new, count=1):
    assert source.count(old) == count, (old[:120], source.count(old), count)
    REPLACEMENTS.append((old, new, count))
    return source.replace(old, new)


def build(write=True):
    REPLACEMENTS.clear()
    source = PARENT.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    source = replace(source, "# okc: problem=muse_decode_layer\n", "# okc: problem=muse_decode_layer\n# Fixed SK2 R16 family over exact97af: lossless packing, uniform eight-piece fused router, resource admission.\n# The following provenance describes the inherited parent; this experiment changes only the fixed SK2 family.\n")
    for old, new in (
        ("('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1)", "('pooler', 'pooler', 4096, 16, 2, 1, 8, 1, 2, 1)"),
        ("('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1)", "('pooler32_r144', 'pooler', 4096, 16, 2, 1, 4, 2, 1, 1)"),
        *((f"('{name}', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1)", f"('{name}', 'qkv_gate', 2304, 16, 2, 1, 4, 2, 1, 1)")
          for name in ("qkv16_r176", "qkv32_r208", "qkv8_r176")),
    ):
        source = replace(source, old, new)
    source = replace(source, "    hl = layer == \"router\"\n    tu = 2048 if hl else 768", '''    hl = layer == "router"
    # Fixed row-granularity hypothesis: no parameter search or suffix cap for these configurations.
    fixed = {"pooler": {8: 4, 16: 3, 32: 2}, "pooler32_r144": {32: 2},
             "qkv8_r176": {8: 5}, "qkv16_r176": {16: 3}, "qkv32_r208": {32: 2}}
    if name in fixed and tokr in fixed[name]:
        # Fused pooler ring reserves router payload even though main weight payload is smaller.
        wby = NWR * WSK * UPW * NT * 768
        wbyr = NWR * WSK * 2048 if name == "pooler" else 0
        sst = (WSK * UPW * tokr * 128 + max(wby, wbyr) + 1023) // 1024 * 1024
        assert sst > 0 and R == 16
        return hl, fixed[name][tokr]
    tu = 2048 if hl else 768''')
    # Both device and host geometry use the same payload maximum. Router WBYR declaration moves earlier.
    source = replace(source, "  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;", "  constexpr int WBYR = NW * 2048;\n  constexpr int SST = ((XBY + (FUSE && WBYR > WBY ? WBYR : WBY)) + 1023) & ~1023;", 2)
    source = replace(source, '''  // (tile rt = cta / 4) over one quarter of K (rq = cta % 4), from the router's own stage-major hi|lo stream
  // ([16 stages][32 tiles][8 warps][2048 B]); 4 more stages through the same ring
  constexpr int NSTR = FUSE ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;''', '''  // (tile rt = cta / 8) over one eighth of K (rq = cta % 8), from the router's own stage-major hi|lo stream
  // ([16 stages][32 tiles][8 warps][2048 B]); 2 more stages through the same ring
  constexpr int NSTR = FUSE ? 2 : 0;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && NWR == 1 && NT == 2), "sk2 fused: R16 eight-piece config only");
  const int rt = cta >> 3, rq = cta & 7;''')
    source = replace(source, '''    // this CTA's quarter-K partial -> part[rt][rq][TOKS][8]; the last of the 4 quarters (release/acquire counter)
    // adds them in quarter order (deterministic) and writes the router output
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;''', '''    // This CTA's eighth-K partial -> part[rt][rq][32][8]; last of8 release/acquire arrivals
    // adds all pieces in fixed ascending order and writes the router output.
    float* pp = part + ((size_t)rt * 8 + rq) * 32 * 8;''')
    source = replace(source, "s_rlast = (old & 3) == 3;", "s_rlast = (old & 7) == 7;")
    source = replace(source, "// router piece: global router stage rq * 4 + j of tile rt", "// router piece: global router stage rq * 2 + j of tile rt")
    source = replace(source, "const float* pr = part + (size_t)rt * 4 * 32 * 8;", "const float* pr = part + (size_t)rt * 8 * 32 * 8;")
    for index in ("iu", "iv", "i"):
        old = " + ".join(f"__ldcg(pr + {str(256*j) + ' + ' if j else ''}{index})" for j in range(4))
        new = " + ".join(f"__ldcg(pr + {str(256*j) + ' + ' if j else ''}{index})" for j in range(8))
        source = replace(source, old, new)
    source = replace(source, "torch.empty(32 * 4 * 32 * 8, dtype=torch.float32", "torch.empty(32 * 8 * 32 * 8, dtype=torch.float32", 2)
    # Host-only admission uses exact launch template dispatch before any device input access.
    anchor = "template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,\n          bool FUSE = false, bool BND = false>\nstatic void launch_sk2"
    source = replace(source, anchor, HOST + "\n" + anchor)
    source = replace(source, "Sk2Bnd bnd = Sk2Bnd{}, int64_t* cap = nullptr, TArgs ta = TArgs{}) {", "Sk2Bnd bnd = Sk2Bnd{}, int64_t* cap = nullptr, TArgs ta = TArgs{}, int r16_minimum = 0) {")
    kernline = "    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE, BND>;\n"
    source = replace(source, kernline, kernline + PREFLIGHT_BRANCH)
    source = replace(source, "int64_t sk2b(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,", '''int64_t sk2_r16_preflight(int64_t cfg, int64_t pmax) {
  at::Tensor z;  // preflight returns before any tensor pointer or descriptor access
  @@R16_PREFLIGHT@@
  TORCH_CHECK(false, "R16 preflight: no changed configuration");
  return 0;
}

int64_t sk2b(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,''')
    source = replace(source, "void sk2_set_pdl(int64_t n);", "void sk2_set_pdl(int64_t n);\nint64_t sk2_r16_preflight(int64_t cfg, int64_t pmax);")
    source = replace(source, "'void sk2_set_pdl(int64_t) {}\\n'", "'void sk2_set_pdl(int64_t) {}\\n'\n                'int64_t sk2_r16_preflight(int64_t, int64_t) { TORCH_CHECK(false, \"R16 needs sm90\"); return 0; }\\n'")
    source = replace(source, '    return (bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\\n  ".join(lines))', GENERATE_PREFLIGHT + '    return (bfp_SK2_SRC.replace("@@R16_PREFLIGHT@@", "\\n  ".join(plines))\n            .replace("@@SK2_DISPATCH@@", "\\n  ".join(lines))')
    source = replace(source, 'name=f"dl_okc_bf16proj_v86y4b1_{arch}"', 'name=f"dl_okc_bf16proj_r16_v14_{arch}"')
    source = replace(source, '"sk2b", "sk2t", "sk2ft"],', '"sk2b", "sk2t", "sk2ft", "sk2_r16_preflight"],')
    source = replace(source, "def bfp_prepare(weights):", PYTHON_PREFLIGHT + "def bfp_prepare(weights):")
    source = replace(source, "    bfp_sk2_prepare(weights, state)\n", "    bfp_sk2_prepare(weights, state)\n    bfp_sk2_r16_preflight(state)\n")
    source = replace(source, "    _bfp_prepare(weights, st)\n", '''    _bfp_prepare(weights, st)
    # _bfp_prepare catches backend failures; this fixed experiment must never hide a failed preflight.
    if not st.get("bfp") or not st["bfp"].get("r16_preflight_complete"):
        raise RuntimeError("R16 resource preflight failed; fixed candidate cannot use fallback")
''')
    compile(source, str(CANDIDATE), "exec")
    if write:
        CANDIDATE.write_text(source)
    return source


if __name__ == "__main__":
    source = build()
    print(CANDIDATE.relative_to(ROOT), hashlib.sha256(source.encode()).hexdigest(), len(REPLACEMENTS), "patches")
