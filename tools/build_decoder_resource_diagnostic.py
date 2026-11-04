#!/usr/bin/env python3
"""Construct the host-only diagnostic from frozen 97af; never execute candidate."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
PARENT = ROOT / "experiments/candidates/muse_decode_layer" / (PARENT_SHA + ".py")
CANDIDATE = ROOT / "muse_decode_layer/candidates/native_selected_resources_diag_v13.py"
OWN_REPLACEMENTS = []

SK2_HOST = r'''
// RESOURCE_DIAG_SK2_HOST_BEGIN: host-only, no device source or launch change.
#include <cstdio>
#include <cstring>
#include <mutex>
#include <atomic>
namespace sk2_resource_diag {
static std::mutex mu;
static std::atomic<bool> enabled{false}, exhausted{false};
static long long generation = 0;
static size_t used = 0;
static int count = 0;
static char seen[64][256] = {};
constexpr size_t LIMIT = 8192;
constexpr size_t RESERVE = 96;
static void stop(const char* reason) {
  if (!exhausted) {
    char line[RESERVE];
    int n = std::snprintf(line, sizeof(line), "SK2R_STOP reason=%s bytes=%zu records=%d\n", reason, used, count);
    if (n > 0 && (size_t)n < sizeof(line) && used + (size_t)n <= LIMIT) {
      std::fwrite(line, 1, (size_t)n, stderr); used += (size_t)n;
    }
    exhausted = true;
  }
}
template<class Kernel>
static void report(Kernel kern, int device, const char* key, int threads, size_t smem) {
  std::lock_guard<std::mutex> guard(mu);
  if (!enabled || exhausted) return;
  for (int i = 0; i < count; ++i) if (!std::strcmp(seen[i], key)) return;
  if (std::strlen(key) >= sizeof(seen[0]) || count == 64) { stop("key_capacity"); return; }
  // Reserve worst-case record space before any CUDA query; no queries after budget exhaustion.
  constexpr size_t MAX_ROW = 512;
  if (used + MAX_ROW + RESERVE > LIMIT) { stop("byte_budget"); return; }
  cudaFuncAttributes fa{};
  int actual_dev = -1, sms = -1, blocks = -1;
  int ed = (int)cudaGetDevice(&actual_dev);
  int es = ed == 0 ? (int)cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, actual_dev) : -1;
  int ef = (int)cudaFuncGetAttributes(&fa, kern);
  int eo = ef == 0 ? (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, threads, smem) : -1;
  if (ed != 0) actual_dev = -1;
  if (es != 0) sms = -1;
  if (eo != 0) blocks = -1;
  char line[MAX_ROW];
  int n = std::snprintf(line, sizeof(line),
      "SK2R g=%lld %s a=%d,%lld,%lld,%d,%d occ=%d,%d dev=%d,%d err=%d,%d,%d,%d\n",
      generation, key, ef == 0 ? fa.numRegs : -1,
      ef == 0 ? (long long)fa.localSizeBytes : -1LL,
      ef == 0 ? (long long)fa.sharedSizeBytes : -1LL,
      ef == 0 ? fa.maxDynamicSharedSizeBytes : -1,
      ef == 0 ? fa.maxThreadsPerBlock : -1, blocks, sms, device, actual_dev, ed, es, ef, eo);
  if (n <= 0 || (size_t)n >= sizeof(line)) { stop("row_format"); return; }
  if (count == 0) {
    constexpr char schema[] = "SK2R_SCHEMA t=MT,TOKR,NT,NWR,WSK,UPW,NS,COMP,HL,NACC,CL,OutBytes,PK,FUSE,BND x=M,N,K,pmax,UV,PDL,threads,grid,dyn a=regs,local,static,maxdyn,maxthreads occ=blocks_per_SM,SMs dev=tensor,current err=device,SMs,attrs,occ CLgt1_block_only\n";
    constexpr size_t len = sizeof(schema) - 1;
    if (used + len + (size_t)n + RESERVE > LIMIT) { stop("schema_budget"); return; }
    std::fwrite(schema, 1, len, stderr); used += len;
  }
  std::strcpy(seen[count++], key);
  std::fwrite(line, 1, (size_t)n, stderr); used += (size_t)n;
}
}
void sk2_set_diag(int64_t on, int64_t generation) {
  std::lock_guard<std::mutex> guard(sk2_resource_diag::mu);
  sk2_resource_diag::enabled = on != 0;
  sk2_resource_diag::generation = (long long)generation;
}
// RESOURCE_DIAG_SK2_HOST_END
'''

SK2_CALL = r'''
    // RESOURCE_DIAG_SK2_CALL_BEGIN
    if (sk2_resource_diag::enabled && !sk2_resource_diag::exhausted) {
      char key[256];
      int kn = std::snprintf(key, sizeof(key),
        "t=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d x=%d,%d,%d,%lld,%d,%d,%d,%d,%llu d=%d",
        MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, (int)COMP, (int)HL, NACC, CL, (int)sizeof(OutT),
        (int)PK, (int)FUSE, (int)BND, M, N, K, (long long)pmax, ta.mode, pdl,
        (NW + 1 + NESC) * 32, N / R * CL, (unsigned long long)smem, x.get_device());
      if (kn > 0 && (size_t)kn < sizeof(key))
        sk2_resource_diag::report(kern, x.get_device(), key, (NW + 1 + NESC) * 32, smem);
      else {
        std::lock_guard<std::mutex> guard(sk2_resource_diag::mu);
        sk2_resource_diag::stop("key_format");
      }
    }
    // RESOURCE_DIAG_SK2_CALL_END
'''


def replace_once(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    OWN_REPLACEMENTS.append((old, new))
    return source.replace(old, new, 1)


def build(write=True):
    OWN_REPLACEMENTS.clear()
    source = PARENT.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    source = replace_once(source, "# okc: problem=muse_decode_layer\n",
        "# okc: problem=muse_decode_layer\n"
        "# Diagnostic only: selected host CUDA resources over exact scored97af; do not promote by score.\n"
        "# All device functions, packing, tuning, build flags and launches retain parent source.\n"
        "# The following KPERM provenance describes the unchanged parent computation.\n")
    anchor = "template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,\n          bool FUSE = false, bool BND = false>\nstatic void launch_sk2"
    source = replace_once(source, anchor, SK2_HOST + "\n" + anchor)
    source = replace_once(source, "    const int pdl = g_sk2_pdl;\n", "    const int pdl = g_sk2_pdl;\n" + SK2_CALL)
    source = replace_once(source, "void sk2_set_pdl(int64_t n);", "void sk2_set_pdl(int64_t n);\nvoid sk2_set_diag(int64_t on, int64_t generation);")
    source = replace_once(source, "'void sk2_set_pdl(int64_t) {}\\n'", "'void sk2_set_pdl(int64_t) {}\\n'\n                'void sk2_set_diag(int64_t, int64_t) {}\\n'")
    source = replace_once(source, '"sk2_set_pdl", "sk2b", "sk2t", "sk2ft"],', '"sk2_set_pdl", "sk2b", "sk2t", "sk2ft", "sk2_set_diag"],')
    source = replace_once(source, 'name=f"dl_okc_bf16proj_v86y4b1_{arch}"', 'name=f"dl_okc_bf16proj_selected_diag_v13_{arch}"')
    source = replace_once(source, '    return bfp__ext\n',
        '        _resource_diag_bind_sk2(bfp__ext)\n    return bfp__ext\n')
    # A separate contributor owns this host-only fragment; candidate remains single-writer.
    from decoder_moe_resource_diag_fragment import apply_moe_fragment, PYTHON_LIFECYCLE
    source = apply_moe_fragment(source)
    source = replace_once(source, "def prepare(weights):\n", PYTHON_LIFECYCLE + "\n\ndef prepare(weights):\n    _resource_diag_begin()\n")
    source = replace_once(source, "    _check_guards()            # after the tuner (every moe build) and the attention self-test\n    return st",
        "    _check_guards()            # after the tuner (every moe build) and the attention self-test\n    _resource_diag_end()\n    return st")
    compile(source, str(CANDIDATE), "exec")
    if write:
        CANDIDATE.write_text(source)
        print(hashlib.sha256(source.encode()).hexdigest(), CANDIDATE.relative_to(ROOT))
    return source


if __name__ == "__main__":
    build()
