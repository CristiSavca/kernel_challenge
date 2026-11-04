"""Host-only diagnostic fragment; combined candidate has a separate owner.

Importing this file does not build, import or execute a submission.
"""

MOE_HOST = r'''
// RESOURCE_DIAG_MOE_HOST_BEGIN: host-only; no device functions or launches change.
#include <cstdio>
#include <mutex>
#include <atomic>
namespace moe_resource_diag {
static std::mutex mu;
static std::atomic<bool> enabled{false}, exhausted{false};
static long long generation = 0;
static size_t limit = 0, used = 0;
static int count = 0;
struct Key { int device, big, threads; size_t dynamic; };
static Key seen[1] = {};
constexpr size_t MAX_BUDGET = 320, RESERVE = 48;
static void stop() {
  if (!exhausted) {
    constexpr char line[] = "MOER_STOP coverage=partial\n";
    constexpr size_t n = sizeof(line) - 1;
    if (used + n <= limit) { std::fwrite(line, 1, n, stderr); used += n; }
    exhausted = true;
  }
}
template<class Kernel>
static void report(Kernel kern, int device, int M, int big, long long grid,
                   int pdl, int threads, size_t smem) {
  std::lock_guard<std::mutex> guard(mu);
  if (!enabled || exhausted) return;
  for (int i = 0; i < count; ++i)
    if (seen[i].device == device && seen[i].big == big &&
        seen[i].threads == threads && seen[i].dynamic == smem) return;
  // The 320-byte module allocation supports one schema + one resource sample.
  // A different BIG/device/shared key is explicitly incomplete, before querying.
  if (count == 1 || used + RESERVE >= limit) { stop(); return; }
#if OKC_TINY && OKC_WS
  cudaFuncAttributes fa{};
  int actual_dev = -1, sms = -1, blocks = -1;
  int ed = (int)cudaGetDevice(&actual_dev);
  int es = ed == 0 ? (int)cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, actual_dev) : -1;
  int ef = (int)cudaFuncGetAttributes(&fa, kern);
  int eo = ef == 0 ? (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, threads, smem) : -1;
  if (ed != 0) actual_dev = -1;
  if (es != 0) sms = -1;
  if (eo != 0) blocks = -1;
  char line[256];
  int n = std::snprintf(line, sizeof(line),
      "MOER g=%lld f=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d x=%d,%d,%lld,%d,%d,%zu a=%d,%lld,%lld,%d,%d,%d,%d d=%d,%d e=%d,%d,%d,%d\n",
      generation, OKC_NW, OKC_NS, OKC_WS, OKC_TINY, OKC_SPLIT, OKC_MULDEC, OKC_DQ7,
      OKC_PRMTSC, OKC_FINPF, OKC_WSBLK, OKC_SKSC2, OKC_FTSC2, OKC_L2H, OKC_CPF,
      M, big, grid, pdl, threads, smem,
      ef == 0 ? fa.numRegs : -1, ef == 0 ? (long long)fa.localSizeBytes : -1LL,
      ef == 0 ? (long long)fa.sharedSizeBytes : -1LL,
      ef == 0 ? fa.maxDynamicSharedSizeBytes : -1, ef == 0 ? fa.maxThreadsPerBlock : -1,
      blocks, sms, device, actual_dev, ed, es, ef, eo);
  constexpr char schema[] = "MOER_SCHEMA f=NW,NS,WS,TY,SP,MUL,DQ,PR,FP,BK,SC,FC,L2,CP x=M,B,G,P,th,dyn a=R,L,S,MD,MT,O,SM d=tensor,current e=D,S,A,O g=last_arm\n";
  const size_t header = count == 0 ? sizeof(schema)-1 : 0;
  if (n <= 0 || (size_t)n >= sizeof(line) || used + header + (size_t)n + RESERVE > limit) { stop(); return; }
  if (header) { std::fwrite(schema, 1, header, stderr); used += header; }
  seen[count++] = Key{device, big, threads, smem};
  std::fwrite(line, 1, (size_t)n, stderr); used += (size_t)n;
#endif
}
}
void okc_set_diag(int64_t on, int64_t generation, int64_t total_byte_budget) {
  std::lock_guard<std::mutex> guard(moe_resource_diag::mu);
  moe_resource_diag::enabled = false;
  // A module's allocation is immutable after assignment. Never reset bytes or keys.
  if (moe_resource_diag::limit == 0 && total_byte_budget > 0)
    moe_resource_diag::limit = (size_t)(total_byte_budget > 320 ? 320 : total_byte_budget);
  moe_resource_diag::generation = (long long)generation;
  moe_resource_diag::enabled = on != 0 && moe_resource_diag::limit != 0;
}
// RESOURCE_DIAG_MOE_HOST_END
'''

MOE_CALL = r'''
    // RESOURCE_DIAG_MOE_CALL_BEGIN
#if OKC_TINY && OKC_WS
    if (moe_resource_diag::enabled && !moe_resource_diag::exhausted) {
      const bool diag_big = p.M >= OKC_BIGM;
      auto diag_kernel = diag_big ? okc::moe_fused<true> : okc::moe_fused<false>;
      moe_resource_diag::report(diag_kernel, x.get_device(), p.M, (int)diag_big,
                                (long long)grid, g_okc_pdl, okc::NTL, sm);
    }
#endif
    // RESOURCE_DIAG_MOE_CALL_END
'''

PYTHON_LIFECYCLE = r'''
# RESOURCE_DIAG_LIFECYCLE_BEGIN: host metadata only; never build or select a backend.
import sys as _resource_diag_sys
_RESOURCE_DIAG_ARMED = False
_RESOURCE_DIAG_GENERATION = 0
_RESOURCE_DIAG_MOE_MODULES = {}
_RESOURCE_DIAG_MOE_ALLOCATED = 0
_RESOURCE_DIAG_MOE_OVERFLOW = False
_RESOURCE_DIAG_MOE_OVERFLOW_PRINTED = False


def _resource_diag_allowed_moe_keys():
    allowed = set()
    for base in (MOE_TINY_DEFINES, MOE_TINY2_DEFINES, MOE_MID_DEFINES):
        for suffix in ((), tuple(_MA_SFX_IF_PRMT)):
            key = tuple(base) + suffix
            if DQ7 and "-DOKC_DQ7=1" not in key:
                key += ("-DOKC_DQ7=1",)
            allowed.add(key)
    return allowed


def _resource_diag_overflow_notice():
    global _RESOURCE_DIAG_MOE_OVERFLOW_PRINTED
    if _RESOURCE_DIAG_ARMED and _RESOURCE_DIAG_MOE_OVERFLOW and not _RESOURCE_DIAG_MOE_OVERFLOW_PRINTED:
        # At most35 additional bytes, covered by the128B global reserve.
        print("MOER_STOP module_budget_or_flags", file=_resource_diag_sys.stderr)
        _RESOURCE_DIAG_MOE_OVERFLOW_PRINTED = True


def _resource_diag_bind_moe(ext, defines):
    global _RESOURCE_DIAG_MOE_ALLOCATED, _RESOURCE_DIAG_MOE_OVERFLOW
    key = tuple(defines)
    if ext in _RESOURCE_DIAG_MOE_MODULES:
        budget = _RESOURCE_DIAG_MOE_MODULES[ext][1]
    else:
        eligible = key in _resource_diag_allowed_moe_keys()
        budget = 320 if eligible and _RESOURCE_DIAG_MOE_ALLOCATED + 320 <= 1920 else 0
        if (eligible and not budget) or (not eligible and "-DOKC_TINY=1" in key and "-DOKC_WS=1" in key):
            _RESOURCE_DIAG_MOE_OVERFLOW = True
        _RESOURCE_DIAG_MOE_ALLOCATED += budget
        if budget:
            _RESOURCE_DIAG_MOE_MODULES[ext] = (key, budget)
    ext.okc_set_diag(int(_RESOURCE_DIAG_ARMED), _RESOURCE_DIAG_GENERATION,
                     budget)
    _resource_diag_overflow_notice()


def _resource_diag_bind_sk2(ext):
    ext.sk2_set_diag(int(_RESOURCE_DIAG_ARMED), _RESOURCE_DIAG_GENERATION)


def _resource_diag_begin():
    global _RESOURCE_DIAG_ARMED, _RESOURCE_DIAG_GENERATION
    _RESOURCE_DIAG_ARMED = False
    _RESOURCE_DIAG_GENERATION += 1
    if bfp__ext is not None:
        _resource_diag_bind_sk2(bfp__ext)
    for defines, ext in tuple(_MA_EXTS.items()):
        _resource_diag_bind_moe(ext, defines)


def _resource_diag_end():
    global _RESOURCE_DIAG_ARMED
    _RESOURCE_DIAG_ARMED = True
    if bfp__ext is not None:
        _resource_diag_bind_sk2(bfp__ext)
    for defines, ext in tuple(_MA_EXTS.items()):
        _resource_diag_bind_moe(ext, defines)
    _resource_diag_overflow_notice()
# RESOURCE_DIAG_LIFECYCLE_END
'''

_ATTR_ANCHOR = '''    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which'''

MOE_REPLACEMENTS = [
    ('static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)',
     MOE_HOST+'\nstatic int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)'),
    (_ATTR_ANCHOR, _ATTR_ANCHOR.replace('  cudaStream_t st',MOE_CALL+'\n  cudaStream_t st')),
    ('void okc_set_pdl(int64_t on);', 'void okc_set_pdl(int64_t on);\nvoid okc_set_diag(int64_t on, int64_t generation, int64_t total_byte_budget);'),
    ('functions=["okc_run", "okc_set_pdl"],', 'functions=["okc_run", "okc_set_pdl", "okc_set_diag"],'),
    ('name=f"dl_okc_moe_kperm3_v12_{tag}_sm{arch}"', 'name=f"dl_okc_moe_selected_diag_v13_{tag}_sm{arch}"'),
    ('        _MA_EXTS[defines].okc_set_pdl(1 if _MA_PDL else 0)\n    return _MA_EXTS[defines]',
     '        _MA_EXTS[defines].okc_set_pdl(1 if _MA_PDL else 0)\n        _resource_diag_bind_moe(_MA_EXTS[defines], defines)\n    return _MA_EXTS[defines]'),
]


def apply_moe_fragment(source):
    for old,new in MOE_REPLACEMENTS:
        assert source.count(old)==1,(old[:80],source.count(old))
        source=source.replace(old,new,1)
    return source


def reverse_moe_fragment(source):
    for old,new in reversed(MOE_REPLACEMENTS):
        assert source.count(new)==1,(new[:80],source.count(new))
        source=source.replace(new,old,1)
    return source
