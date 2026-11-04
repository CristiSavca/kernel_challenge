"""Bounded eligibility diagnostics over frozen V29, with no selection/math change."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SHA = "bbbc37f80aa3a5263dd0a66611bae62b2376c8b77ed67c14ba8ffd8db78e8a2c"

PY_HELPER = '''# V31: prepare-only eligibility records; no input tensors or tuning decisions changed.
_v31_diag_bytes = {"self": 0, "inner": 0, "outer": 0}
_v31_diag_stopped = set()


def _v31_diag_emit(kind, **fields):
    import json
    group = "outer" if kind == "outer" else "inner" if kind == "choice" else "self"
    budget = {"self": 4096, "inner": 2048, "outer": 3072}[group]
    if group in _v31_diag_stopped:
        return
    def clean(value):
        if isinstance(value, float) and not -float("inf") < value < float("inf"):
            return None
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value
    row = "V31P " + json.dumps(clean(dict(kind=kind, **fields)), separators=(",", ":"), allow_nan=False)
    n = len(row.encode()) + 1
    if _v31_diag_bytes[group] + n > budget - 80:
        print("V31P_STOP group=" + group + " coverage=incomplete reason=log_capacity", flush=True)
        _v31_diag_stopped.add(group)
        return
    print(row, flush=True)
    _v31_diag_bytes[group] += n


'''

HOST_HELPER = r'''
// V31: attributes of actual compiled pointers; observations only, no admission changes.
#include <cstdio>
#include <mutex>
#include <atomic>
static std::mutex v31_resource_mutex;
static size_t v31_resource_bytes = 0;
static bool v31_resource_stopped = false;
static void v31_resource(const char* identity, const void* fn, int nt, size_t sm) {
  std::lock_guard<std::mutex> lock(v31_resource_mutex);
  if (v31_resource_stopped) return;
  if (v31_resource_bytes + 512 > 4096 - 80) {
    fprintf(stderr, "V31R_STOP coverage=incomplete reason=log_capacity\n");
    v31_resource_stopped = true;
    return;
  }
  cudaFuncAttributes a{};
  const cudaError_t attr = cudaFuncGetAttributes(&a, fn);
  int occ = -1;
  const cudaError_t cap = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, nt, sm);
  char row[512];
  const int n = snprintf(row, sizeof(row),
      "V31R %s nt=%d dyn=%zu attr=%d,%zu,%zu,%d occ=%d api=%d,%d\n",
      identity, nt, sm, a.numRegs, a.localSizeBytes, a.sharedSizeBytes,
      a.maxDynamicSharedSizeBytes, occ, (int)attr, (int)cap);
  if (n < 0 || (size_t)n >= sizeof(row)) {
    fprintf(stderr, "V31R_STOP coverage=incomplete reason=row_capacity\n");
    v31_resource_stopped = true;
    return;
  }
  fputs(row, stderr);
  v31_resource_bytes += n;
}
'''

HOST_CALL = '''    // First normal launch of this exact template/PK pointer; cap-only calls do not consume the record.
    static std::atomic<bool> v31_reported{false};
    if (!v31_reported.exchange(true, std::memory_order_relaxed)) {
      char identity[224];
      snprintf(identity, sizeof(identity),
          "t=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d x=%d,%d,%d,%lld",
          MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, (int)COMP, (int)HL, NACC, CL,
          (int)PK, (int)FUSE, (int)BND, (int)sizeof(OutT), M, N, K, (long long)pmax);
      v31_resource(identity, (const void*)kern, (NW + 1 + NESC) * 32, smem);
    }
'''


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def build():
    parent = (HERE / "parent.py").read_text()
    assert sha(parent) == PARENT_SHA
    s, edits = parent, []

    def edit(old, new):
        nonlocal s
        assert s.count(old) == 1, (old[:90], s.count(old))
        s = s.replace(old, new)
        edits.append({"old": old, "new": new})

    edit('bfp_SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)',
         PY_HELPER + 'bfp_SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)')
    edit('''                try:
                    out = bfp__run(ch, x, layer, m, nn, k, state).double()''',
         '''                v31_exception = None
                try:
                    out = bfp__run(ch, x, layer, m, nn, k, state).double()''')
    edit('''                except RuntimeError as e:  # launch failure -> fall back
                    err = float("inf")''',
         '''                except RuntimeError as e:  # launch failure -> fall back
                    v31_exception = type(e).__name__ + ": " + str(e)[:240]
                    err = float("inf")''')
    anchor = '                    if False: print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)'
    edit(anchor, anchor + '''
                if ch[0] == "sk2":
                    _v31_diag_emit("selftest", layer=layer, m=m, key=key,
                                   error=err if -float("inf") < err < float("inf") else None,
                                   tolerance=tol, passed=bool(err < tol), exception=v31_exception)
    _v31_diag_emit("disabled", keys=sorted(k for k in state["disabled"] if len(k) > 1 and k[1] == "sk2"))''')
    anchor = '        pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")'
    edit(anchor, anchor + '''
        if m <= 32:
            _v31_diag_emit("choice", layer=layer, m=m, candidates=tl, pick=pick)''')
    anchor = '            xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)'
    edit(anchor, '            v31_outer_errors = {}\n' + anchor)
    edit('''                except Exception:  # noqa: BLE001
                    t_bfp = 1e30''', '''                except Exception as v31_error:  # noqa: BLE001
                    v31_outer_errors["bfp"] = type(v31_error).__name__ + ": " + str(v31_error)[:120]
                    t_bfp = 1e30''')
    edit('''                except Exception:  # noqa: BLE001  (sm_80 or an older bf16_proj file: not offered)
                    t_bfpf = 1e30''', '''                except Exception as v31_error:  # noqa: BLE001  (sm_80 or an older bf16_proj file: not offered)
                    v31_outer_errors["bfpf"] = type(v31_error).__name__ + ": " + str(v31_error)[:120]
                    t_bfpf = 1e30''')
    anchor = '            st["mm_pick"][(T, name)] = bestp'
    edit(anchor, anchor + '''
            if T <= 32:
                _v31_diag_emit("outer", m=T, layer=name,
                               us=[round(v, 2) if v < 1e29 else None for v in (t_tri, t_mm, t_bfp, t_bfpf)],
                               pick={False: "triton", True: "torch", "bfp": "bfp", "bfpf": "bfpf"}[bestp],
                               **({"ex": v31_outer_errors} if v31_outer_errors else {}))''')
    anchor = '''template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false, bool BND = false>
static void launch_sk2'''
    edit(anchor, HOST_HELPER + "\n" + anchor)
    anchor = '    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);'
    edit(anchor, HOST_CALL + anchor)
    edit('dl_okc_bf16proj_exactpatch_v29_', 'dl_okc_bf16proj_exactpatch_v31_diagnostic_')
    ast.parse(s)
    restored = s
    for e in reversed(edits):
        assert restored.count(e["new"]) == 1
        restored = restored.replace(e["new"], e["old"])
    assert restored == parent
    return s, edits


if __name__ == "__main__":
    source, edits = build()
    (HERE / "submission.py").write_text(source)
    (HERE / "edits.json").write_text(json.dumps(edits, indent=2) + "\n")
    manifest = {"parent_sha256": PARENT_SHA, "candidate_sha256": sha(source),
                "builder_sha256": sha(Path(__file__).read_bytes()), "source_bytes": len(source.encode()),
                "diagnostics_only": True, "device_tested": False, "promotion_allowed": False,
                "caps": {"selftest_bytes": 4096, "inner_choice_bytes": 2048, "outer_choice_bytes": 3072,
                         "host_resource_bytes": 4096},
                "gpu_work_unchanged": True, "selftest_and_tuning_decisions_unchanged": True}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
