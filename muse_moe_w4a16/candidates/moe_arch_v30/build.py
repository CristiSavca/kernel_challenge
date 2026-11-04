"""Reproducible source-local V30 rewrite. No CUDA execution or imports."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT_SHA = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def value(source, name):
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)


def replace_once(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


def extract(s, start, end):
    a = s.index(start)
    return s[a:s.index(end, a)]


def device_addition(cu):
    gather = extract(cu, "__device__ void wgs_gather(", "// Work left from the previous segment")
    gather = gather[:gather.index("  if (tid == 0) {  // count this CTA's rows")] + "}\n"
    gather = replace_once(gather, "void wgs_gather(", "void v30_gather(")
    seg = extract(cu, "template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(",
                  "template <bool PB>\n__device__ __forceinline__ void wgs_phase(")
    seg = seg[:seg.index("  if (OKC_WGSX & 8)")]
    seg = replace_once(seg, "template <bool PB, int NTT>", "template <bool PB, bool COLD, int NTT>")
    seg = replace_once(seg, "void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,\n                                        int& seq, WgsPend& pd)",
                       "void v30_seg(const Params& p, Smem& S, unsigned char* ring, int j, int ub, int& seq)")
    seg = replace_once(seg, "  constexpr int UBL = PB ? 4 : 5;", "  constexpr int WBY = COLD ? 18432 : UNIT_BYTES;\n  constexpr int XBY = COLD ? 4096 : WGS_XS;")
    seg = seg.replace("WGS_NS * UNIT_BYTES", "WGS_NS * WBY").replace("ws * UNIT_BYTES", "ws * WBY")
    seg = seg.replace("xs * (WGS_XS >> 4)", "xs * (XBY >> 4)").replace("xs * WGS_XS", "xs * XBY")
    old_flush = """        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
"""
    seg = replace_once(seg, old_flush, "")
    seg = replace_once(seg, "for (int st = s0 + 1; st < s1; ++st)", "for (int st = 1; st < 16; ++st)")
    seg += "  const int r16 = ub * (COLD ? 8 : 16) + wg * 4 + wp;\n#pragma unroll\n  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n}\n"
    final = extract(cu, "#if OKC_WGS && OKC_WGSFINSS\n  // item", "#elif OKC_FINCOL")
    final = final[final.index("\n") + 1:]
    template = (HERE / "new_kernels.cuh").read_text()
    new = gather + template.replace("// @V30_SEGMENT@", seg).replace("// @V30_FINAL@", final)
    return "\n#if defined(OKC_V30) && OKC_V30\n" + new + "\n#endif\n"


def build():
    source = (HERE / "parent.py").read_text()
    assert sha(source) == PARENT_SHA
    cu = value(source, "CUDA_SRC")
    added = device_addition(cu)
    result = replace_once(source, "}  // namespace okc", added + "\n}  // namespace okc")
    host = "\n#if defined(OKC_V30) && OKC_V30\n" + (HERE / "new_host.cuh").read_text() + "\n#endif\n"
    result = replace_once(result, "void okc_set_pdl(int64_t on) {", host + "\nvoid okc_set_pdl(int64_t on) {")
    start = "  static bool attr = false;\n  size_t sm = okc_smem_bytes();"
    result = replace_once(result, start, """#if defined(OKC_V30) && OKC_V30
  TORCH_CHECK(!g_okc_pdl, "V30 standalone uses ordinary ordered stream launches");
  v30_launch(p, (int)grid, at::cuda::getCurrentCUDAStream());
#else
""" + start)
    end = "\n}\n\n\n'''\n\n\nCPP_SRC"
    result = replace_once(result, end, "\n#endif  // V30 dispatch\n}\n\n\n'''\n\n\nCPP_SRC")
    result = replace_once(result, "    return MAIN_DEFINES\n\n\ndef _wgs_on():", "    return MAIN_DEFINES + (\"-DOKC_V30=1\",)\n\n\ndef _wgs_on():")
    result = replace_once(result, 'name=f"okc_moe_kperm3_v9_{tag}_sm{arch}"', 'name=f"okc_moe_wholek_v30_{tag}_sm{arch}"')
    result = replace_once(result, "            verbose=False,", '            verbose=("-DOKC_V30=1" in defs),')
    result = replace_once(result, "# Isolated KPERM3 adaptation over frozen ef7b210b; no SC6/MFIN or pipeline changes.",
                          "# V30: whole-K MAIN tiles and GPU sparse/dense job partition over frozen 247888acca.\n# KPERM3 lineage below remains; public fallback/tiny source and all packed formats are preserved.")
    ast.parse(result)
    return result


if __name__ == "__main__":
    result = build()
    (HERE / "submission.py").write_text(result)
    manifest = {"parent_sha256": PARENT_SHA, "source_sha256": sha(result),
                "files": {p.name: sha(p.read_bytes()) for p in
                          [HERE / "build.py", HERE / "new_kernels.cuh", HERE / "new_host.cuh"]},
                "gpu_status": "NOT_RUN", "dispatch": "MAIN_PRMT_only; tiny/MID/general-scale fallbacks unchanged",
                "launches": ["route_gather", "coldA", "hotA", "coldB", "hotB", "final"]}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
