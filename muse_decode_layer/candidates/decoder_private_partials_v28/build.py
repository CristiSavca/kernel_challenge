"""Build one reduction-only decoder candidate from the exact public incumbent.

This never imports or executes a candidate. V26 supplies only its reviewed
private-payload/owner-sum source; its serving arithmetic and launch are excluded.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
DONOR_SHA = "944768d40e380bfb88f917ed5e69fcc1f9470995d3e6d65cfb6267b756e04bb7"
PARENT = ROOT / f"experiments/candidates/muse_decode_layer/{PARENT_SHA}.py"
DONOR = ROOT / "serving/candidates/native_serving_v26/component.py"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def literal(text, name):
    return next(ast.literal_eval(node.value) for node in ast.parse(text).body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))


def make():
    parent, donor = PARENT.read_text(), DONOR.read_text()
    assert sha(parent) == PARENT_SHA and sha(donor) == DONOR_SHA
    dc = literal(donor, "CUDA_SRC")
    source, edits = parent, []

    def change(old, new):
        nonlocal source
        assert source.count(old) == 1, (old[:90], source.count(old))
        edits.append([old, new])
        source = source.replace(old, new)

    change("constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW",
           "constexpr size_t WGS_FIXF = (size_t)4 * WGS_SLOTS * 4 * WGS_FIXW")
    a = dc.index("// Two immutable boundary payloads per CTA/phase;")
    z = dc.index("__device__ __forceinline__ int* wgs_cnt(", a)
    private = dc[a:z].replace("p.fixbuf", "p.ybuf + (size_t)EMAX * HD").replace("v26_sum", "v28_sum")
    # Public global FP32 red.add flushes subnormal operands/results. Preserve
    # that policy; V26's serving-specific non-FTZ arithmetic is not imported.
    private = private.replace("add.rn.f32", "add.rn.ftz.f32")
    change("""__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
""", private)
    change("return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW)",
           "return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)4 * WGS_SLOTS * 4 * WGS_FIXW)")
    da = dc.index("  const int ntt =", dc.index('asm volatile("fence.acq_rel.gpu;'))
    dz = dc.index("  if (wt == 0) *fc = 0;", da)
    owner = dc[da:dz].replace("v26_sum", "v28_sum")
    a = source.index("  float* fb = wgs_fix(p, PB)")
    z = source.index("  if (wt == 0) *fc = 0;", a)
    change(source[a:z], owner)
    a = source.index("    float* fb = wgs_fix(p, PB)")
    z = source.index("    // kind 3:", a)
    da = dc.index("    const int boundary = u ==")
    dz = dc.index("    // kind 3:", da)
    change(source[a:z], dc[da:dz])
    change("p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW)",
           "p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)4 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW)")
    change("WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16   # moe WGS_FIXF: stream-K fixups + doneAs after ybuf (zero between calls)",
           "WGS_FIXF = 4 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16   # two immutable boundary payloads per CTA/phase; counters reset between calls")
    change('name=f"dl_okc_moe_kperm3_v12_{tag}_sm{arch}"',
           'name=f"dl_okc_moe_private_partials_v28_{tag}_sm{arch}"')
    change('template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p)',
           '#if OKC_WGS\n#define moe_fused moe_private_partials_v28\n#endif\n'
           'template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p)')
    change("# okc: problem=muse_decode_layer\n",
           "# okc: problem=muse_decode_layer\n"
           "# V28: immutable private split partials, ordered FP32 owner sum; exact 97af parent otherwise.\n"
           "# V26 supplies reduction code only; public weights, arithmetic, PDL and fused finalizer are retained.\n")
    ast.parse(source)
    restored = source
    for old, new in reversed(edits):
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == parent
    return source, edits


def build():
    source, edits = make()
    (HERE / "submission.py").write_text(source)
    (HERE / "edits.json").write_text(json.dumps(edits, indent=2) + "\n")
    manifest = dict(
        parent_sha256=PARENT_SHA, donor_reduction_source_sha256=DONOR_SHA,
        source_sha256=sha(source), builder_sha256=sha(Path(__file__).read_text()),
        edits_sha256=sha((HERE / "edits.json").read_text()),
        parent_moe_cuda_sha256=sha(literal(PARENT.read_text(), "MOE_CUDA_SRC")),
        candidate_moe_cuda_sha256=sha(literal(source, "MOE_CUDA_SRC")),
        private_payload_bytes=41943040, added_scratch_bytes=20971520,
        parent_public_job="7a8c4941741645ee", parent_private_job="b2a773e2d7194de8",
        affected_indices=[4, 5, 8, 9, 10, 11], control_indices=[0, 1, 2, 3, 6, 7],
        scope="Only WGS split-payload reduction, scratch offsets/size and isolated module namespace.")
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build()
