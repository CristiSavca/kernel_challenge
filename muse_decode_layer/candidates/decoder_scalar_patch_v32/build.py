"""One representation rewrite over frozen V29, retaining frozen V31 observations."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SHA = "bbbc37f80aa3a5263dd0a66611bae62b2376c8b77ed67c14ba8ffd8db78e8a2c"
OBSERVED_SHA = "0d552bf386fdbcae6837d97371ae74aa778709b234d993b4e341e655fe9216f7"

VALUE_HELPERS = r'''// V32: COMP weight words stay in value objects; no decoded-array address escapes.
__device__ __forceinline__ uint4 sk2_decode_value_v32(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4) {
  const uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  const uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  return make_uint4(prmt(lw0, ha, 0x5140), prmt(lw0, ha, 0x7362),
                    prmt(lw1, hb, 0x5140), prmt(lw1, hb, 0x7362));
}

template <unsigned WORD>
__device__ __forceinline__ uint32_t sk2_patch_word_v32(uint32_t old, uint32_t d, int lane) {
  static_assert(WORD < 8, "BF16x2 word index");
  const unsigned v = (d >> 16) & 15u;
  if ((d >> 30) == 1u && lane == (int)((d >> 20) & 31u) && (v >> 1) == WORD) {
    const unsigned shift = (v & 1u) * 16u;
    const uint32_t mask = 0xffffu << shift;
    return (old & ~mask) | ((d & 0xffffu) << shift);
  }
  return old;
}

template <unsigned HALF>
__device__ __forceinline__ uint4 sk2_decode_patch_v32(uint32_t lw0, uint32_t lw1, uint32_t nw,
                                                    uint32_t base4, uint32_t d, int lane) {
  static_assert(HALF < 2, "eight-weight half");
  const uint4 q = sk2_decode_value_v32(lw0, lw1, nw, base4);
  return make_uint4(sk2_patch_word_v32<4 * HALF + 0>(q.x, d, lane),
                    sk2_patch_word_v32<4 * HALF + 1>(q.y, d, lane),
                    sk2_patch_word_v32<4 * HALF + 2>(q.z, d, lane),
                    sk2_patch_word_v32<4 * HALF + 3>(q.w, d, lane));
}
'''


def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def block(s, start):
    end = s.index("{", start) + 1
    depth = 1
    while depth:
        depth += (s[end] == "{") - (s[end] == "}")
        end += 1
    return s[start:end]


def build():
    parent = (HERE / "parent.py").read_text()
    assert sha(parent) == PARENT_SHA
    s = parent
    edits = json.loads((HERE / "observations.json").read_text())
    for e in edits:
        assert s.count(e["old"]) == 1
        s = s.replace(e["old"], e["new"])
    assert sha(s) == OBSERVED_SHA
    observation_edits = len(edits)

    def edit(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (old[:100], s.count(old), count)
        s = s.replace(old, new)
        edits.append({"old": old, "new": new, "count": count})

    old = block(s, s.index("__device__ __forceinline__ void sk2_patch_v29("))
    edit("// V29: exact BF16 bit repair; every b index is a compile-time unrolled literal.\n" + old,
         VALUE_HELPERS.rstrip())
    edit("      uint32_t b[NT][8];\n      uint32_t bl[HL ? NT : 1][8];",
         "      uint32_t b[COMP ? 1 : NT][8];  // unchanged non-COMP path only\n"
         "      uint4 bv0[COMP ? NT : 1], bv1[COMP ? NT : 1];\n"
         "      uint32_t bl[HL ? NT : 1][8];")
    edit("""            b[nt][0] = r0.x; b[nt][1] = r0.y; b[nt][2] = r0.z; b[nt][3] = r0.w;
            b[nt][4] = r1.x; b[nt][5] = r1.y; b[nt][6] = r1.z; b[nt][7] = r1.w;""",
         "            bv0[nt] = r0; bv1[nt] = r1;")
    edit("""            decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
            decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
            sk2_patch_v29(repair[nt], lane, b[nt]);""",
         """            bv0[nt] = sk2_decode_patch_v32<0>(lw[nt].x, lw[nt].y, cw[nt].x, base4, repair[nt], lane);
            bv1[nt] = sk2_decode_patch_v32<1>(lw[nt].z, lw[nt].w, cw[nt].y, base4, repair[nt], lane);""")
    edit("""            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);""",
         """            if constexpr (COMP) {
              const uint4 qv = j < 2 ? bv0[nt] : bv1[nt];
              const uint32_t b0 = (j & 1) ? qv.z : qv.x, b1 = (j & 1) ? qv.w : qv.y;
              if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b0, b1);
              else acc[j % NACC][0][nt][0] += __uint_as_float((b0 ^ b1) & a[j][0][0] & 0x007fffffu);
            } else {
              if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
              else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
            }""")
    old = "            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);"
    edit(old, """            if constexpr (COMP) {
              const uint4 qv = j < 2 ? bv0[nt] : bv1[nt];
              mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                       (j & 1) ? qv.z : qv.x, (j & 1) ? qv.w : qv.y);
            } else {
  """ + old + "\n            }")
    edit("sk2_exact_patch_v29", "sk2_exact_patch_v32", count=s.count("sk2_exact_patch_v29"))
    edit("dl_okc_bf16proj_exactpatch_v31_diagnostic_", "dl_okc_bf16proj_scalarpatch_v32_")
    edit("V31P", "V32P", count=s.count("V31P"))
    edit("V31R", "V32R", count=s.count("V31R"))
    ast.parse(s)
    restored = s
    for e in reversed(edits):
        assert restored.count(e["new"]) == e.get("count", 1)
        restored = restored.replace(e["new"], e["old"])
    assert restored == parent
    return s, edits, observation_edits


if __name__ == "__main__":
    source, edits, nobs = build()
    (HERE / "submission.py").write_text(source)
    (HERE / "edits.json").write_text(json.dumps(edits, indent=2) + "\n")
    manifest = {"parent_sha256": PARENT_SHA, "observation_parent_sha256": OBSERVED_SHA,
                "candidate_sha256": sha(source), "builder_sha256": sha(Path(__file__).read_bytes()),
                "observation_edits": nobs, "total_reversible_edits": len(edits),
                "geometry_packing_math_order_unchanged": True, "device_tested": False,
                "intended_marker": "sk2_exact_patch_v32", "COMP_template_index": 7}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
