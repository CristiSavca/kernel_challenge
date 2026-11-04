"""Fixed matrix-window codec with missing-high-nibble completion for escaped units."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SHA = "ea4470fc16989b9d66cbab40fdae588fb30b2ffe5446f86c14e8b13854813e35"

COMPRESS = '''def bfp_sk2_compress(w):
    """V33: original matrix window; escaped units retain only missing high nibbles.

    Every primary unit is768B: low8 bits plus four code bits/weight. Normal
    units use parent sign/window codes. Raw-mode codes are literal high-byte
    low nibbles; a256B sidecar supplies high nibbles, including sign. Header0
    means normal; bit31 plus index31 selects the exact raw completion unit.
    """
    n, k = w.shape
    tu = bfp_sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    off = (e71 - base).clamp(0, 7)
    code = (sign << 3) | off
    T, U = n // 8, k // 64
    esc = ((e71 < base) | (e71 > base + 7)).reshape(T, U, 512)
    raw_mask = esc.sum(-1) > 0
    raw_count = int(raw_mask.sum())
    units = T * U
    if units * 772 + raw_count * 256 >= units * 1024:
        return None
    assert raw_count < (1 << 31)
    code[raw_mask] = (hi & 15)[raw_mask]
    c8 = code.to(torch.uint8)
    codes = torch.cat([c8[..., 0:4] | (c8[..., 4:8] << 4),
                       c8[..., 8:12] | (c8[..., 12:16] << 4)], dim=-1)
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512),
                        codes.reshape(T, U, 256)], dim=-1).contiguous()
    h8 = (hi >> 4)[raw_mask].to(torch.uint8)
    raw = torch.cat([h8[..., 0:4] | (h8[..., 4:8] << 4),
                     h8[..., 8:12] | (h8[..., 12:16] << 4)], dim=-1).reshape(raw_count, 256).contiguous()
    if raw_count == 0:
        raw = torch.zeros((1, 256), dtype=torch.uint8, device=w.device)
    desc = torch.zeros((T, U), dtype=torch.int64, device=w.device)
    desc[raw_mask] = (1 << 31) | torch.arange(raw_count, dtype=torch.int64, device=w.device)
    return {"packed": packed, "base4": base * 0x01010101,
            "descriptors": desc.to(torch.int32).contiguous(), "raw": raw,
            "stats": (units, units - raw_count, raw_count, units * 772 + raw_count * 256)}
'''

RAW_HELPER = r'''// V33 raw completion: identical nibble order in primary and missing-bit planes.
__device__ __forceinline__ uint4 sk2_decode_raw_v33(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t hw) {
  const uint32_t ha = (nw & 0x0f0f0f0fu) | ((hw & 0x0f0f0f0fu) << 4);
  const uint32_t hb = ((nw >> 4) & 0x0f0f0f0fu) | (hw & 0xf0f0f0f0u);
  return make_uint4(prmt(lw0, ha, 0x5140), prmt(lw0, ha, 0x7362),
                    prmt(lw1, hb, 0x5140), prmt(lw1, hb, 0x7362));
}
'''


def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def fn(s, name):
    n = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(s, n)


def build():
    parent = (HERE / "parent.py").read_text()
    assert sha(parent) == PARENT_SHA
    s, edits = parent, []
    def edit(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (old[:100], s.count(old), count)
        s = s.replace(old, new)
        edits.append({"old": old, "new": new, "count": count})
    edit(fn(s, "bfp_sk2_compress"), COMPRESS.rstrip())
    start = s.index("template <unsigned WORD>\n__device__ __forceinline__ uint32_t sk2_patch_word_v32")
    end = s.index("// ============================================================================ sk2", start)
    edit(s[start:end], RAW_HELPER + "\n")
    start = s.index("        if constexpr (COMP) {\n          const uint32_t ha = s0 + XBY + WPAY")
    end = s.index("\n        } else {\n          const uint32_t a0 = wb", start)
    edit(s[start:end], '''        if constexpr (COMP) {
          // Primary payload has no dependency on the header: issue it unconditionally.
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
          const uint32_t ha = s0 + XBY + WPAY + ((warp * UPW + uu) * NT + nt) * 4;
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(repair[nt]) : "r"(ha));''')
    edit('''          if ((repair[nt] >> 30) != 2u) {
            bv0[nt] = sk2_decode_patch_v32<0>(lw[nt].x, lw[nt].y, cw[nt].x, base4, repair[nt], lane);
            bv1[nt] = sk2_decode_patch_v32<1>(lw[nt].z, lw[nt].w, cw[nt].y, base4, repair[nt], lane);
          }''', '''          if (repair[nt] & 0x80000000u) {
            const uint2* hp = reinterpret_cast<const uint2*>(ptab) +
                (size_t)(repair[nt] & 0x7fffffffu) * 32 + lane;
            const uint2 h = __ldg(hp);
            bv0[nt] = sk2_decode_raw_v33(lw[nt].x, lw[nt].y, cw[nt].x, h.x);
            bv1[nt] = sk2_decode_raw_v33(lw[nt].z, lw[nt].w, cw[nt].y, h.y);
          } else {
            bv0[nt] = sk2_decode_value_v32(lw[nt].x, lw[nt].y, cw[nt].x, base4);
            bv1[nt] = sk2_decode_value_v32(lw[nt].z, lw[nt].w, cw[nt].y, base4);
          }''')
    edit('"""V29 ABI: ptab retains the immutable BF16 sidecar; no shared correction table."""',
         '"""V33 ABI: ptab retains immutable missing-high-nibble units; no shared correction table."""')
    edit('print("SK2_PATCH_V29", layer, "units,single,raw,requested_bytes", comp["stats"], flush=True)',
         'print("SK2_NIBBLE_V33", layer, "units,normal,raw,requested_bytes", comp["stats"], flush=True)')
    edit("sk2_exact_patch_v32", "sk2_nibble_sidecar_v33", s.count("sk2_exact_patch_v32"))
    edit("dl_okc_bf16proj_scalarpatch_v32_", "dl_okc_bf16proj_nibblesidecar_v33_")
    edit("V32P", "V33P", s.count("V32P"))
    edit("V32R", "V33R", s.count("V32R"))
    ast.parse(s)
    restored = s
    for e in reversed(edits):
        assert restored.count(e["new"]) == e["count"]
        restored = restored.replace(e["new"], e["old"])
    assert restored == parent
    return s, edits


if __name__ == "__main__":
    source, edits = build()
    (HERE / "submission.py").write_text(source)
    (HERE / "edits.json").write_text(json.dumps(edits, indent=2) + "\n")
    manifest = {"parent_sha256": PARENT_SHA, "candidate_sha256": sha(source),
                "builder_sha256": sha(Path(__file__).read_bytes()), "reversible_edits": len(edits),
                "codec": "matrix_window_missing_high_nibble", "unit_primary_bytes": 768,
                "unit_header_bytes": 4, "raw_sidecar_unit_bytes": 256,
                "selection_geometry_and_mma_order_unchanged": True,
                "marker": "sk2_nibble_sidecar_v33", "COMP_template_index": 7,
                "device_tested": False}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
