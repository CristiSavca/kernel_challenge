"""Build one exact-register-repair SK2 experiment from the frozen public decoder."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
PARENT = ROOT / "experiments/candidates/muse_decode_layer" / (PARENT_SHA + ".py")

PATCH_HELPER = r'''
// V29: exact BF16 bit repair; every b index is a compile-time unrolled literal.
__device__ __forceinline__ void sk2_patch_v29(uint32_t d, int lane, uint32_t (&b)[8]) {
  if ((d >> 30) == 1u && lane == (int)((d >> 20) & 31u)) {
    const unsigned v = (d >> 16) & 15u;
    const unsigned shift = (v & 1u) * 16u;
    const uint32_t mask = 0xffffu << shift;
    const uint32_t bits = (d & 0xffffu) << shift;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if ((v >> 1) == (unsigned)j) b[j] = (b[j] & ~mask) | bits;
  }
}
'''

COMPRESS = '''def bfp_sk2_compress(w):
    """V29: parent 12-bit primary plus exact single-bit-pattern repair or raw unit.

    Descriptor: top two bits 00=none, 01=single(position9,bits16), 10=raw index30.
    No distribution assumption: every overflow unit retains all 1024 original bytes.
    Entire matrices whose requested bytes exceed raw BF16 use existing other backends.
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
    esc = ((e71 < base) | (e71 > base + 7)).reshape(n // 8, k // 64, 512)
    counts = esc.sum(-1)
    raw_mask, single = counts >= 2, counts == 1
    raw_count, single_count = int(raw_mask.sum()), int(single.sum())
    units = (n // 8) * (k // 64)
    if units * 772 + raw_count * 1024 >= units * 1024:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    codes = torch.cat([c8[..., 0:4] | (c8[..., 4:8] << 4),
                       c8[..., 8:12] | (c8[..., 12:16] << 4)], dim=-1)
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512),
                        codes.reshape(T, U, 256)], dim=-1).contiguous()
    position = esc.to(torch.int32).argmax(-1)
    original = bits.reshape(T, U, 512).gather(-1, position.unsqueeze(-1)).squeeze(-1)
    desc = torch.zeros((T, U), dtype=torch.int64, device=w.device)
    desc[single] = (1 << 30) | (position[single].long() << 16) | original[single].long()
    raw_index = torch.arange(raw_count, dtype=torch.int64, device=w.device)
    desc[raw_mask] = (2 << 30) | raw_index
    raw = bits.reshape(T, U, 512)[raw_mask].to(torch.int16).contiguous().view(torch.uint8).reshape(raw_count, 1024)
    if raw_count == 0:
        raw = torch.zeros((1, 1024), dtype=torch.uint8, device=w.device)
    return {"packed": packed, "base4": base * 0x01010101,
            "descriptors": desc.to(torch.int32).contiguous(), "raw": raw,
            "stats": (units, single_count, raw_count, units * 772 + raw_count * 1024)}
'''

STREAM = '''def bfp_sk2_patch_stream_v29(comp, n_cfg, R, NT, NWR, WSK, UPW):
    """Append aligned descriptor tail to each complete stage/CTA primary payload."""
    primary = bfp_sk2_stream(comp["packed"], n_cfg, R, NT, NWR, WSK, UPW)
    desc = bfp_sk2_stream(comp["descriptors"].unsqueeze(-1), n_cfg, R, NT, NWR, WSK, UPW)
    s, c = primary.shape[:2]
    tail = desc.contiguous().view(torch.uint8).reshape(s, c, -1)
    assert tail.shape[-1] % 16 == 0
    return torch.cat([primary.reshape(s, c, -1), tail], dim=-1).contiguous()
'''

TABLE = '''def bfp_sk2_table(comp, n_cfg, R):
    """V29 ABI: ptab retains the immutable BF16 sidecar; no shared correction table."""
    return comp["raw"], 0
'''


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def build():
    parent = PARENT.read_text()
    assert sha(parent) == PARENT_SHA
    text, edits = parent, []

    def edit(old, new, count=1):
        nonlocal text
        assert text.count(old) == count, (old[:110], text.count(old), count)
        text = text.replace(old, new)
        edits.append({"old": old, "new": new, "count": count})

    def function(name, replacement):
        n = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == name)
        lines = text.splitlines(True)
        edit(''.join(lines[n.lineno-1:n.end_lineno]), replacement)

    # Only the SK2 CUDA literal is edited below; shared dense decode8 is unchanged.
    tree = ast.parse(text)
    n = next(x for x in tree.body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bfp_SK2_SRC' for t in x.targets))
    src = ast.literal_eval(n.value)
    before = src
    def cuda(old, new, count=1):
        nonlocal src
        assert src.count(old) == count, (old[:100],src.count(old),count)
        src = src.replace(old, new)

    cuda('// ============================================================================ sk2 (m <= 32): swapped operands', PATCH_HELPER + '\n// ============================================================================ sk2 (m <= 32): swapped operands')
    cuda('(NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32', '(NWR * WSK + 1 + (COMP ? 0 : 1)) * 32')
    cuda('sk2_kernel', 'sk2_exact_patch_v29', 2)
    cuda('constexpr int WBY = NW * UPW * NT * TU;', 'constexpr int WPAY = NW * UPW * NT * TU;\n  constexpr int WBY = WPAY + (COMP ? ((NW * UPW * NT * 4 + 15) & ~15) : 0);',2)
    cuda('constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;', 'constexpr int NESC = COMP ? 0 : 1;',2)
    cuda('mbar_init(cbar, 32 * NESC);', 'if constexpr (NESC > 0) mbar_init(cbar, 32 * NESC);',2)
    start = src.index('    if (COMP) {   // this CTA\'s escape table')
    stop = src.index('    if constexpr (CL == 1) {', start + 100)
    # The first CL branch belongs to the table block: locate its closing tail.
    stop = src.index('    if constexpr (CL == 1) {\n      mbar_init(empty0', start)
    cuda(src[start:stop], '    // V29 sidecar is immutable global weight data, not copied into the ring.\n')
    start = src.index('  if (warp >= NW + 1) {')
    stop = src.index('\n  // ---------------- consumers',start)
    cuda(src[start:stop], '''  if constexpr (!COMP) {
    if (warp >= NW + 1) {
      mbar_arrive(cbar);
      __syncwarp();
      if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      return;
    }
  }
''')
    cuda('if constexpr (COMP) mbar_wait(cbar, 0);', 'if constexpr (COMP && NESC > 0) mbar_wait(cbar, 0);',2)
    cuda('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");', '  static_assert(!COMP || CL == 1, "V29 compressed geometry is CTA-local");\n  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");')
    cuda('      uint2 cw[NT];', '      uint2 cw[NT];\n      uint32_t repair[COMP ? NT : 1];')
    old='''        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {'''
    new='''        if constexpr (COMP) {
          const uint32_t ha = s0 + XBY + WPAY + ((warp * UPW + uu) * NT + nt) * 4;
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(repair[nt]) : "r"(ha));
          if ((repair[nt] >> 30) == 2u) {
            const uint4* rp = reinterpret_cast<const uint4*>(ptab) +
                (size_t)(repair[nt] & 0x3fffffffu) * 64 + lane * 2;
            const uint4 r0 = __ldg(rp), r1 = __ldg(rp + 1);
            b[nt][0] = r0.x; b[nt][1] = r0.y; b[nt][2] = r0.z; b[nt][3] = r0.w;
            b[nt][4] = r1.x; b[nt][5] = r1.y; b[nt][6] = r1.z; b[nt][7] = r1.w;
          } else {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
          }
        } else {'''
    cuda(old,new)
    cuda('''          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);''','''          if ((repair[nt] >> 30) != 2u) {
            decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
            decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
            sk2_patch_v29(repair[nt], lane, b[nt]);
          }''')
    edit(before, src)
    function('bfp_sk2_compress', COMPRESS)
    function('bfp_sk2_table', STREAM + '\n\n' + TABLE)
    edit('bfp_sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW)', 'bfp_sk2_patch_stream_v29(comp, n_rows, R, NT, NWR, WSK, UPW)')
    edit('''                state["sk2_comp"][layer] = comp
            if comp is None:''','''                state["sk2_comp"][layer] = comp
                if comp is not None:
                    print("SK2_PATCH_V29", layer, "units,single,raw,requested_bytes", comp["stats"], flush=True)
            if comp is None:''')
    edit('dl_okc_bf16proj_v86y4b1_', 'dl_okc_bf16proj_exactpatch_v29_')
    # Geometry selection/ring depth stay parent-identical. Host/device compute actual larger stages.
    ast.parse(text)
    restored=text
    for e in reversed(edits):
        assert restored.count(e['new']) == e['count']
        restored=restored.replace(e['new'],e['old'])
    assert restored == parent
    return text, edits


if __name__ == '__main__':
    text, edits=build()
    (HERE/'submission.py').write_text(text)
    (HERE/'edits.json').write_text(json.dumps(edits,indent=2)+'\n')
    (HERE/'manifest.json').write_text(json.dumps({'parent_sha256':PARENT_SHA,'candidate_sha256':sha(text),
        'source_bytes':len(text.encode()),'reversible':True,'device_tested':False},indent=2)+'\n')
    print(sha(text))
