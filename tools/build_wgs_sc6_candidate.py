#!/usr/bin/env python3
"""Fixed, lossless WGS scale packing over the exact scored KPERM3 parent."""
import ast
import hashlib
from pathlib import Path
from build_wgs_shared_a_candidate import HOST as _RESOURCE_HOST

ROOT=Path(__file__).resolve().parents[1]
PARENT_SHA='247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0'
PARENT=ROOT/f'experiments/candidates/muse_moe_w4a16/{PARENT_SHA}.py'
CANDIDATE=ROOT/'muse_moe_w4a16/candidates/native_wgs_sc6_v16.py'
HOST=(_RESOURCE_HOST.replace('okc_shared_a_resources','okc_sc6_resources')
      .replace('SSAR v=1 big=', 'SC6R v=1 sc6=%d big=')
      .replace('          okc::NTL, sm,', '          (int)OKC_SC6W, okc::NTL, sm,')
      .replace('shared-A WGS resource admission failed; inspect SSAR records', 'SC6 WGS resource admission failed; inspect SC6R records'))


def once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a))
    return s.replace(a,b)


HELPERS='''def _sc6_scales_exact(*sfs):
    """Lossless SC6 domain; 0x7f is excluded as E4M3 NaN."""
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x40) | (b > 0x7E)).any()):
            return False
    return True


def _sc6_defines(defines, enabled):
    """Encoding follows this prepared layout, never mutable global state."""
    defines = tuple(defines)
    return defines + (("-DOKC_SC6W=1",) if enabled and "-DOKC_WGS=1" in defines else ())


def _sc6_wgs_pack(s):
    """Qualified [...,1024] exact scale bytes -> W512/H256 bit planes.
    Caller qualifies the entire W13/W2 bank before packing any expert.
    Layout/decode adapted from licensed keell0renz commit890809cee421500b.
    """
    lead = s.shape[:-1]
    v = s.view(*lead, 128, 8).to(torch.int32) - 0x40
    v6, v7 = v[..., 6], v[..., 7]
    pieces = [(v6 >> 0) & 3, (v6 >> 2) & 3, (v6 >> 4) & 3,
              v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
    wb = torch.stack([(v[..., n] << 2) | pieces[n] for n in range(4)], -1).to(torch.uint8)
    hb = torch.stack([(v[..., 4 + n] << 2) | pieces[4 + n] for n in range(2)], -1).to(torch.uint8)
    return torch.cat([wb.reshape(*lead, 512), hb.reshape(*lead, 256)], -1).contiguous()


'''


SC6_DECODE='''#if OKC_SC6W
    // Exact six-bit scale values in W/H planes, one record per lane per stage.
    uint32_t scale_pairs[4];
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2));
      const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;
      scale_pairs[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      scale_pairs[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      scale_pairs[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      scale_pairs[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;
    }
#else
'''


def modify_cuda(s):
    s=once(s,'#if OKC_WGS\n// ---------------------------------------------------------------- WGS:', '#ifndef OKC_SC6W\n#define OKC_SC6W 0\n#endif\n\n#if OKC_WGS\nstatic_assert(!OKC_SC6W || OKC_PRMTSC, "SC6 requires the qualified PRMT scale domain");\n// ---------------------------------------------------------------- WGS:')
    s=once(s,"constexpr int WGS_PIECE = 9216;                 // one warpgroup's weights in a stage", "constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;  // unchanged codes, lossless W/H scale planes\nconstexpr int WGS_STG = 4 * WGS_PIECE;           // global payload; shared slots keep UNIT_BYTES")
    a=s.index('__device__ void wgs_wproducer(');b=s.index('// x producer:',a)
    producer=s[a:b]
    producer=once(producer,'* (size_t)UNIT_BYTES;', '* (size_t)WGS_STG;')
    assert producer.count('"r"(UNIT_BYTES)')==2
    producer=producer.replace('"r"(UNIT_BYTES)', '"r"(WGS_STG)')
    s=s[:a]+producer+s[b:]
    s=once(s,'  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode\'s multiply stays an IMAD (FMA pipe)', '#if OKC_SC6W\n  const uint32_t k16c = 4u + p.zero;  // exact shifted six-bit planes, uint32 IMAD wrap\n#else\n  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode\'s multiply stays an IMAD (FMA pipe)\n#endif')
    a=s.index('    const uint2 scn =',s.index('void wgs_seg('));b=s.index('#pragma unroll\n    for (int lg',a)
    s=s[:a]+SC6_DECODE+s[a:b]+'#endif\n'+s[b:]
    s=once(s,'void okc_run(std::vector<torch::Tensor>',HOST+'\nvoid okc_run(std::vector<torch::Tensor>')
    s=once(s,'    attr = true;','#if OKC_WGS\n    okc_sc6_resources(sm, e0, e1);\n#endif\n    attr = true;')
    return s


def build():
    text=PARENT.read_text();assert hashlib.sha256(text.encode()).hexdigest()==PARENT_SHA
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    cuda=ast.literal_eval(node.value);text=once(text,cuda,modify_cuda(cuda))
    text=once(text,'def _wgs_copy(codes, sf, rows, positions=None):',HELPERS+'def _wgs_copy(codes, sf, rows, positions=None, sc6=False):')
    text=once(text,'[warpgroup q (4)][9216 B].', '[warpgroup q (4)][8960 B SC6 / 9216 B ordinary].')
    text=once(text,'    This format retains 8-bit scales and both existing code-bit positions mappings.', '    Both code-bit layouts are preserved; qualified SC6 packs exact scales into six bits.')
    target='    return torch.cat([cb, s], dim=3).contiguous().view(-1)'
    text=once(text,target,'    if sc6:\n        s = _sc6_wgs_pack(s)\n'+target)
    start=text.index('def _prepare_layout(');stop=text.index('\n\ndef prepare(',start);prep=text[start:stop]
    prmt='    prmt = PRMT_SCALES and _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])\n'
    prep=once(prep,prmt,'')
    prep=once(prep,'    lo, cnt = 0, 32\n','    lo, cnt = 0, 32\n'+prmt+'    sc6 = bool(use_mul and prmt and _sc6_scales_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt]))\n')
    prep=once(prep,'    if _wgs_on():', '    state["sc6_wgs"] = sc6\n    if _wgs_on():')
    prep=once(prep,'wgs_rowsA, positions) for e', 'wgs_rowsA, positions, sc6=sc6) for e')
    prep=once(prep,'rowsB, positions) for e', 'rowsB, positions, sc6=sc6) for e')
    prep=once(prep,'        _ext(d, use_mul=use_mul)', '        _ext(_sc6_defines(d, sc6), use_mul=use_mul)')
    prep=once(prep,'    state["use_mul"] = use_mul', '    if use_mul:\n        print(f"SC6W enabled={int(sc6)} piece={8960 if sc6 else 9216} stage={35840 if sc6 else 36864}", flush=True)\n    state["use_mul"] = use_mul')
    text=text[:start]+prep+text[stop:]
    text=once(text,'ext = _ext(tuple(_main_defines(sfx)) + sfx, use_mul=state["use_mul"])', 'ext = _ext(_sc6_defines(tuple(_main_defines(sfx)) + sfx, state["sc6_wgs"]), use_mul=state["use_mul"])')
    text=once(text,'name=f"okc_moe_kperm3_v9_{tag}_sm{arch}",', 'name=f"okc_moe_wgs_sc6_v16_{tag}_sm{arch}",')
    text=once(text,'            verbose=False,', '            verbose="-DOKC_WGS=1" in defs,  # retain WGS compiler evidence')
    text=once(text,'# Isolated KPERM3 adaptation over frozen ef7b210b; no SC6/MFIN or pipeline changes.', '# Isolated lossless WGS SC6 scales over scored 247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0.\n# SC6 layout/decode adapted from the same Apache-2.0 keell0renz source below.\n# Only qualified larger-layout WGS payloads change; parent math, KPERM and pipeline remain.')
    ast.parse(text);return text


if __name__=='__main__':
    text=build();CANDIDATE.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(),CANDIDATE)
