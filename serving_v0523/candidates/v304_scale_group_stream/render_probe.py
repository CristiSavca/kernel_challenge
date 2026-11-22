"""Specialize the actual vendor scale helper and keep the native dot and weight transform."""
from pathlib import Path
import sys

base, out = map(Path,sys.argv[1:])
out.mkdir(exist_ok=True)
marker='// Each output band accumulates the four K16 slices in their native order.'
for path in base.glob('*.cu'):
    if '_d' in path.stem: continue
    if path.stem not in ('w13_m8_n256','w13_m16_n512','w13_m24_n512','w2_m8_n128','w2_m16_n256','w2_m24_n256'): continue
    src=path.read_text()
    assert src.count(marker)==1
    (out/(path.stem+'_native.cu')).write_text(src)
    for method in ('simd','positive','pairs','raw','bias','sign','expand','atleast8'):
        if method in ('simd','positive','bias','sign','expand','atleast8'):
            impl='''uint32_t high = 0x3c3c3c3cu + ((bytes >> 4) & 0x07070707u);
    MASK
    uint32_t low = (bytes & 0x0f0f0f0fu) << 4;
    dq_bs[j] = __byte_perm(low, high, 0x5140);
    dq_bs[j + 1] = __byte_perm(low, high, 0x7362);'''
            impl=impl.replace('MASK','high &= __vcmpne4(bytes, 0u);' if method=='simd' else '')
            if method in ('bias','sign','expand','atleast8'):
                prefix = {
                    'bias': 'uint32_t one = ((bytes + 0x7f7f7f7fu) & 0x80808080u) >> 7;\n    uint32_t high = one * 60u + ((bytes >> 4) & 0x07070707u);',
                    'sign': 'uint32_t nonzero;\n    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(nonzero) : "r"(bytes + 0x7f7f7f7fu));\n    uint32_t high = (nonzero & 0x3c3c3c3cu) + ((bytes >> 4) & 0x07070707u);',
                    'expand': 'uint32_t one = ((bytes + 0x7f7f7f7fu) & 0x80808080u) >> 5;\n    uint32_t high = (one << 4) - one + ((bytes >> 4) & 0x07070707u);',
                    'atleast8': 'uint32_t nonzero;\n    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(nonzero) : "r"(bytes + 0x78787878u));\n    uint32_t high = (0x3c3c3c3cu + ((bytes >> 4) & 0x07070707u)) & nonzero;',
                }[method]
                impl=prefix+'\n    uint32_t low'+impl.split('uint32_t low',1)[1]
        elif method=='pairs':
            impl='''constexpr uint32_t bias = 0x3c003c00u;
    uint32_t even = bytes & 0x00ff00ffu, odd = (bytes >> 8) & 0x00ff00ffu;
    uint32_t x = (even << 4) + (__vcmpne2(even, 0u) & bias);
    uint32_t y = (odd << 4) + (__vcmpne2(odd, 0u) & bias);
    dq_bs[j] = __byte_perm(x, y, 0x5410);
    dq_bs[j + 1] = __byte_perm(x, y, 0x7632);'''
        else:
            impl='''PRAGMA_UNROLL
    for (unsigned k = 0; k < 2; ++k) {
      const uint32_t a = (bytes >> (16 * k)) & 255u;
      const uint32_t b = (bytes >> (16 * k + 8)) & 255u;
      dq_bs[j + k] = (a ? (a << 4) + 0x3c00u : 0u) | ((b ? (b << 4) + 0x3c00u : 0u) << 16);
    }'''
        custom='''template <> CUDA_INLINE void MainloopArithmetic<MuseOverlapCtx>::may_process_bs_before_apply_on_b(uint32_t j, uint32_t buffer_id) {
  static_assert(ElementBS::kBits == 8 && ElementA::kBits == 16 && kIsGroupWeightScale && !kUseFusedE8m0Scale);
  if ((j & 1u) == 0) {
    uint32_t bytes = bs[buffer_id][j / 2];
    BODY
  }
}
'''.replace('BODY',impl)
        (out/(path.stem+'_'+method+'.cu')).write_text(src.replace(marker,custom+'\n'+marker))
