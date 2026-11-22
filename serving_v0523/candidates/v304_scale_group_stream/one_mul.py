"""Test whether a different BF16 exponent split plus a packed scale saves physical issue work."""
from pathlib import Path
import sys
base,out=map(Path,sys.argv[1:])
out.mkdir(exist_ok=True)
for image in sorted(base.glob('*.cu')):
    if image.stem.rsplit('_',1)[1] not in ('positive','atleast8'): continue
    src=image.read_text().replace('0x3c3c3c3cu', '0x77777777u')
    name=image.stem.replace('atleast8','one_mask').replace('positive','one_positive')
    pos='  CUDA_INLINE void transform_b(uint32_t buffer, uint32_t iteration) {\n    if constexpr (bands == 2) Base::transform_b(buffer, iteration);\n  }'
    add='''  CUDA_INLINE static uint32_t prmt(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t v;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a),"r"(b),"r"(c));
    return v;
  }
  CUDA_INLINE void prepare_band(uint32_t buffer, uint32_t j) {
    arith.may_process_bs_before_apply_on_b(j, buffer);
    uint32_t word = regs_qb[buffer][j], mag = word & 0x77777777u, scales = arith.dq_bs[j];
    uint32_t a = prmt(0x0f0e0c00u, 0x13121110u, mag);
    uint32_t b = prmt(0x0f0e0c00u, 0x13121110u, mag >> 16);
    PRAGMA_UNROLL
    for (uint32_t i=0; i<4; ++i) {
      uint32_t values = (prmt(a,b,0x8480u + 0x0101u*i) << 6) | (((word >> (4*i)) & 0x00080008u) << 12);
      uint32_t scale = __byte_perm(scales,scales,i % 2 ? 0x3232 : 0x1010);
      asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(regs_b[buffer][j][0][i]) : "r"(values),"r"(scale));
    }
  }
  CUDA_INLINE void transform_b(uint32_t buffer, uint32_t iteration) {
    if constexpr (bands == 2) {
      PRAGMA_UNROLL
      for (uint32_t j=0; j<bands; ++j) prepare_band(buffer,j);
    }
  }'''
    assert src.count(pos)==1;src=src.replace(pos,add)
    a=src.index('        auto* frag = reinterpret_cast<uint32_t*>(&input[j][0]);'); b=src.index('        wgmma_fence();',a)
    src=src[:a]+'        prepare_band(buffer,j);\n'+src[b:]
    (out/(name+'.cu')).write_text(src)
