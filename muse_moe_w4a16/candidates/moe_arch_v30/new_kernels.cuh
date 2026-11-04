// V30: GPU-created cold/hot job directories; no values are cached across calls.
static_assert(OKC_NW == 16 && OKC_FT_T == 64 && OKC_WGSGS == 2,
              "V30 preserves the pinned MAIN matrix schedule");
static_assert(OKC_WGSFINSS && OKC_PRMTSC && OKC_MULDEC, "V30 exact parent format");
constexpr size_t V30_META = ((sizeof(Smem) + 127) / 128) * 128;
static_assert(V30_META == 22272, "fixed metadata layout");
static_assert(sizeof(Smem) <= WGS_FIXF * sizeof(float), "metadata fits unused fixup region");

// Only defined fields are exported. ent/eid are local to route/gather.
__global__ void __launch_bounds__(512, 1) v30_route_gather(Params p, Smem* meta) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x;
  route(p, S);
  if (tid < 32) {
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, d);
        if (tid >= d) incl += t;
      }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
  v30_gather(p, S);
  if (blockIdx.x == 0) {
    const int nj = S.njF + S.njS;
    for (int j = tid; j < nj; j += NT) {
      meta->jE[j] = S.jE[j]; meta->jS[j] = S.jS[j];
      meta->jC[j] = S.jC[j]; meta->jrow[j] = S.jrow[j];
    }
    for (int i = tid; i < p.M * 8; i += NT) meta->pos[i] = S.pos[i];
    if (tid == 0) {
      meta->jrow[nj] = S.jrow[nj];
      int cold = 0;
      for (int j = 0; j < nj; ++j) if (S.jC[j] <= 8) meta->forder[cold++] = j;
      int end = cold;
      for (int j = 0; j < nj; ++j) if (S.jC[j] > 8) meta->forder[end++] = j;
      meta->njF = cold; meta->njS = end - cold;
    }
  }
}

template <bool PB, bool COLD>
__device__ __forceinline__ void v30_producer(const Params& p, Smem& S,
                                           unsigned char* ring, bool xproducer) {
  if ((threadIdx.x & 31) != 0) return;
  constexpr int WBY = COLD ? 18432 : UNIT_BYTES;
  constexpr int XBY = COLD ? 4096 : WGS_XS;
  constexpr int UBL = PB ? (COLD ? 5 : 4) : (COLD ? 6 : 5);
  constexpr int OLD_UBL = PB ? 4 : 5;
  const int nj = COLD ? S.njF : S.njS;
  const int offset = COLD ? 0 : S.njF;
  const unsigned char* base = PB ? p.wBs : p.wAs;
  const __nv_bfloat16* xbase = PB ? p.xB : p.xA;
  unsigned char* xring = ring + WGS_NS * WBY;
  int seq = 0;
  for (int q = blockIdx.x; q < (nj << UBL); q += gridDim.x) {
    const int j = S.forder[offset + (q >> UBL)];
    const int ub = q & ((1 << UBL) - 1);
    const int oldub = COLD ? (ub >> 1) : ub;
    const int half = COLD ? (ub & 1) * 18432 : 0;
    const int npad = ((S.jC[j] + 7) >> 3) << 3;
    const __nv_bfloat16* xb = xbase + (size_t)S.jrow[j] * HD;
    for (int st = 0; st < 16; ++st, ++seq) {
      const int sl = seq % WGS_NS;
      if (seq >= WGS_NS)
        wgs_wait(S.bars + (xproducer ? BX_E : BW_E) + sl,
                 ((seq / WGS_NS) & 1) ^ 1, 7000000 + seq);
      uint64_t* bar = S.bars + BW_F + sl;
      const uint32_t bytes = xproducer ? npad * 512 : WBY;
      const void* src = xproducer
          ? static_cast<const void*>(xb + (size_t)st * npad * 256)
          : static_cast<const void*>(base +
              (size_t)(((S.jE[j] << OLD_UBL) + oldub) * 16 + st) * UNIT_BYTES + half);
      void* dst = xproducer ? static_cast<void*>(xring + sl * XBY)
                            : static_cast<void*>(ring + sl * WBY);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                   ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      // Same cache policy as the corresponding parent producer.
      if (xproducer) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      } else {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) L2H_ARG : "memory");
      }
    }
  }
}

// @V30_SEGMENT@

template <bool PB, bool COLD>
__global__ void __launch_bounds__(COLD ? 320 : 576, COLD ? 2 : 1)
v30_tiles(Params p, const Smem* meta) {
  constexpr int NC = COLD ? 256 : 512;
  constexpr int CW = NC / 32;
  constexpr int UBL = PB ? (COLD ? 5 : 4) : (COLD ? 6 : 5);
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  unsigned char* ring = smem_raw + V30_META;
  const int tid = threadIdx.x;
  const int nj = meta->njF + meta->njS;
  for (int j = tid; j < nj; j += blockDim.x) {
    S.jE[j] = meta->jE[j]; S.jS[j] = meta->jS[j];
    S.jC[j] = meta->jC[j]; S.jrow[j] = meta->jrow[j]; S.forder[j] = meta->forder[j];
  }
  if (tid == 0) {
    S.njF = meta->njF; S.njS = meta->njS;
    for (int k = 0; k < BX_E + WGS_NX; ++k)
      mbar_init(S.bars + k, k < BW_E ? 2 : k < BX_F ? CW : k < BX_E ? 1 : CW);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (tid >= NC) {
    v30_producer<PB, COLD>(p, S, ring, tid >= NC + 32);
  } else {
    const int njb = COLD ? S.njF : S.njS;
    const int offset = COLD ? 0 : S.njF;
    int seq = 0;
    for (int q = blockIdx.x; q < (njb << UBL); q += gridDim.x) {
      const int j = S.forder[offset + (q >> UBL)];
      const int ub = q & ((1 << UBL) - 1);
      if constexpr (COLD) {
        v30_seg<PB, true, 1>(p, S, ring, j, ub, seq);
      } else {
        const int ntt = (S.jC[j] + 7) >> 3;
        switch (ntt) {
          case 2: v30_seg<PB, false, 2>(p, S, ring, j, ub, seq); break;
          case 3: v30_seg<PB, false, 3>(p, S, ring, j, ub, seq); break;
          case 4: v30_seg<PB, false, 4>(p, S, ring, j, ub, seq); break;
          case 5: v30_seg<PB, false, 5>(p, S, ring, j, ub, seq); break;
          case 6: v30_seg<PB, false, 6>(p, S, ring, j, ub, seq); break;
          case 7: v30_seg<PB, false, 7>(p, S, ring, j, ub, seq); break;
          case 8: v30_seg<PB, false, 8>(p, S, ring, j, ub, seq); break;
        }
      }
    }
  }
  __syncthreads(); // producers and every consuming warp finish before CTA retirement
}

__global__ void __launch_bounds__(512, 1) v30_final(Params p, const Smem* meta) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x;
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + V30_META);
  for (int i = tid; i < p.M * 8; i += NT) S.pos[i] = meta->pos[i];
  CTA_SYNC();
  // @V30_FINAL@
}
