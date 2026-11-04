// Isolated Q8 full-attention ownership experiment. No GPU validation yet.
// This fragment is embedded into the single-file candidate by its build script.
struct Q8RoleMetaV10 {
  float alpha[4][2][2][64];
  float offset[4][2][64];
  int moved[4][2][2][4];
  uint64_t ready[4][2][2];
  uint64_t vempty[2][2];
};
static_assert(sizeof(Q8RoleMetaV10) == 6560, "Q8 metadata layout changed");
static_assert(sizeof(Q8RoleMetaV10) <= 8192, "Q8 metadata exceeds dead prologue scratch");

template <int L, bool F> struct Q8RoleEnabledV10 {
  static constexpr bool value = L == 8 && F && OKC_SM90 && Cfg<L, F>::WS &&
      PV8ON && OKC_PV8FLAT && OKC_QSS && OKC_Q8RAW && !OKC_CONT && !OKC_DYN &&
      !OKC_PTR && !OKC_PROF && !OKC_PK2 && OKC_PCHK && OKC_LAZYMAX &&
      !OKC_RAWTHR && !OKC_EXH8 && OKC_PVSPLIT == 1 && OKC_XP_ARRALL &&
      !OKC_F2 && !OKC_XP_NOEXP && !OKC_EXPPOLY && Cfg<L, F>::STAGES == 4 &&
      Cfg<L, F>::NP == 128 && OKC_SETMAXNREG && OKC_MERGE3 && !OKC_QDIRECT;
};

__device__ __forceinline__ bool q8_role_runtime_v10(const Params& p) {
  return p.coop && p.resident && p.window == 0 && p.rope == 0;
}

// Same m64n72 accumulators as the incumbent, but P is shared rather than RS.
__device__ __forceinline__ void q8_pv_ss_v10(float* d, float* e, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "%36, %37, 1, 1, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "l"(da), "l"(db));
#endif
}

__device__ __forceinline__ uint32_t q8_poff_v10(int row, int head, int col) {
  return row * 128 + (((head * 4 + (col >> 4)) ^ (row & 7)) << 4) + (col & 15);
}

__device__ __forceinline__ void q8_role_qk_v10(const Params& p, Smem<8, true>& S,
    Q8RoleMetaV10& M, int start, int end, int seq, float ssc) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tig = lane & 3, r0 = 16 * warp + g, r1 = r0 + 8;
  const int ntiles = (end - start + 63) / 64;
  float cr[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  // The common q staging has drained; these ones never change during a segment.
#pragma unroll
  for (int h = 0; h < 2; ++h)
    *reinterpret_cast<uint2*>(S.kvq + ((tid >> 6) * 2 + h) * 4608 + 4096 + (tid & 63) * 8) =
        make_uint2(0x38383838u, 0x38383838u);
  fence_proxy_async();
  wgsync(0);
  for (int i = 0; i < ntiles; ++i) {
    const int st = i & 3;
    mbar_wait_full(&S.full[st], (i >> 2) & 1);
    // New rows are patched only by this128-thread worker. The PV worker has
    // not been given ready credits for this tile, and the producer still owns
    // no empty credit, so both heads' K/V bytes are exclusive here.
    const int t0 = start + 64 * i, lo = max(t0, seq), hi = min(t0 + 64, end);
    if (lo < hi) {
      for (int e = tid; e < (hi - lo) * 8; e += 128) {
        const int pos = lo + (e >> 3), ch = e & 7, row = pos - t0;
        const int so = row * 128 + ((ch ^ (row & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + so) =
            *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * 128 + ch * 16);
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + so) =
            *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * 128 + ch * 16);
      }
      fence_proxy_async();
      wgsync(0);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (i >= 2) mbar_wait_full(&M.vempty[h][i & 1], ((i - 2) >> 1) & 1);
      float s[32];
      const uint64_t qd = desc_sw64(smem_u32(S.q8s) + h * 4096);
      const uint64_t kd = desc_sw128(smem_u32(S.k + st * STAGE_BYTES) + h * 64);
      wgmma_fence();
      wgmma_m64n64k32_e4m3_ss(s, qd, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd + 2, kd + 2, 1);
      wgmma_commit();
      // Unchanged FP8 V transpose/permuted-key layout, performed by QK while
      // its tensor work is in flight. Its own two-slot lifetime is independent
      // of the four-slot K/P ring.
      uint8_t* vt = S.kvq + ((i & 1) * 2 + h) * 4608;
      const int vw = lane >> 2, vq = lane & 3, vr0 = vw >= 4;
      const uint32_t sel0 = vw >= 4 ? 0x7531u : 0x6420u, sel1 = vw >= 4 ? 0x6420u : 0x7531u;
      const uint32_t vo0 = 128 * vw + ((warp ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
      const uint32_t vo1 = 128 * vw + ((warp ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
      const int vk = 16 * warp + 8 * ((lane >> 3) & 1) + (lane & 7);
      const uint32_t vl0 = vk * 128 + (((4 * h + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t x0, x1, x2, x3;
        ldsm_x4_t(smem_u32(S.v + st * STAGE_BYTES) + (u ? vl0 ^ 32u : vl0), x0, x1, x2, x3);
#pragma unroll
        for (int d = 0; d < 2; ++d) {
          const uint32_t a = d ? x2 : x0, b = d ? x3 : x1;
          *reinterpret_cast<uint32_t*>(vt + vo0 + 1024 * (2 * u + d)) = prmt(a, b, sel0);
          *reinterpret_cast<uint32_t*>(vt + vo1 + 1024 * (2 * u + d)) = prmt(a, b, sel1);
        }
      }
      wgmma_wait<0>();
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(s[e]);
      const bool interior = t0 >= start && t0 + 63 <= min(end - 1, seq);
      if (!interior) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int vhi = min(end - 1, seq + 2 * warp + hf);
#pragma unroll
          for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int pos = t0 + 8 * n + 2 * tig + e;
              if (pos < start || pos > vhi) s[4 * n + 2 * hf + e] = -INFINITY;
            }
        }
      }
      if (i == 0) {
        float ml = -INFINITY, mh = -INFINITY;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          ml = fmaxf(ml, fmaxf(s[4 * n], s[4 * n + 1]));
          mh = fmaxf(mh, fmaxf(s[4 * n + 2], s[4 * n + 3]));
        }
        // Match incumbent ordering: scale partial maxima, then quad reduce.
        ml = quad_max(ml * ssc); mh = quad_max(mh * ssc);
        cr[h][0] = ml == -INFINITY ? 0.f : ml;
        cr[h][1] = mh == -INFINITY ? 0.f : mh;
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float* sv = s + 16 * half;
        float al = 1.f, ah = 1.f;
        bool moved = false, check = i != 0;
        uint32_t pa[4];
#pragma unroll 1
        for (;;) {
          const float dl = cr[h][0], dh = cr[h][1];
          pa[0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dl)), ex2(fmaf(sv[1], ssc, -dl)), ex2(fmaf(sv[4], ssc, -dl)), ex2(fmaf(sv[5], ssc, -dl)));
          pa[1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dh)), ex2(fmaf(sv[3], ssc, -dh)), ex2(fmaf(sv[6], ssc, -dh)), ex2(fmaf(sv[7], ssc, -dh)));
          pa[2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dl)), ex2(fmaf(sv[9], ssc, -dl)), ex2(fmaf(sv[12], ssc, -dl)), ex2(fmaf(sv[13], ssc, -dl)));
          pa[3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dh)), ex2(fmaf(sv[11], ssc, -dh)), ex2(fmaf(sv[14], ssc, -dh)), ex2(fmaf(sv[15], ssc, -dh)));
          if (!check) break;
          constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
          const uint32_t ov = ((pa[0] + PB) | (pa[1] + PB) | (pa[2] + PB) | (pa[3] + PB)) & 0x80808080u;
          if (!__any_sync(0xffffffffu, ov != 0u)) break;
          float ml = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
          float mh = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
          ml = quad_max(ml) * ssc; mh = quad_max(mh) * ssc;
          const float nl = fmaxf(ml, cr[h][0]), nh = fmaxf(mh, cr[h][1]);
          al = ex2(cr[h][0] - nl); ah = ex2(cr[h][1] - nh);
          cr[h][0] = nl; cr[h][1] = nh;
          moved = true; check = false;
        }
#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
          const int row = (reg & 1) ? r1 : r0;
          const int col = 32 * half + 4 * tig + 16 * (reg >> 1);
          *reinterpret_cast<uint32_t*>(S.k + st * STAGE_BYTES + q8_poff_v10(row, h, col)) = pa[reg];
        }
        if (tig == 0) {
          M.alpha[st][h][half][r0] = al; M.alpha[st][h][half][r1] = ah;
          if (half == 1) { M.offset[st][h][r0] = cr[h][0]; M.offset[st][h][r1] = cr[h][1]; }
        }
        if (lane == 0) M.moved[st][h][half][warp] = moved;
        // All128 release arrivals are required. PV's acquire wait follows
        // every store and every generic->async proxy fence, including V^T.
        fence_proxy_async();
        mbar_arrive(&M.ready[st][h][half]);
      }
    }
    // Half of the existing256-thread empty-credit count. The other128 may
    // arrive only after PV has completed both heads, not merely loaded P.
    mbar_arrive(&S.empty[st]);
  }
}

__device__ __forceinline__ void q8_role_pv_v10(const Params& p, Smem<8, true>& S,
    Q8RoleMetaV10& M, int b, int pair, int copy, int ntiles) {
  const int wt = threadIdx.x & 127, lane = wt & 31, warp = wt >> 5;
  const int g = lane >> 2, tig = lane & 3, r0 = 16 * warp + g, r1 = r0 + 8;
  float o[2][32] = {}, l[2][4] = {}, final_offset[2][2] = {};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(o[h][e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(l[h][e]);
  }
  for (int i = 0; i < ntiles; ++i) {
    const int st = i & 3;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        mbar_wait_full(&M.ready[st][h][half], (i >> 2) & 1);
        const int moved = M.moved[st][h][half][0] | M.moved[st][h][half][1] |
                          M.moved[st][h][half][2] | M.moved[st][h][half][3];
        if (moved) {
          // PK2=0 semantics: a half1 offset move rescales half0's already
          // accumulated contribution as well as every older tile.
          if (half == 1) wgmma_wait<0>();
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(o[h][e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(l[h][e]);
          const float al = M.alpha[st][h][half][r0], ah = M.alpha[st][h][half][r1];
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            o[h][4 * t] *= al; o[h][4 * t + 1] *= al;
            o[h][4 * t + 2] *= ah; o[h][4 * t + 3] *= ah;
          }
          l[h][0] *= al; l[h][1] *= al; l[h][2] *= ah; l[h][3] *= ah;
        }
        if (half == 1) {
          final_offset[h][0] = M.offset[st][h][r0];
          final_offset[h][1] = M.offset[st][h][r1];
        }
        const uint64_t pd = desc_sw128(smem_u32(S.k + st * STAGE_BYTES) + h * 64);
        const uint64_t vd = desc_sw64(smem_u32(S.kvq + ((i & 1) * 2 + h) * 4608));
        wgmma_fence();
        q8_pv_ss_v10(o[h], l[h], pd + 2 * half, vd + 2 * half);
        wgmma_commit();
      }
    }
    wgmma_wait<0>();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(o[h][e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(l[h][e]);
      mbar_arrive(&M.vempty[h][i & 1]);
    }
    mbar_arrive(&S.empty[st]);
  }
  // Same FP32 partial reduction as the parent; this WG uses virtual head
  // indices in turn rather than reading another WG's register accumulators.
  const int bp = b * (p.Hkv >> 1) + pair;
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int j = 2 * warp + hf, hh = h * 8 + g;
      const float ls = quad_sum(0.25f * l[h][2 * hf]);
      const float f = ex2(fminf(final_offset[h][hf], 126.f));
      float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * 8 + j) * 16 + hh) * 64;
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        const float a0 = o[h][4 * t + 2 * hf] * f, a1 = o[h][4 * t + 2 * hf + 1] * f;
        const float b0 = o[h][4 * (t + 1) + 2 * hf] * f, b1 = o[h][4 * (t + 1) + 2 * hf + 1] * f;
        const float x0 = __shfl_xor_sync(0xffffffffu, odd ? a0 : b0, 1);
        const float x1 = __shfl_xor_sync(0xffffffffu, odd ? a1 : b1, 1);
        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1),
               odd ? x0 : a0, odd ? x1 : a1, odd ? b0 : x0, odd ? b1 : x1);
      }
      if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * 8 + j) * 16 + hh, ls * f);
    }
}

__device__ __forceinline__ void q8_role_body_v10(const Params& p, Smem<8, true>& S,
    int b, int pair, int start, int end, int seq, int copy, float ssc) {
  const int tid = threadIdx.x, wt = tid & 127, lane = tid & 31;
  const int head = tid >> 7, warp = wt >> 5, g = lane >> 2, tig = lane & 3;
  const int r0 = 16 * warp + g, r1 = r0 + 8;
  // Unchanged FP16->FP8 Q staging from the parent's WS path. All256 threads
  // participate before the old staging and raw-K/V union are reused.
#pragma unroll
  for (int ks = 0; ks < 2; ++ks)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int kb = 32 * ks + 16 * hf + 4 * tig;
      const uint2 q0 = *reinterpret_cast<const uint2*>(S.kvq + ((r0 >> 3) * 16 + head * 8 + (r0 & 7)) * 128 + 2 * kb);
      const uint2 q1 = *reinterpret_cast<const uint2*>(S.kvq + ((r1 >> 3) * 16 + head * 8 + (r1 & 7)) * 128 + 2 * kb);
      *reinterpret_cast<uint32_t*>(S.q8s + head * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) = pack_e4m3x4(q0.x, q0.y);
      *reinterpret_cast<uint32_t*>(S.q8s + head * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) = pack_e4m3x4(q1.x, q1.y);
    }
  fence_proxy_async();
  csync<true, 256>();
  // This phase starts only after common normalization has consumed kvst;
  // gst is cluster-only, and this path is full/flat/cooperative only.
  Q8RoleMetaV10& M = *reinterpret_cast<Q8RoleMetaV10*>(S.q8_role_scratch_v10);
  if (tid == 0) {
    for (int st = 0; st < 4; ++st)
      for (int h = 0; h < 2; ++h)
        for (int half = 0; half < 2; ++half) mbar_init_n(&M.ready[st][h][half], 128);
    for (int h = 0; h < 2; ++h)
      for (int slot = 0; slot < 2; ++slot) mbar_init_n(&M.vempty[h][slot], 128);
    mbar_fence_init();
  }
  csync<true, 256>();
  if (tid < 128) q8_role_qk_v10(p, S, M, start, end, seq, ssc);
  else q8_role_pv_v10(p, S, M, b, pair, copy, (end - start + 63) / 64);
  // Publishes PV's partials before the unchanged cooperative finalize, and
  // drains all P/V readers before raw scratch or barrier phases can restart.
  csync<true, 256>();
  pdl_launch_dependents();
  __syncthreads();  // matches the original producer segment's final barrier
}
