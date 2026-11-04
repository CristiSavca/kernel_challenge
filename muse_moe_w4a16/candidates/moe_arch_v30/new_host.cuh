#include <cstdio>
static void v30_resource(const char* name, const void* fn, int nt, size_t sm, int required) {
  const cudaError_t set = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaFuncAttributes a{};
  const cudaError_t attr = cudaFuncGetAttributes(&a, fn);
  int occ = -1;
  const cudaError_t cap = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, nt, sm);
  fprintf(stderr, "V30R name=%s nt=%d dyn=%zu regs=%d local=%zu static=%zu maxdyn=%d occ=%d min=%d api=%d,%d,%d\n",
          name, nt, sm, a.numRegs, a.localSizeBytes, a.sharedSizeBytes,
          a.maxDynamicSharedSizeBytes, occ, required, (int)set, (int)attr, (int)cap);
  TORCH_CHECK(set == cudaSuccess && attr == cudaSuccess && cap == cudaSuccess,
              "V30 resource query failed: ", name);
  TORCH_CHECK(a.maxDynamicSharedSizeBytes >= (int)sm && occ >= required,
              "V30 actual capacity below fixed design: ", name, " occ=", occ);
}

static void v30_launch(okc::Params p, int grid, cudaStream_t st) {
  constexpr size_t base = okc::V30_META;
  constexpr size_t cold = base + 3 * (18432 + 4096);
  constexpr size_t hot = base + 3 * (36864 + 32768);
  static_assert(cold == 89856 && hot == 231168, "V30 shared footprints");
  TORCH_CHECK(p.M >= 0 && p.M <= okc::MMAX && p.nloc > 0 && p.nloc <= 32 && grid > 0,
              "V30 input metadata outside fixed domain");
  static bool ready = false;
  if (!ready) {
    v30_resource("route", (const void*)okc::v30_route_gather, 512, base, 1);
    v30_resource("coldA", (const void*)okc::v30_tiles<false, true>, 320, cold, 2);
    v30_resource("hotA", (const void*)okc::v30_tiles<false, false>, 576, hot, 1);
    v30_resource("coldB", (const void*)okc::v30_tiles<true, true>, 320, cold, 2);
    v30_resource("hotB", (const void*)okc::v30_tiles<true, false>, 576, hot, 1);
    v30_resource("final", (const void*)okc::v30_final, 512, base + 1024, 1);
    ready = true;
  }
  // The former fixup region holds only input-derived metadata, overwritten every call.
  auto* meta = reinterpret_cast<okc::Smem*>(p.ybuf + (size_t)okc::EMAX * okc::HD);
  okc::v30_route_gather<<<grid, 512, base, st>>>(p, meta);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "V30 route launch");
  okc::v30_tiles<false, true><<<grid * 2, 320, cold, st>>>(p, meta);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "V30 coldA launch");
  okc::v30_tiles<false, false><<<grid, 576, hot, st>>>(p, meta);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "V30 hotA launch");
  okc::v30_tiles<true, true><<<grid * 2, 320, cold, st>>>(p, meta);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "V30 coldB launch");
  okc::v30_tiles<true, false><<<grid, 576, hot, st>>>(p, meta);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "V30 hotB launch");
  okc::v30_final<<<grid, 512, base + 1024, st>>>(p, meta);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "V30 final launch");
}
