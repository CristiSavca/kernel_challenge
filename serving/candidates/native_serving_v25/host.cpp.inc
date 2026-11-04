// V25: cooperative launch is required by the persistent grid's global barrier.
// No ordinary-launch fallback and no mutable PDL state.
static void v25_cuda(cudaError_t status, const char* what) {
  TORCH_CHECK(status == cudaSuccess, what, ": ", cudaGetErrorString(status));
}
int64_t okc_init() {
  int device = 0, cooperative = 0, sms = 0;
  v25_cuda(cudaGetDevice(&device), "get device");
  v25_cuda(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device), "cooperative support");
  v25_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SM count");
  TORCH_CHECK(cooperative && sms > 0 && sms < okc::WGS_SLOTS, "unsupported cooperative grid");
  const size_t sm = okc_smem_bytes();
  for (int big = 0; big < 2; ++big) {
    auto fn = big ? okc::native_serving_v25<true> : okc::native_serving_v25<false>;
    v25_cuda(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm), "dynamic shared attribute");
    cudaFuncAttributes a{};
    v25_cuda(cudaFuncGetAttributes(&a, fn), "kernel attributes");
    int resident = 0;
    v25_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, fn, okc::NTL, sm), "kernel occupancy");
    TORCH_CHECK(resident >= 1, "cooperative kernel cannot occupy a block per SM");
    printf("V25R big=%d threads=%d dyn=%zu regs=%d local=%zu static=%zu occ=%d grid=%d cooperative=%d\n",
           big, okc::NTL, sm, a.numRegs, a.localSizeBytes, a.sharedSizeBytes, resident, sms, cooperative);
  }
  return sms;
}
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor expert_out, torch::Tensor local_out, int64_t lo, int64_t nloc, int64_t grid) {
  TORCH_CHECK(w.size() == 5 && scratch.size() == 3, "V25 expects one bank and one explicit workspace");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == okc::HD,
              "V25 activation domain is contiguous CUDA BF16 [M,4096]");
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 0 && M <= okc::MMAX && nloc > 0 && nloc <= 32 && lo >= 0 && lo + nloc <= 256, "V25 geometry");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == M && ids.size(1) == 8 &&
              (((uintptr_t)ids.data_ptr() & 15u) == 0), "V25 aligned int32 global IDs");
  TORCH_CHECK(expert_out.scalar_type() == at::kBFloat16 && expert_out.is_contiguous() && expert_out.numel() == (int64_t)M * 8 * okc::HD,
              "V25 BF16 expert output shape");
  TORCH_CHECK(local_out.scalar_type() == at::kInt && local_out.is_contiguous() && local_out.numel() == (int64_t)M * 8,
              "V25 int32 local IDs");
  for (auto& t : w) TORCH_CHECK(t.is_contiguous() && t.device() == x.device(), "V25 weight device/contiguity");
  for (auto& t : scratch) TORCH_CHECK(t.is_contiguous() && t.device() == x.device(), "V25 scratch device/contiguity");
  TORCH_CHECK(ids.device() == x.device() && expert_out.device() == x.device() && local_out.device() == x.device(), "V25 output device");
  TORCH_CHECK(w[0].scalar_type() == at::kByte && w[1].scalar_type() == at::kByte &&
              w[0].numel() == nloc * okc::UB_A * 16 * okc::UNIT_BYTES && w[1].numel() == nloc * okc::UB_B * 16 * okc::UNIT_BYTES,
              "V25 packed WGS bank extent");
  TORCH_CHECK(w[2].scalar_type() == at::kBFloat16 && w[3].scalar_type() == at::kBFloat16 &&
              w[2].numel() == nloc && w[3].numel() == nloc && w[4].scalar_type() == at::kInt && w[4].numel() == 256,
              "V25 prepared BF16 globals / scale LUT");
  TORCH_CHECK(scratch[0].scalar_type() == at::kInt && scratch[0].numel() >= 8 + 1024 &&
              scratch[1].scalar_type() == at::kFloat && scratch[1].numel() >= (int64_t)okc::WGS_FIXF &&
              scratch[2].scalar_type() == at::kBFloat16 && scratch[2].numel() >= (int64_t)2 * okc::WGS_XROWS * okc::HD,
              "V25 workspace extent");
  if (M == 0) return;
  int device = 0, cooperative = 0, sms = 0;
  v25_cuda(cudaGetDevice(&device), "get device");
  TORCH_CHECK(device == x.get_device(), "V25 current device differs from tensors");
  v25_cuda(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device), "cooperative support");
  v25_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SM count");
  auto fn = M >= OKC_BIGM ? okc::native_serving_v25<true> : okc::native_serving_v25<false>;
  const size_t sm = okc_smem_bytes();
  int resident = 0;
  v25_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, fn, okc::NTL, sm), "selected kernel occupancy");
  TORCH_CHECK(cooperative && resident >= 1 && grid > 0 && grid < okc::WGS_SLOTS && grid <= sms * resident,
              "V25 cooperative admission failed");
  okc::Params p{};
  p.wAs = (const unsigned char*)w[0].data_ptr(); p.wBs = (const unsigned char*)w[1].data_ptr();
  p.marlin_gsA = (const __nv_bfloat16*)w[2].data_ptr(); p.marlin_gsB = (const __nv_bfloat16*)w[3].data_ptr();
  p.lut = (const uint32_t*)w[4].data_ptr();
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr();
  p.M = M; p.lo = (int)lo; p.nloc = (int)nloc;
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8; p.xready = ib + 8 + 512;
  p.fixbuf = (float*)scratch[1].data_ptr();
  p.doneAs = reinterpret_cast<int*>(p.fixbuf + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
  p.xA = (__nv_bfloat16*)scratch[2].data_ptr();
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.out = (__nv_bfloat16*)expert_out.data_ptr(); p.local_out = (int*)local_out.data_ptr();
  void* args[] = {&p};
  v25_cuda(cudaLaunchCooperativeKernel((const void*)fn, dim3((unsigned)grid), dim3(okc::NTL), args, sm,
                                      at::cuda::getCurrentCUDAStream()), "V25 cooperative launch");
}
