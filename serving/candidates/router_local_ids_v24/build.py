"""Build the isolated, stock-based V24 patch; never modify serving/sglang."""
from pathlib import Path
import hashlib, json, subprocess, difflib
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REV = 'a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f'
FILES = [
 'python/sglang/kernels/jit/csrc/muse_spark_v12/routing.cuh',
 'python/sglang/kernels/ops/muse_spark_v12/routing.py',
 'python/sglang/kernels/ops/muse_spark_v12/fused_muse_spark_routing.py',
 'python/sglang/srt/layers/muse_spark_fp4_moe.py',
 'python/sglang/srt/models/muse_spark.py',
]
EDITS = {}
def replace(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new)
def edit(path, old, new):
    EDITS.setdefault(path, []).append((old,new))
C,P,F,L,M=FILES
edit(C, '  int num_tokens;\n', '''  // V24: independent Marlin output; topk_ids keeps its original contract.
  const int32_t* __restrict__ marlin_expert_map;
  int32_t* __restrict__ marlin_topk_ids;
  int64_t stride_mm;
  int num_tokens;
''')
edit(C, '      params.topk_ids[token * params.stride_im + lane] = -1;\n', '''      params.topk_ids[token * params.stride_im + lane] = -1;
      if (params.marlin_topk_ids != nullptr) {
        params.marlin_topk_ids[token * params.stride_mm + lane] = -1;
      }
''')
edit(C, '    params.topk_weights[token * params.stride_wm + lane] = weight;\n', '''    params.topk_weights[token * params.stride_wm + lane] = weight;
    if (params.marlin_topk_ids != nullptr) {
      params.marlin_topk_ids[token * params.stride_mm + lane] = params.marlin_expert_map[out_id];
    }
''')
edit(C, '    double division_eps) {\n', '''    double division_eps,
    tvm::ffi::Optional<tvm::ffi::TensorView> marlin_expert_map,
    tvm::ffi::Optional<tvm::ffi::TensorView> marlin_topk_ids) {
''')
edit(C, '      .num_tokens = num_tokens,\n', '''      .marlin_expert_map = nullptr,
      .marlin_topk_ids = nullptr,
      .stride_mm = 0,
      .num_tokens = num_tokens,
''')
edit(C, '  if (num_tokens == 0) return;\n', '''  RuntimeCheck(marlin_expert_map.has_value() == marlin_topk_ids.has_value(),
               "Marlin map and output must be supplied together");
  if (marlin_topk_ids.has_value()) {
    const auto& map = marlin_expert_map.value();
    const auto& ids = marlin_topk_ids.value();
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device_).verify(map);
    TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device_).verify(ids);
    RuntimeCheck(map.stride(0) == 1, "Marlin map must be contiguous");
    RuntimeCheck(ids.stride(1) == 1 && ids.stride(0) == kTopK, "Marlin output must be contiguous");
    // Reject overlapping storage, including offset views, before any write.
    const auto lo = reinterpret_cast<uintptr_t>(ids.data_ptr());
    const auto hi = lo + static_cast<uint64_t>(num_tokens) * kTopK * sizeof(int32_t);
    const auto disjoint = [&](tvm::ffi::TensorView x, uint64_t elements) {
      const auto begin = reinterpret_cast<uintptr_t>(x.data_ptr());
      const auto end = begin + elements * sizeof(int32_t);
      return lo == hi || begin == end || hi <= begin || end <= lo;
    };
    const auto rows_span = [&](tvm::ffi::TensorView x, int cols) -> uint64_t {
      return num_tokens == 0 ? 0 : static_cast<uint64_t>(num_tokens - 1) * x.stride(0) + cols;
    };
    RuntimeCheck(disjoint(gating_output, rows_span(gating_output, num_experts)) &&
                 disjoint(bias, num_experts) && disjoint(map, num_experts) &&
                 disjoint(topk_weights, rows_span(topk_weights, kTopK)) &&
                 disjoint(topk_ids, rows_span(topk_ids, kTopK)), "Marlin output aliases router storage");
    if (packed_topk_ids.has_value())
      RuntimeCheck(disjoint(packed_topk_ids.value(), rows_span(packed_topk_ids.value(), kTopK)),
                   "Marlin output aliases packed output");
    if (local_expert_mapping.has_value())
      RuntimeCheck(disjoint(local_expert_mapping.value(), num_experts), "Marlin output aliases old map");
    if (num_valid_tokens.has_value())
      RuntimeCheck(disjoint(num_valid_tokens.value(), 1), "Marlin output aliases valid-token count");
    const auto map_lo = reinterpret_cast<uintptr_t>(map.data_ptr());
    const auto map_hi = map_lo + static_cast<uint64_t>(num_experts) * sizeof(int32_t);
    const auto map_disjoint = [&](tvm::ffi::TensorView x, uint64_t elements) {
      const auto begin = reinterpret_cast<uintptr_t>(x.data_ptr());
      const auto end = begin + elements * sizeof(int32_t);
      return map_lo == map_hi || begin == end || map_hi <= begin || end <= map_lo;
    };
    RuntimeCheck(map_disjoint(topk_weights, rows_span(topk_weights, kTopK)) &&
                 map_disjoint(topk_ids, rows_span(topk_ids, kTopK)), "Marlin map aliases writable routing output");
    if (packed_topk_ids.has_value())
      RuntimeCheck(map_disjoint(packed_topk_ids.value(), rows_span(packed_topk_ids.value(), kTopK)),
                   "Marlin map aliases packed output");
    params.marlin_expert_map = static_cast<const int32_t*>(map.data_ptr());
    params.marlin_topk_ids = static_cast<int32_t*>(ids.data_ptr());
    params.stride_mm = ids.stride(0);
  }

  if (num_tokens == 0) return;
''')
edit(P, '    packed_topk_ids: Optional[torch.Tensor] = None,  # [num_tokens, topk] int32\n', '''    packed_topk_ids: Optional[torch.Tensor] = None,  # [num_tokens, topk] int32
    marlin_expert_map: Optional[torch.Tensor] = None,  # separate global->local map
    marlin_topk_ids: Optional[torch.Tensor] = None,  # separate [num_tokens, topk] out
''')
edit(P, '    module = _jit_module(\n', '''    assert (marlin_expert_map is None) == (marlin_topk_ids is None)
    # The new output is exact-sized; no implicit slicing or aliasing with IDs.
    if marlin_topk_ids is not None:
        assert marlin_topk_ids.shape == (num_tokens, topk)
    module = _jit_module(
''')
edit(P, '''            router_score_division_eps if router_score_division_eps is not None else 0.0
        ),
    )
''', '''            router_score_division_eps if router_score_division_eps is not None else 0.0
        ),
        marlin_expert_map,
        marlin_topk_ids,
    )
''')
edit(F, 'from sglang.kernels.fused_op import BaseFusedOp\n', 'from sglang.kernels.fused_op import BaseFusedOp, get_fused_op_backend\n')
# All old entry signatures/returns remain default-compatible. The opt-in wrapper
# refuses unknown dispatch targets before passing the extra internal keyword.
entry='''    def forward_with_local_ids(
        self, hidden_states, gating_output, topk, renormalize, marlin_expert_map
    ):
        """Return (scores, global IDs, packed or None, Marlin IDs or None).

        Only actual JIT dispatch fills the fourth result. Unsupported state or
        dispatch uses the original packed call; callers must retain its mapping.
        No backend is forced and no result validity is inferred from priority.
        """
        fallback = lambda: (*self(
            hidden_states, gating_output, topk, renormalize, want_packed=True
        ), None)
        if (
            type(self) is not FusedMuseSparkRouting
            or self.is_torch_compile
            or self.local_expert_mapping is not None
            or self.num_valid_tokens is not None
            or self.sol_topk_out is not None
            or marlin_expert_map is None
            or marlin_expert_map.dtype != torch.int32
            or marlin_expert_map.device != gating_output.device
            or not marlin_expert_map.is_cuda
            or marlin_expert_map.shape != (gating_output.shape[-1],)
            or not marlin_expert_map.is_contiguous()
        ):
            return fallback()
        forced = get_fused_op_backend()
        if forced not in (None, KernelBackend.JIT, KernelBackend.TRITON, KernelBackend.TORCH):
            return fallback()
        if forced is None:
            method = self._forward_method
            if method is None:
                method = self._resolve_forward_method()
            known = (
                BaseFusedOp._forward_backend_dynamic,
                FusedMuseSparkRouting.forward_jit,
                FusedMuseSparkRouting.forward_triton,
                FusedMuseSparkRouting.forward_native,
                FusedMuseSparkRouting.forward_cpu,
            )
            if getattr(method, "__func__", None) not in known:
                return fallback()
        return self(
            hidden_states, gating_output, topk, renormalize, want_packed=True,
            marlin_expert_map=marlin_expert_map,
        )

'''
edit(F, '    def backend_eligible(\n', entry+'    def backend_eligible(\n')
# Replace only the class methods' signature lines, not unrelated functions.
src=subprocess.check_output(['git','-C',str(ROOT/'serving/sglang'),'show',f'{REV}:{F}']).decode()
for name in ('backend_eligible','forward_jit','forward_native','forward_cpu','forward_triton'):
    start=src.index('    def '+name+'(',src.index('class FusedMuseSparkRouting'))
    end=src.index('    ):',start) if name!='backend_eligible' else src.index('    ) -> bool:',start)
    sig=src[start:end]
    edit(F,sig,sig.replace('        want_packed: bool = False,\n','        want_packed: bool = False,\n        marlin_expert_map: Optional[torch.Tensor] = None,\n'))
for name,doc in [('forward_native','PyTorch reference implementation.'),('forward_cpu','sgl-kernel sigmoid + biased top-k; the tail stays shared.'),('forward_triton','Fused Triton kernel implementation.')]:
    # Place fallback at start after each full docstring; default body untouched.
    start=src.index('    def '+name+'(',src.index('class FusedMuseSparkRouting'))
    a=src.index('        """',start);b=src.index('"""',a+11)+3
    old=src[a:b]
    extra=f'''
        if marlin_expert_map is not None:
            return (*self.{name}(
                hidden_states, gating_output, topk, renormalize, want_packed=True
            ), None)
'''
    edit(F,old,old+extra.rstrip('\n'))
edit(F, '        """Fused JIT CUDA kernel implementation."""\n', '''        """Fused JIT CUDA kernel implementation."""
        if marlin_expert_map is not None:
            assert self.local_expert_mapping is None and self.num_valid_tokens is None
            assert self.sol_topk_out is None
''')
edit(F, '            if want_packed\n            else None\n        )\n        topk_weights, topk_ids, packed = _jit_fused_muse_spark_routing(\n', '''            if want_packed and marlin_expert_map is None
            else None
        )
        marlin_ids = (
            torch.empty((gating.shape[0], topk), dtype=torch.int32, device=gating.device)
            if marlin_expert_map is not None else None
        )
        topk_weights, topk_ids, packed = _jit_fused_muse_spark_routing(
''')
edit(F, '            packed_topk_ids=packed,\n        )\n        if want_packed:\n', '''            packed_topk_ids=packed,
            marlin_expert_map=marlin_expert_map,
            marlin_topk_ids=marlin_ids,
        )
        if marlin_ids is not None:
            return topk_weights, topk_ids, packed, marlin_ids
        if want_packed:
''')
edit(L, '''    def _marlin_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
''', '''    def _marlin_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        local_topk_ids: Optional[torch.Tensor] = None,
''')
edit(L, '''        local_topk_ids = topk_ids.to(torch.int32)
        if expert_map is not None:
            local_topk_ids = expert_map[local_topk_ids]
        local_topk_ids = local_topk_ids.contiguous()
''', '''        if local_topk_ids is None:
            local_topk_ids = topk_ids.to(torch.int32)
            if expert_map is not None:
                local_topk_ids = expert_map[local_topk_ids]
            local_topk_ids = local_topk_ids.contiguous()
        elif (
            local_topk_ids.dtype != torch.int32
            or local_topk_ids.device != topk_ids.device
            or local_topk_ids.shape != topk_ids.shape
            or not local_topk_ids.is_contiguous()
        ):
            raise ValueError("Explicit Marlin local IDs must match the contiguous int32 routing output")
''')
start='''    def _forward_marlin(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: Optional[nn.Module],
        pre_expert_norm: Optional[nn.Module] = None,
'''
edit(L,start,start+'        local_topk_ids: Optional[torch.Tensor] = None,\n')
edit(L, '''        expert_outputs, local_topk_ids = self._marlin_expert_outputs(
            hidden_states, topk_scores, topk_ids
        )
''', '''        expert_outputs, local_topk_ids = self._marlin_expert_outputs(
            hidden_states, topk_scores, topk_ids, local_topk_ids=local_topk_ids
        )
''')
edit(L, '''        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> torch.Tensor:
        if self.use_marlin:
''', '''        *,
        ar_workspace: FinalizePushWorkspace | None = None,
        local_topk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if local_topk_ids is not None and not self.use_marlin:
            raise ValueError("Explicit local IDs are only supported by Marlin")
        if self.use_marlin:
''')
edit(L, '''                post_expert_norm,
                pre_expert_norm=pre_expert_norm,
            )

        from flashinfer.fused_moe''', '''                post_expert_norm,
                pre_expert_norm=pre_expert_norm,
                local_topk_ids=local_topk_ids,
            )

        from flashinfer.fused_moe''')
edit(M, '''        with (
            torch.cuda.stream(side_stream)
            if use_side_stream
            else contextlib.nullcontext()
        ):
            router_logits = (
''', '''        # Read the current EP map per invocation, never cache it on the router.
        marlin_expert_map = (
            self.experts._expert_map
            if self.nvfp4_checkpoint and not sp_active and getattr(self.experts, "use_marlin", False)
            else None
        )
        local_topk_ids = None
        with (
            torch.cuda.stream(side_stream)
            if use_side_stream
            else contextlib.nullcontext()
        ):
            router_logits = (
''')
edit(M, '''            topk_scores, topk_ids, packed_topk_ids = self._routing_op()(
                None, router_logits, self.top_k, True, want_packed=True
            )
''', '''            if marlin_expert_map is not None:
                topk_scores, topk_ids, packed_topk_ids, local_topk_ids = self._routing_op().forward_with_local_ids(
                    None, router_logits, self.top_k, True, marlin_expert_map
                )
            else:
                topk_scores, topk_ids, packed_topk_ids = self._routing_op()(
                    None, router_logits, self.top_k, True, want_packed=True
                )
''')
edit(M, '                else (topk_scores, topk_ids, packed_topk_ids)\n', '                else (topk_scores, topk_ids, packed_topk_ids, local_topk_ids)\n')
edit(M, '''                    tensor.record_stream(main_stream)

        if (capturer := get_global_experts_capturer()) is not None:
''', '''                    tensor.record_stream(main_stream)
            if local_topk_ids is not None:
                marlin_expert_map.record_stream(side_stream)

        if (capturer := get_global_experts_capturer()) is not None:
''')
edit(M, '''                prepared_input=prepared_input,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
''', '''                prepared_input=prepared_input,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
                local_topk_ids=local_topk_ids,
''')
def build():
    manifest={'parent_revision':REV,'scope':'stock-only serving dual-ID router; no finalizer/alignment changes','files':[]}
    diff=[]
    for path in FILES:
        stock=subprocess.check_output(['git','-C',str(ROOT/'serving/sglang'),'show',f'{REV}:{path}'])
        text=stock.decode();new=text
        for old,repl in EDITS[path]:new=replace(new,old,repl)
        for role,data in [('baseline',stock),('candidate',new.encode())]:
            target=OUT/role/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        restored=new
        for old,repl in reversed(EDITS[path]):restored=replace(restored,repl,old)
        assert restored==text
        manifest['files'].append({'path':path,'baseline_sha256':hashlib.sha256(stock).hexdigest(),'candidate_sha256':hashlib.sha256(new.encode()).hexdigest()})
        diff.extend(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
    patch=''.join(diff).encode();(OUT/'router_local_ids_v24.patch').write_bytes(patch)
    license_bytes=subprocess.check_output(['git','-C',str(ROOT/'serving/sglang'),'show',f'{REV}:LICENSE'])
    (OUT/'LICENSE').write_bytes(license_bytes)
    manifest.update(patch_sha256=hashlib.sha256(patch).hexdigest(),patch_bytes=len(patch),added_lines=sum(x.startswith('+') and not x.startswith('+++') for x in diff),license_sha256=hashlib.sha256(license_bytes).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':build()
