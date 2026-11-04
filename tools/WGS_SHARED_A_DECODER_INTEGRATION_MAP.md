# Conditional shared-A → decoder integration map

November4,2026. Read-only source/profile audit; no integration candidate, GPU
allocation or official call. Integration remains conditional on a useful
standalone result under the fixed shared-A gates.

Pinned sources: standalone parent247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0,
shared-A93c2d0fbfe1d37792389402b335d34f8e00328e84ce25ad8ea732b9726d08a38,
decoder97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2.
Machine-readable evidence: `experiments/iterations/20261104-wgs-shared-a-decoder-integration-map.json`.

## Minimal computational port

Decoder `MOE_CUDA_SRC` at5398 equals the standalone parent's `CUDA_SRC` byte for
byte after removing one trailing historical comment on `#define OKC_BIGM 32`.
`MOE_CPP_SRC` equals standalone `CPP_SRC` exactly. There is no different decoder
MoE algorithm to merge and no C++ export/pointer ABI change.

Only two existing AST nodes need change:

1. `MOE_CUDA_SRC`: apply the exact reviewed shared-A CUDA delta while preserving
   the decoder's BIGM comment convention. This includes both shared-ring depths,
   derived barrier indices, SS wrappers, exact pair stores/publication barriers,
   dynamic-shared calculation and the same bounded host resource checks.
2. `_ma_ext` at8991: give the changed extension a unique cache prefix. Preserve
   decoder flag construction, dispatch, `_MA_EXTS` keys and PDL initialization.

No Python packing, shape policy, weight pointer, external workspace, tuning
policy, dense/attention source or UV helper needs change. The parent and shared-A
standalone retain identical packing helpers. Therefore the existing decoder
`_POS_PACK` convention requires no new adaptation: `MOE_MULDEC=True` selects
`_stored_pos_mul`, and decoder `_wgs_copy(codes,sf,rows)` already supplies that
layout through `_pack_codes_nat`. Do not transplant the standalone optional
`positions` API or its dual tiny/larger state.

## Flags, pointers and scratch

The MAIN flag tuple is identical: NW16, FAST1=0, EPI4=0, WGS=1. With eligible
weight scales, both MAIN builds add PRMTSC1, MULDEC1 and DQ7; H200 defaults are
NS3/FNST4/NSF3. These generic defaults do not override the WGS-only ring depths.
The sole build-option difference is informational ptxas verbosity: standalone
uses `-v,-O3`; decoder uses `-O3`. Preserve decoder optimization flags. Adding
WGS-only verbosity is optional evidence collection requiring explicit review,
not a computational port requirement or proof of unchanged binary allocation.

Tiny controls differ and must remain decoder-specific: decoder TINY uses
MULDEC1 and omits FINPF1; standalone TINY uses MULDEC0 and FINPF1. TINY2 and MID
flag tuples match. Keep `_moe(T)` thresholds4/8/16 and the weight-only PRMT
eligibility/fallback. The shared-A change affects MAIN only when selected.

`okc_run` signatures and all Params pointer offsets are unchanged. Decoder
passes already-local expert IDs0..31/-1, lo0/count32, BF16 `xm[T,4096]`, FP32
route weights, BF16 expert output and the existing SM-count grid. WGS weights
remain `ma_w[12:14]`; norm naming is decoder `post_expert_norm`, not standalone
`post_norm_w`, so preserve `_ma_prepare` wholesale.

`_ma_make_scratch`, guard sentinels, `WGS_FIXF`, `WGS_XROWS`, xA/xB, doneAs and
stream-K fixups remain unchanged. Both Python allocations already overprovision
the FT64 X-row count consistently. The only storage change is CTA-local dynamic
shared231168→194304; the32768-byte A region is not a global scratch addition.

## PDL, UV and activation evidence

Decoder `_after_attention` sets PDL per call: T<=32 enabled, larger disabled.
Thus affected T32 exercises a PDL chain absent from the standalone benchmark.
Preserve the exact inherited initial grid dependency wait, completion trigger,
finalizer and host programmatic-launch branch. Additional WG-local A barriers
must not alter any producer/consumer grid protocol. A standalone gain cannot
validate the complete decoder's overlap or downstream numerical behavior.

UV can send2T physical u/v rows through dense pooler/router kernels, but
`_uv_pooler` returns pooled/logit tensors with T rows. `_route_split` constructs
the same BF16 `xm[T,4096]`; MoE sees T, not2T. UV, SK2B, boundary, router and
unpooler source stay byte-identical. In particular small T1/8/16 still select
their existing non-WGS builds and preserve the decoder's UV behavior.

Compilation exceptions propagate after the existing threaded build joins.
However, `_tune` catches per-backend runtime exceptions and can select Marlin;
its outer exception handler can also restore default picks. The new resource
check first runs inside `okc_run`, so an integration PASS alone is not evidence
that SS-A executed successfully. Preserve the tuner policy, require successful
SSAR resource records and inspect intended MAIN backend/profile coverage.
Retained first-call markers are not benchmark-phase identities. A missing or
failed admission record, unexpected backend or incomplete activation evidence
prevents promotion; it must not be presented as a shared-A win. If a guaranteed
out-of-tuner resource preflight becomes necessary, that is an additional
host-only export/call change to review separately before implementation.

## Actual parent profile scope

Scored97af job7a8c4941741645ee has one `moe_fused` row in every profile. Its
expected changed MAIN cases are:

| Indices | T / load | MoE rows, µs | Share of profiler totals |
| --- | --- | --- | --- |
|4,5|32 / decode32|80.3,85.2|50.0%,25.6%|
|8,9|128 / verify16|143.9,140.6|58.5%,42.0%|
|10,11|256 / verify32|205.8,204.6|61.7%,40.9%|

Indices0/1(T1),2/3(T16),6/7(T8) are controls. These are separate profiler sums,
not removable-time bounds; PDL overlaps and backend retuning limit attribution.
The shared-A feasibility report's conditional20% MAIN-row reduction corresponds
to about5.045% whole-decoder reduction, not a prediction.

If standalone gates pass, review one isolated port that changes only the two
nodes above. Verify full decoder AST preservation, exact embedded checked CUDA
after the comment normalization, decoder-specific non-WGS preprocessed controls,
unchanged real packing/scratch, PDL and resource-failure handling. Then obtain
independent source review and one complete private decoder check with all13
tests/12 benchmarks/12 profiles against both exact97af baselines. No integration
or performance claim is authorized by this map alone.
