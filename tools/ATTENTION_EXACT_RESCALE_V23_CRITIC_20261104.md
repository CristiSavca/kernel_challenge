# V23 exact-one output rescale — independent source review

**GO for one fixed private correctness/performance check of source `c6b90f16…`, subject to root's independent CPU rerun.** No source blocker found. This is not GPU correctness, a universal floating-point bit-equivalence proof, performance acceptance, or permission to promote/port the candidate. Review performed2026-11-04; no GPU/provider/submission calls or candidate edits.

## Frozen scope and verification

- Parent87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c.
- Candidate `muse_attention_decode/candidates/exact_rescale_v23.py`: c6b90f168bfdc21a8583b6a4bd03cebc0e6c4e03f994f74759c54ab5b677335c.
- Plan70c4e49bf08a58894d84007d18f7260cf4fd41c87bcba96591e1986d65804bda; verifier869bd22bd9a2a8f25b5a219ada5844ed13cbc1552203a6fa61261be701ba879f.

I read the full focused diff, surrounding `segment` control flow, launch/build/dispatch and verifier. Reversing the one scalar guard, provenance-comment replacement and cache namespace restores the entire parent file byte for byte. All inherited MIT/Apache notices, first-line problem marker, normal flags, output allocation, caches/workspaces, timing/debug defaults and fallback logic survive. No donor source body was copied; the PipeLink mechanism is credited with its pinned Apache-2.0 source.

The independently executed verifier returns exactly the author's saved report: three rejected scope mutations;4091 finite O patterns ×nine alphas;1,178,208 checked state elements;130,912 alpha-one element checks. Counts include16 affected and16 untouched O elements per invocation. These are host compiler/extracted-source tests, not independent GPU execution cases. Host compile disables fast math and fusion equally; it cannot certify CUDA FMA/control lowering. Five nonfinite O inputs are reported separately, including a signaling-NaN difference.

## Arithmetic, participation and lifetime

At candidate2926, `if (alpha != 1.f)` encloses only the existing16 O-register multiplications in Q1, non-LAZY, non-ONE. Alpha, maximum update, eight probability exponentials/thread, probability casts, row-sum expression, all QK/PV MMAs, every key/value, merge/finalization and synchronization remain unchanged. The first live finite tile has alpha0 and retains its original products. The existing all-invalid branch is unchanged. NaN alpha compares unequal and executes the original products.

Alpha is quad-uniform, not necessarily warp-uniform. The new branch contains only private scalar arithmetic and closes before the existing MMA calls; it adds no vote, barrier, shuffle, early return or release. Ordinary reconvergence/predication handles mixed row groups. This avoids the rejected full-mask-vote participation hazard. No shared/global data traffic or ring lifetime changes occur.

Finite FP32 multiplication by positive one preserves its exact value and signed zero under gradual underflow. NVIDIA documents non-FTZ `mul.f32` subnormal support on sm20+, with flushing selected by its `.ftz` form; NVCC defaults FTZ to false and fast-math would enable it. The unchanged build has neither global fast-math nor FTZ. The separate `ex2.approx.ftz.f32` does not imply an FTZ O multiply. These primary semantics support the finite-value argument but are not inspection of the judge's actual binary. [PTX multiply](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul), [NVCC FTZ](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ftz).

There is no universal NaN-payload equivalence: host input O0x7f812345 quiets under multiply and remains signaling under a copy. Both remain NaN and cannot be a valid finite task output; this does not prove all possible intermediate O values finite. GPU output checks remain mandatory. No threshold comparison or altered precision is permitted as a rescue.

## Coverage, cost and fixed decision

Both actual saved87a9 profiles confirm affected indices1–5: B1full `<1,false,11>`, B16/B32sliding `<1,false,1>`, and B16/B32full `<1,true,0>`. Control0 is B1sliding `<1,false,13>` with ONE; controls6–11 are Q8. `OKC_LAZY1=0`; ONE is `!FLAT && (MODE & 4) != 0`. Thus five affected/seven controls is source/profile grounded, not an inferred alpha-one frequency.

Source work removed is bounded by16 FP32 products/thread/eligible tile,512 lane products per warp. Those are not measured instructions/cycles. An added comparison and conditional execution may consume issue slots; mixed scales, predication, code layout or register changes may erase or reverse the benefit. Initial compilation under a unique cache name is outside measured graph replays but still must fit the official total run limit. No new per-call startup/launch/host allocation was introduced. Actual compiled resource use and skip frequency are unknown.

The previously recorded decoder profile scenario gives approximately0.393/0.788/1.587% whole-decoder latency reduction if the affected attention paths become5/10/20% faster in latency terms; deleting those complete paths models8.413%. These are separate-profile accounting scenarios, **not hard runtime bounds or forecasts**. Removing only rescaling is much narrower; no first-place claim follows.

I independently reproduced the exact six-load scorer: for each load use `(46*baseline_sliding +16*baseline_full)/(46*candidate_sliding +16*candidate_full)`, then the geometric mean over six loads. Saved parent public8f9e5971f95841a0 gives2.6119701144979857; privatec21d391dbf604bb5 gives2.6425388867913835. Both bind87a9, H200 and the same evaluator/image. Seeds differ; future result comparison must retain that limitation and confidence intervals.

Root's final pre-result gate governs: clean16 tests/12 benchmarks/12 profiles; official aggregate ratio≥1.01 against both parents; five affected-case normalized speed GM≥1.03 against both; at least3/5 jointly faster; no affected latency ratio>1.02; seven-control latency GM within[0.98,1.02] against each. Verify source/evaluator/shape and selected profile dispatch. This is an operational consistency screen, not statistical significance. Any correctness/resource/provenance/gate failure closes this fixed attempt: no retry, threshold/branch sweep, Q8 expansion or automatic decoder port. A useful passing result still requires separate decoder evidence before integration/promotion.
