# New-K/V packed FP8 conversion, fixed v18

2026-11-04. Isolated candidate only; no official result yet.

Source `native_new_kv_pack_v18.py`, SHA
`41809d0c47f9469bd3095e50939fc5a742b97e07294dcd75aea62616d07aa663`,
is built from exact scored87a9 by `tools/build_attention_kv_pack_candidate.py`.
The sole executable CUDA delta replaces the new-K/V cache-write scalar loop
with two calls to the unchanged `e4m3x4f` helper. The Python extension namespace
and provenance comment also change. All other source bytes, compile flags,
launches, scheduling, stores, norm/RoPE and probability/output math restore
exactly to the parent. No decoder port is included.

The hypothesis is fewer conversion/packing instructions in the `has_new`
prologue. Eight scalar conversions become four explicit paired conversions
per active thread, with identical FP32 products and independent RN/satfinite
E4M3 lane conversions. NVIDIA CUDA12.8.90's scalar helper uses that same
paired instruction with a dummy zero in the upper lane; pinned primary
provenance is in `tools/ATTENTION_NEW_KV_PACKING_FEASIBILITY_20261104.md`.
Actual evaluator header lowering and ptxas combining remain unknown.

Scope is deliberately small: L1 executes one new-token segment per request;
L8 may duplicate the new-row prologue across a suffix boundary. The saved
feasibility table gives source counts, not measured instruction savings.
No repeated cached-K/V tile decode, QK/PV arithmetic or memory byte is removed.
Compiler combining can make this a no-op; changed instruction scheduling can
also regress total runtime. There is no predicted percentage gain.

The focused CPU proof passes: complete source restoration and deterministic
rebuild; unchanged helper/flags and exact x[e]*sc products per byte;
262,144 opaque conversion-byte pair layouts across all four pair positions;
three rejected source mutations. It checks the pinned local header. Boundary
review includes signed zero, subnormal/normal rounding ties, finite overflow,
infinity and NaNs through identical lane semantics. It does not emulate FP8
conversion, establish GPU NaN encodings, compile CUDA or measure SASS/timing.

```sh
python3 tools/test_attention_kv_pack_cpu.py \
  --expected-sha 41809d0c47f9469bd3095e50939fc5a742b97e07294dcd75aea62616d07aa663 \
  --output experiments/iterations/20261104-attention-kv-pack-root-cpu.json
```

Before one private check, require independent exact-source GO and root's CPU
rerun. Require complete clean correctness and12 benchmark/profile cases with
exact source/evaluator provenance. Compare public `8f9e5971f95841a0` and private
`c21d391dbf604bb5`, both87a9. The fixed pre-GPU performance screen is:

- Official aggregate speed ratio>=1.005 against BOTH parent receipts. The
  official score is the geometric mean of SIX load ratios, each formed from
  summed weighted case latencies (sliding46, full16), not a weighted12-case GM.
- At least3/6 weighted loads jointly faster than both parents.
- Every one of12 normalized case latencies<=1.02 against both parents.

All cases are potentially affected; there is no unchanged-control claim.
Sliding/full subgroup figures are descriptive. This is an operational screen,
not significance. A failed fixed attempt closes without a policy sweep or
repeat; a standalone gain would still need a separately authorized decoder
port and official whole-layer proof before any decoder promotion.
