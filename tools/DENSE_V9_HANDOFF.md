# Dense V9 measurement handoff

No new candidate, runner change, GPU allocation or upload accompanies this
handoff. Frozen V9 is `muse_decode_layer/candidates/native_dense_chain_select_v9.py`,
SHA `3d5b34620a05faa628c30077728328a28e362b4c9dfc54dbb9ee6d5f6dd4d58c`;
compare against frozen decoder `80a4eaa3...`.

## Bounded first comparison

Use `experiments/dense_v9_heldout_cases.json`, `--seed 91021`,
`--weight-seed 92011`, `--repeats 6`, and `--profile`. The six cases are,
in order, sliding T16, sliding T128, unchanged T8 control, sliding T32,
sliding T256, full T128. Sliding contexts are 2048; full is 8192. This order
also exercises transitions between prepared token-size paths.

The runner supplies actual input seeds `91021 + 1000*case_index + step`,
with eager steps 0/1, capture step 2 and timed steps 100..105. Those seeds
and the weight seed are disjoint from V9's three synthetic seeds
`47003 + 3*T + seed_index` and its original tuners' seeds. Do not embed
`seed` or `device` in the case JSON. Omit `local_start`: let the runner's
existing `case_fields` policy choose one common bank per case. That policy
is a local debugging approximation, not the official routing calibration.

At six repeats, this set makes 96 numerical checks (two implementations,
two eager inputs and six graph inputs per case), plus separate profiles.
Use the existing runner timeout/credit guard; two independent prepare calls
still dominate cost. The runner now records prepare wall time including
weight cloning, which is a conservative comparison with the 300-second
official preparation allowance.

The optional second file, `experiments/dense_v9_correctness_cases.json`,
covers T24/page64/outliers, T33/mixed lengths, and T32/q_len8/cache-scale1/64.
These probe nearby dispatch, a changed query-block layout, and scale handling.
Do not spend a second preparation run on it after a null first comparison;
root can append these three cases before the one allocation if desired.
Use a different base seed, such as 101021, if run separately. At two repeats
the optional three-case file makes 24 checks. Neither file replaces the
official full suite.

## Gate and attribution

First require all checks to pass, preparation below the official limit,
and a logged accepted V9 choice that changes an actual launch. A budget
skip or no accepted choice is an unexercised policy, not a negative timing
result for a changed backend. Compare actual prepared choices before
attributing a timing difference to V9.

For this short-context screening set, calculate the geometric mean of
the four affected sliding-case paired speedups (indices 0,1,3,4). Require
at least 1.03x and no affected case slower by more than 3%; the full T128
case must also be nonregressing within 3%. This is a screening gate only.
It is **not** the official six-load weighted decoder score. Long-context
attention dilutes a dense gain, so passing this screen still requires the
real scored shapes/full private check before a >=3% total-decoder claim.

T8 is untouched by V9's selector, but the two original tuners can still
choose different T8 paths. If metadata matches yet T8 changes by >3%, treat
the comparison as attribution uncertainty; do not divide target speedups
by the control or claim a small gain. Exact expert identities, local counts,
U and J should be recorded outside timing when available. The current
runner only records route summaries automatically for standalone MoE,
not decoder. Do not claim decoder counts were captured unless added.

## Safe metadata whitelist after each prepare

Capture these immediately after each module's prepare/synchronize, before
the next module prepares. Serialize only Python scalar/container values;
encode tuple-keyed dictionaries as key/value entry lists. Reject tensors,
extension objects, ctypes handles and other objects rather than using
`repr`, `str`, `__dict__`, or a recursive dump of the prepared state.

| Source | Capture | Purpose |
| --- | --- | --- |
| `state` | `mm_pick`, `moe_pick`, `gluev2`, `marlin_cfg`, `pbmax`, `att_ok`, `ma_ok`, `nsm` | Outer dense/expert policy and capabilities |
| `state['bfp']` | `choice`, `disabled`, `sm90`, `old_paths`, `comp`, `tune_log` | Inner choices, disabled paths, lossless packing and isolated timings |
| `state` | `dense_chain_log` when present | Accepted V9 changes, rejected trials, seed ratios, synthetic U/counts |
| `state['sk2b']`, `state['sk2b1']` | Only bucket keys and each entry's integer `grid`/`cap` | Whether fused boundaries are available |
| `state['ub2']`, `state['bank']`, `state['fa3']`, `state['bfp']` | Presence booleans only | Fallback/fusion availability |
| `state['bfp']['rq']` | Keys of `t`; keys of `plans` and each plan's Boolean `transposed` only | Available RQ weight layouts/plans; no handles |
| Module globals | `PDL`, `PDL_MAX_T`, `BFP_PDL_N`, `FORCE_GLUEV2`, `FORCE_MM`, `FORCE_PBMAX`, `PIN_PICK`, `BFP_MAX_T`, `UV`, `UV_MAX_T`, `UB2`, `SK2B`, `SK2B_UN_TOKR`, `SK2B1` | Runtime overrides; `FORCE_GLUEV2=False` overrides the state table |
| Module globals | `MOE_DEFINES`, `MOE_TINY_DEFINES`, `MOE_TINY2_DEFINES`, `MOE_MID_DEFINES`, their `*_MAX_M` values, `MAIN_FALLBACK_DEFINES`, `_MA_SFX`, `_MA_MAIN_NS`, `DQ7`, `MOE_MULDEC`, `MOE_HAS_PDL` | Effective expert builds, scale-decode suffix and launch policy |
| Module `_MA_EXTS` | Only already-existing dictionary keys (compile-define tuples) | Confirms built expert variants without invoking a builder |

Do not call `_moe`, `_att7`, `_uv_on`, `bfp__get_ext`, or another builder to
collect metadata: those calls can compile, allocate or mutate state. A
post-prepare value of `bfp_SK2_PDL`/`_MA_PDL` is the last setter's state,
not the per-T policy; use the policy globals above instead.

For the parent/V9 source, T8 selects TINY2, T16 MID, and T32/T128/T256 MAIN
when `moe_pick` is `agent`; MAIN uses the fallback defines if `_MA_SFX` is
empty and WGS would otherwise be selected. The already-built `_MA_EXTS`
keys include the DQ7 and scale suffixes; `MOE_MULDEC` and default NS/FNST/NSF
are added by the builder separately. Current resource collection visits
`_EXTS` only, so it does **not** automatically cover this decoder's
`_MA_EXTS`, `_ATT7`, `bfp__ext` or `_UB2_EXT` stores.

The most useful causal check is parent versus V9 **before** selections
(`dense_chain_log[*]['before']`) and final choices. Unrelated changes in
QKV, inner router, MoE/Marlin, T8, disabled paths or fusion availability
invalidate a clean attribution to the new selector; report them rather
than silently treating the kernels as matched.
