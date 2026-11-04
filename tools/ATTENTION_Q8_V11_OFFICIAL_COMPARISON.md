# Official V11 comparison checklist

Prepared offline for private job `88f84f79fb11462d`. This note does not poll the
job or imply a result. Exact submitted wrapper:
`6724284b1e8aef03a3af38d1d9d008c1df680b241a37b2d688464b8eeb60633f`.
Its decoder core is `7b57f0f8a1936a818ac6a399c047d9a1739e3c865070621a57afdd9e420cc5ed`.

Use two distinct historical references:

- Standalone scored attention job `8f9e5971f95841a0`, source `87a9bddc…`,
  speedup2.611970. Its12 scored rows are comparable workload descriptions, but
  its compiler flags/PDL policy differ from the V11 wrapper.
- Decoder scored job `4b216098993e4112`, exact parent source `80a4eaa3…`.
  Its attention kernel uses the wrapper's exact parent source/flags. Its
  profiled durations are whole-chain component observations on soft decoder
  inputs, not paired standalone timings on coherent attention inputs.

The exact decoder-parent standalone wrapper is `dc6d0c4884dba218…`. There is
no completed paired V11 timing against it: both local screens stopped at a
resource gate. Do not manufacture that missing comparison from another run.

## Scored case mapping

All12 scored cases use TP8 (one KV pair), page1 and context80000. B16/B32 use
agentic spread. The values below are saved active87a9 official **reported**
latencies, not raw kernel durations. Index is zero-based.

| Index | Batch / Q length / layer / distribution | Expected kernel template | V11 ownership versus exact decoder parent | Active87a9 reported µs |
| --- | --- | --- | --- | ---: |
| 0 | 1 / 1 / sliding / soft | `<1,false,13>` | unchanged control | 4.9192 |
| 1 | 1 / 1 / full / coherent | `<1,false,11>` | unchanged control | 10.7860 |
| 2 | 16 / 1 / sliding / soft | `<1,false,1>` | unchanged control | 7.5785 |
| 3 | 16 / 1 / full / coherent | `<1,true,0>` | unchanged control; L1 excludes V11 | 86.7566 |
| 4 | 32 / 1 / sliding / soft | `<1,false,1>` | unchanged control | 9.8504 |
| 5 | 32 / 1 / full / coherent | `<1,true,0>` | unchanged control; L1 excludes V11 | 166.1107 |
| 6 | 1 / 8 / sliding / soft | `<8,false,15>` | unchanged control | 6.5373 |
| 7 | 1 / 8 / full / coherent | `<8,false,11>` | unchanged control; not flat | 14.2501 |
| 8 | 16 / 8 / sliding / soft | `<8,false,3>` | unchanged control; window/RoPE | 11.2186 |
| **9** | **16 / 8 / full / coherent** | **`<8,true,0>`** | **intended V11 target** | **100.4905** |
| 10 | 32 / 8 / sliding / soft | `<8,false,3>` | unchanged control; window/RoPE | 14.7387 |
| **11** | **32 / 8 / full / coherent** | **`<8,true,0>`** | **intended V11 target** | **183.1197** |

Expected target plan on the132-SM H200: flat264×1×1, cooperative, resident,
full/noRoPE, Q8. Local resource diagnostics established that plan and two-CTA
occupancy for the instrumented wrapper. The uninstrumented official build has
no resource export. A profile name `<8,true,0>` is necessary but alone does not
prove the runtime guarded branch was entered: occupancy failure can leave the
same kernel template running the fallback. Preserve that distinction if the
official output offers no direct residency evidence.

“Unchanged control” describes algorithm/source scope relative to the exact
decoder parent. It does not promise identical machine-code layout, and it does
not mean unchanged flags relative to active87a9.

## Parent timing references for the two targets

| Batch | Active87a9 reported µs | Active87a9 raw graph median µs | Active87a9 profiled attention µs | Decoder80a4 profiled attention µs |
| --- | ---: | ---: | ---: | ---: |
| 16 | 100.4905 | 104.176 | 100.3 | 105.2 |
| 32 | 183.1197 | 187.200 | 186.0 | 186.1 |

Keep these measurement columns separate. Reported timing is the judge's
paired-ratio/overhead-adjusted result; raw graph timing includes replay cost;
profiling is a separate diagnostic replay. Decoder rows use soft input produced
by its projections; standalone full rows use coherent input. The values cannot
be divided across columns to establish ownership speedup or DRAM bandwidth.

## Why the active87a9 comparison is confounded

The parent decoder wrapper and active87a9 contain **byte-identical CUDA and
C++ source strings** before V11 changes. CUDA SHA256 is
`17ccf33fe224afe1a3470747db337378cd9ec00ed2f7dbf2bd505d7f529ee787`.
Their compilation/launch policies differ:

| Policy | Active87a9 | Exact decoder parent and V11 wrapper | Cases affected independently of V11 ownership |
| --- | --- | --- | --- |
| `OKC_QDIRECT` | 1 | 0 | Non-flat Q8: direct FP8 Q staging versus intermediate staging. Target flat WS path already excludes QDIRECT. |
| `OKC_PK2` | 1 | 0 | Flat Q8: paired-half packing/offset tests and dependent late-sync schedule versus per-half path. |
| `OKC_W0ENABLE` | 0 | 1 | Q8 full resident grid: uneven two-CTA shares using W0_L8=1135. It also disables the compile-time FPART shortcut for flat Q1, so scored controls3/5 are confounded too. |
| PDL | off | on when B×L≤32 | All six Q1 rows and B1 Q8 rows6/7. No predecessor exists in the standalone wrapper; it cannot reproduce decoder-chain overlap. Targets9/11 and Q8 sliding rows8/10 have PDL off. |
| Architecture specification | explicit `-gencode` | `TORCH_CUDA_ARCH_LIST` | Both intend sm90a on H200; compiler setup and extension namespaces differ. |

The wrapper also carries the decoder's supported-shape/indexing bounds and
standalone workspace allocation without decoder outer sentinels. No listed
official case is expected to trigger the added bounds. Native launch/plan,
tuned overrides, normalization and PDL policy were proved equivalent to the
decoder parent by the wrapper builder. A wrapper's aggregate score therefore
cannot isolate the effect of QK/PV ownership from these pre-existing differences.

## Sixteen correctness cases: which exercise the new path

This is the saved official suite. Omitted Q length defaults1; omitted page
defaults64; omitted distribution defaultsnormal. All activation predictions
assume the expected H200 residency and supported-size checks.

| Test | Case summary | Expected V11 role path |
| --- | --- | --- |
| 0 | B1 ctx10 TP8 sliding Q1 | no |
| 1 | B3 ctx2047 TP8 sliding Q1 | no |
| 2 | B5 ctx5000 TP8 sliding Q8 page1 mixed | no |
| 3 | B4 ctx3000 TP8 full Q1 | no |
| 4 | B8 ctx8192 DP full Q1 mixed | no |
| **5** | **B2 ctx100000 TP8 full Q8 page1 normal** | **yes** |
| 6 | B4 ctx6000 TP8 sliding Q1 page1 edges | no |
| **7** | **B4 ctx6000 TP8 full Q8 page64 edges** | **yes** |
| 8 | B1 ctx1048576 TP8 full Q8 | no: single-request non-flat |
| 9 | B3 ctx4000 TP8 sliding Q8 page1 scale1/64 | no: sliding |
| 10 | B2 ctx9000 TP8 full Q1 scale1/64 | no: Q1 |
| 11 | B4 ctx80000 TP8 full Q1 soft mixed | no: Q1 |
| 12 | B4 ctx6000 TP8 sliding Q8 page1 soft | no: sliding |
| 13 | B2 ctx20000 DP full Q8 soft | no: eight KV pairs, grid33×1×8, resident flag false |
| 14 | B4 ctx80000 TP8 full Q1 page1 coherent mixed | no: Q1 |
| 15 | B2 ctx20000 DP full Q8 coherent | no: eight KV pairs, resident flag false |

Passing these checks plus scored replays covers normal/edges and coherent
target inputs. It does **not** directly establish the new TP8/full/Q8 path's
scale1/64 behavior or its performance on soft decoder inputs. In particular,
the DP soft/coherent tests do not validate the new role body. The local page64
crossing-tail/scaled cases were never reached before the resource gates.

## Interpretation when root supplies the result

1. Verify exact source identity, terminal status, all16 tests, flags/rejections,
   and scored replay correctness. A failure in tests5/7 or scored9/11 is a
   target-path failure; a failure elsewhere first points to controls, wrapper
   policy or shared compilation effects. Do not reclassify any failure as safe.
2. Inspect scored9/11 separately: reported latency, same-job SGLang paired
   ratio/CI, raw graph timing, and actual profile specialization. Preserve
   whole-log compiler warnings where provided. No ratio against a historical
   different-flag source is an isolated ownership measurement.
3. Read all ten controls. Interpret Q8 non-flat shifts with QDIRECT/PDL,
   Q1 flat shifts with W0ENABLE/FPART and PDL, and global shifts with machine,
   evaluator and timing variation. Do not divide target ratios by control
   ratios as a correction or silently discard slow controls.
4. Compare like measurement types and distributions. Historical decoder
   component times105.2/186.1µs provide scale/context, not a paired baseline.
   A passing useful standalone result justifies an exact-source whole-decoder
   check for soft input and applicable scaled/bounds coverage before integration.
   Keep remaining untested tail/scale combinations explicit if that suite does
   not exercise them.
5. Original local zero-local-gate failures remain terminal. An official private
   check supplies separate numerical/performance evidence; it does not erase
   the88-byte local footprint or prove the former allocator explanation.

Source records: `experiments/runs/8f9e5971f95841a0.json` and
`experiments/runs/4b216098993e4112.json`; human-readable profiles are beside
their exact candidates in `experiments/candidates/<problem>/profiles/`.
