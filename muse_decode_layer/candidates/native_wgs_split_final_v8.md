# WGS MoE finalizer split experiment

Status: rejected for promotion after a bounded same-GPU Modal comparison.
Local CUDA compilation, numerical checks and fresh graph replays passed, but
there was no useful speedup. No official check or score was submitted.
The passing incumbent is untouched. Initial design notes follow; the result
and independent critique are recorded at the end.

Candidate: `native_wgs_split_final_v8.py`, SHA-256
`9e8881479fc71c985a0724ef60207cdd4d6df66fc027e8fcfafbe1eba55aa01b`.
Parent: exact frozen decoder source
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`,
scored run `4b216098993e4112` at 2.1501860996x.

## Hypothesis and scope

The WGS expert kernel currently holds a 576-thread persistent CTA per SM
through a grid-wide spin barrier and a finalizer that retains up to 64 FP32
route values per thread. A separate finalizer can remove that spin barrier,
isolate the finalizer's register requirements from the two GEMMs, and schedule
small independent token CTAs. It pays an extra launch and rereads the FP32
route rows. This is a structural test, not a claim that occupancy increases:
the main kernel's shared memory still limits it to one CTA per SM.

The candidate changes only the WGS path selected above 16 tokens. It copies
the current route-to-row map from CTA 0, executes the original A/B work, then
returns before the old grid barrier/finalizer. An ordinary same-stream launch
waits for all main-kernel consumers and producers, calculates every route's
FP32 norm and weighted sum, writes BF16 output, and resets per-call flags.
No main-kernel PDL dependent-launch signal executes on this split path.
Tiny/MID implementations, all attention, projections, routing, precision,
weight packing and fallback dispatch remain unchanged.

The main compile definition adds `-DOKC_SPLIT_FINAL=1`. All expert extension
names use `dl_okc_moe_splitfinal_v8_...` to avoid old/new build collisions.
Other extensions are identical and may share their existing compilation cache.
Scratch element 3 is a 3,072-element int16 route map, overwritten for every
live route each call. Existing optional debug storage moves to element 4;
the candidate does not enable it or add timers.

## Measured opportunity and expected ceiling

In the exact scored profile, MoE consumes roughly 40% of decode-32 and
55%/54% of verify-16/32 time after the sliding/full shape weights are applied.
These are the three of six scored loads this candidate can directly improve.
The old profile does not separate GEMM, barrier and finalizer time, so an
empirical finalizer upper bound is not yet known.

A working upper estimate of **20 us net saved per affected shape** would
reduce aggregate normalized latency by **3.87%**, raising 2.1502x to about
2.2367x. Saving 10 us produces 1.91% (2.1921x), and 5 us produces 0.95%
(2.1708x). These calculations use the scored weighted load times 197.30,
279.87 and 369.63 us; they assume unaffected loads stay constant. The actual
gain may be zero or negative because the second pass and launch add work.
This experiment alone is not expected to close the 14.76% latency gap to
the current 2.5226x leader. Its value is a measured improvement and a direct
test of whether persistent-kernel tail handling merits a larger rewrite.

Continue only if same-GPU comparison shows at least 10% expert improvement
for M128/M256 without more than 3% regression at M32 across three fresh
routing seeds. Whole-layer timings should be checked independently because
the prepare-time synthetic tuner can select different dense/glue paths.
For attribution, optional diagnostic builds can enable the incumbent's
existing `OKC_DEBUG` timeline and append int64 `[160, 8]` storage: element 3
for the old source, element 4 here. Compare phase 3→4 (old barrier), 4→5
(old finalizer), and separate finalizer kernel duration. Keep diagnostic
builds separate from source uploaded for an official check.

## Correctness risks and validation

- The final kernel must wait for the whole main kernel. It deliberately uses
  an ordinary launch without programmatic stream serialization.
- The old barrier incremented the scratch generation and reset work flags.
  The new finalizer resets `doneA`, `xready`, `doneAs`, counters 0/1 and advances
  generation once. Split-unit fixup buffers/counters still reset in their
  unchanged GEMM code. Repeated fresh graph replays are essential.
- The norm reduction grouping changes, while all accumulation remains FP32.
  Compare final MoE/whole-layer outputs against the official FP32 reference.
- Tokens with no local routes must produce all-zero expert output; route maps
  contain -1 and the finalizer masks every such row before reading it.
- Check M17/M33 and larger odd counts, and chunks around the existing 384-token
  scratch limit, before treating nearby-shape handling as established.

The accompanying JSON contains M8/M16 controls, an M17 boundary, and M32,
M128, M256 experiments. It uses 2K sliding contexts to isolate the MoE test;
promising changes still require all official shapes and 80K contexts.

Validation from the repository root, using the bounded existing runner:

```bash
python3 -c 'import ast,pathlib; ast.parse(pathlib.Path("muse_decode_layer/candidates/native_wgs_split_final_v8.py").read_text())'
.venv-modal/bin/python tools/modal_h200.py benchmark --problem muse_decode_layer --candidate muse_decode_layer/candidates/native_wgs_split_final_v8.py --cases muse_decode_layer/candidates/native_wgs_split_final_v8.cases.json --timeout 900 --dry-run
```

The dry run does not allocate a GPU. Root owns the bounded real run and the
same-GPU incumbent comparison; existing runner baseline-only output is not
sufficient to attribute a small gain. After a useful same-GPU result, root
can run `okc check` on this exact source. No promotion is authorized by these
local checks alone.

Live `https://openkernelchallenge.dev/llms.txt` was read on November 4 UTC.
Its kernel restrictions still allow BF16 or FP32 activations, FP32 expert
epilogues, and readable source compiled during the run. This candidate makes
no new precision concession and retains the complete upstream licenses.

## Root launch record

The reviewed candidate was launched on Modal at 2026-11-04 04:09 UTC as an
unofficial paired comparison against frozen incumbent `80a4...`. Evidence:
`experiments/modal/20261104T040903Z-f1613663`; app
`ap-fdcwQHJrE5tWvFM50R8ovQ`. The root case file adds the M16 → M32 → M16
scratch transition, with four fresh timed replays and separate CUDA profiles.
No result or gain is established yet. The exact candidate hash is unchanged.

## Completed result and critic verdict

The run passed all 84 local numerical checks, including the M16→M32→M16
scratch transition. Profiles prove that the new finalizer executed on the
changed path. Sandbox termination and stopped-app state are confirmed.

| M | Whole-decoder paired ratio | Incumbent MoE µs | Split MoE + finalizer µs |
| --- | ---: | ---: | ---: |
| 32 | 0.987678× | 105.844 | 103.756 |
| 128 | 1.000261× | 178.449 | 189.535 |
| 256 | 0.998414× | 227.224 | 232.210 |

The independent critic rejected promotion: the required 10% expert gain is
absent. Exact regression percentages are uncertain because there are only
four paired samples and one profiler replay per case; dense autotune choices
also differ at M128. This run uses one weight/router seed. No further V8
confirmation run is justified without a new mechanism. Preserve the evidence
and measure the original kernel's scheduling headroom before rewriting it.

Decision record: `experiments/iterations/20261104-v8-split-final.json`.
