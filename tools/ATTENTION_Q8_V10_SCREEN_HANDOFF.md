# Isolated decoder-attention V10 screen handoff

Frozen decoder candidate: `native_q8_role_pipeline_v10.py`, SHA256
`6eb318df58c3ef5576c4ea456b4b494e1ef4548090fa7707c84e247f960c2222`.
Critic approved these exact bytes after the native/layout/phase review. There
has been no GPU compile, run or upload by the implementing agent.

The first comparison can avoid dense/MoE preparation with these attention-only
wrappers under `muse_attention_decode/candidates/`:

| Side | File | SHA256 |
| --- | --- | --- |
| Exact decoder80a4 attention | `decoder_parent_v10.py` | `dc6d0c4884dba218674421dfc6b6a237c2f5eb437069cb088bf1aef7439039f9` |
| Decoder V10 attention | `decoder_q8_role_v10.py` | `d8be9200bb0357cacf8e48c104a44d9e57aa1e278afe2bdac27a4dd5d7d4790a` |

`tools/build_decoder_attention_screen_v10.py` rebuilds both and writes
`experiments/attention_q8_role_v10_wrapper_provenance.json`. Native CUDA/CPP
strings are byte-exact copies of their decoder sources; compiler setup is the
decoder `_att7`, renamed only for the standalone cache/source names and unique
extension namespace. QDIRECT0/PK2=0/W0ENABLE1 are preserved. Normalized ASTs for
launch, plan, workspace primitives, cluster buffers, inverse frequencies and
SM count match; launch globals and tuned overrides match. Decoder support/cache
bounds and PDL(T<=32) policy are carried over. Standalone workspace allocation
omits the decoder's outer sentinel wrappers, and there is no predecessor for
PDL to overlap. Targets T128/256 have PDL off. These are **isolated decoder
attention wrappers, not the active87a9 standalone incumbent**.

The fixed eight-case screen is `experiments/attention_q8_role_v10_cases.json`:
B16/B32 full Q8 at80K with both soft/coherent distributions; untouched B32 full
Q1 and B16 sliding Q8; page64 new-token crossing tails at3003 and4093 keys,
including scale1/64. Fresh seed base121021 and six pairs are fixed. The existing
two eager plus six graph checks per side total128 required oracle checks.
The runner records actual sequence-length sums outside timing.

Target at least10% lower affected attention-kernel time in both batches and
both distributions;20% is a stretch. No repeatable3% affected gain, any numerical
failure, or material regression ends this hypothesis without parameter sweeps.
Only a useful screen justifies whole-decoder comparison, then an official check.
Controls diagnose drift; never divide target ratios by control ratios.

## Mandatory compiled-resource visibility

`tools/attention_resource_probe.py` is a separate local-only helper. It changes
no scored source. Before importing each local wrapper:

```python
source, provenance = instrument_source(
    original_source, expected_sha256=original_digest, tag="unique_run_side")
```

The digest-pinned transform appends one **host-only** CUDA export and its C++
pybind declaration, adds its function name to binding, uses a unique extension
namespace, and enables `verbose=True` plus `-Xptxas=-v`. Exact reverse restoration
is asserted. No device body, compile optimization, math or timed call changes.
Save both transformed-source digests and provenance beside the experiment.

Compile explicitly without launching attention (the lightweight wrapper has
no `prepare`):

```python
extension, capture = capture_compile_call(module._module)
```

Save `capture["text"]` as the complete compiler-log artifact. Default cap is8MiB;
excess output is drained, marked truncated and explicitly invalidates the
diagnostic gate. Capture exceptions retain logs in `CompileCaptureError.capture`
and the original exception as cause. Calls redirect process FDs and are serialized;
the worker must not run other concurrent builds. Keep `compiler_summary(capture)`
for bounded warning attribution. Context-based attribution is an inference:
inspect full logs before dismissing a serialization warning as a fallback.
Missing compiler output from a reused cache is **not** evidence of no warnings.
The integrated screen also requires actual `ptxas info/warning/error` lines;
a verbose command containing `-Xptxas=-v` does not satisfy this gate.

After both compiler captures have been atomically checkpointed, the first
eager correctness call actually launches Q8 (`_launch<8>`), which sets that
instance's dynamic-shared attribute; case0 of the supplied file does so.
After that warmup, `collect_attention_resources(module)` calls the export on
the already-built `_MOD`/`_ATT7`; it never invokes a builder. The host `kn` table
selects `attn_kernel<8,true>` for flat, whose default MODE is0. The export obtains
attributes and actual occupancy for exactly `attn_kernel<8,true,0>`,384 threads,
`sizeof(Smem<8,true>)+1024` dynamic bytes. It returns only19 bounded integer fields.

Require all of:

- Actual `active_blocks_per_sm >= 2`, `local_bytes == 0`, expected shared size
  (`smem_struct_bytes=113664`, requested dynamic114688), no device-resource errors.
- `compiler_summary(capture)["compile_log_gate"]` must be true: complete
  capture and observed ptxas output, not merely a cache hit. Require no
  changed-flat WGMMA serialization warning;
  unresolved/unattributed warnings require inspection. Preserve unrelated
  fallback warnings separately rather than pretending the entire TU is silent.
- Target launch plan is flat with `n_splits * pairs == 2*sm_count` and one KV
  pair. Together with actual occupancy this proves `p.resident` is true; Q8,
  full/no-RoPE, cooperative flags and frozen macros then activate V10. If the
  guarded optimization is inactive, timing is not a test of the new design.

Do not silently switch to one CTA/SM when the resource gate fails. That requires
a new132-CTA decomposition and a deliberate design review. The resource export
does not launch device work or alter state, but the diagnostic namespace/export
can affect code layout; unchanged-control timing remains necessary.

CPU checks:

```sh
python3 muse_decode_layer/candidates/native_q8_role_pipeline_v10_verify.py
python3 tools/build_decoder_attention_screen_v10.py
python3 -m unittest discover -s tools -p test_attention_resource_probe.py -v
```

All passed locally. The verifier's host syntax test explicitly does not compile
CUDA or PTX.

## Integrated bounded runner

`tools/modal_h200.py attention-v10-screen --timeout 900 --dry-run` validates the
fixed wrapper/helper/case hashes without allocating. Root alone may remove
`--dry-run` after independent review. This dedicated mode rejects source,
case, seed and repeat overrides; the original credit/overlap guard, one-H200
resource limits, timeout and final sandbox termination remain in force.

The five uploaded files are the two original wrappers, public reference/checks
and digest-pinned resource helper. Original and instrumented source identities
are distinct in the manifest; both originals and exact instrumented wrappers
are frozen locally. A fresh namespace per run forces compilation diagnostics.
Both complete compiler captures are atomically checkpointed in
`/tmp/okc-result.json` before any attention launch. The result retains the full
bounded log, and retrieval writes `compile_baseline.log`/`compile_candidate.log`.
On worker failure/timeout the launcher tries to recover the latest checkpoint
before termination and preserves the original exception. Recovery is best
effort: an already expired sandbox may no longer expose its files.

For each side's first eager call per case, a temporary Python `_module` proxy
records only the actual native `run` scalar arguments, forwards the call once,
and restores the original function in `finally`. It is absent during graph
capture and timing. After the first Q8 call, resource attributes and the exact
frozen host occupancy/dispatch rules establish the selected flat kernel's
cooperative/resident flags; the evidence is checkpointed before synchronize.
The four long B16/B32 full-Q8 cases must use grid264×1×1 and activate the guarded
path. Controls and short correctness tails record their actual dispatch without
requiring target activation. No plan or occupancy override is permitted.

Incomplete/missing compiler diagnostics, changed-flat or unattributed WGMMA
serialization warnings, local memory, insufficient two-CTA occupancy or failed
target activation stop the screen. Other warnings remain in the full logs for
review. All128 oracle checks and the original separate-input CUDA-graph paired
timing protocol are retained; extra kernel profiling is diagnostic only.

`tools/test_modal_attention_v10.py` adds12 CPU tests including altered payloads,
native proxy restoration, control dispatch, actual compiler evidence, resource
failures and timeout recovery before final termination. Existing runner and
resource-helper tests also pass. GPU compilation, precision and speed remain
unverified until root runs this bounded screen.
