# V71 independent review

Reviewed the frozen V71 patch at SHA-256
`31f15bb4e125f8574efe2ae2e2ed83f385825a47c4395716b55d7fa65b0d24e9`.
It is 96,956 bytes across 19 files, based on SGLang commit `297e54fd`.
[Frozen patch](patch.diff) and [export validation](validation.json).

## Caller and collective checks

- Executed the exact decoder forward body with mocked calls in 16 cases. The
  tests varied workspace selection, router output, dense or routed FFN, and
  the final-layer boundary. The attention result was consumed by the gated
  normalization before the FFN ran. The borrowed tensor did not escape.
- Executed the exact MoE `_forward_tp` body in 32 cases, including verify and
  fallback modes, Humming and alternate experts, grouped admission, and an
  absent unpool projection. The direct reduction ran once only when the
  finalizer returned the borrowed tensor by identity. The fallback preserved
  the ordinary all-reduce path. Unpool consumed the borrowed result on the
  forward stream before the next collective.
- Checked that the new C++ entry validates the local pull-workspace pointer
  and launches the existing `all_reduce_2shot_pull_kernel` with the same
  multicast specialization, grid size, 512 threads and PDL setting. The
  operation uses the existing pull storage and the same FP32 NVLS accumulation
  with a BF16 result. The producer and consumer keep the BF16 tensor boundary.
- Checked that the frozen helper excludes two-batch and single-batch overlap,
  compiler tracing, deterministic mode, sequence parallelism, and alternate
  parallel layouts. It follows the stock backend resolver and multicast
  configuration for the captured shape.

## Workspace lifetime

The independent model represented two consecutive uses of the pull plane with
784 events for TP4 × 32 CTAs and 800 events for TP8 × 16 CTAs. The existing
first barrier orders each rank's local producer before peer reduction for
the matching CTA. The second barrier orders each peer's multicast store before
the local consumer. On each rank, the next producer is downstream of that
consumer on the same stream. The modeled transitive dependencies cover every
slot in both directions between uses. This verifies the source ordering under
the stock barrier and stream contracts. CUDA compilation and device replay
are still required.

## Scope and performance bounds

The V44 H200 startup log confirmed multicast and pull availability on the
judge node. The pinned TP8 graph policy enters multicast pull at 512 KiB,
which corresponds to 32 rows for the 8,192-wide attention output and 64 rows
for the 4,096-wide expert output. Where both paths are admitted, the two
copy-in/copy-out pairs remove four launches per routed layer. Across 62 routed
layers this is up to 248 launches and 744 MiB of logical copy read/write traffic
at 128 rows, or 1,488 MiB at 256 rows. The traffic corresponds to about 0.16 ms
or 0.33 ms at peak H200 HBM bandwidth if every copy reaches HBM. Actual cache
behavior, launch savings, numerical agreement, peak memory and throughput all
remain judge gates. [V44 startup log](../../runs/c3b7bd142c264853.log).
