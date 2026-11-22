# V66 graph-safe grouped Humming decode

V66 extends V64 with a fixed-capacity, grouped-contiguous W4A16 path for
small target decode and verify batches. It uses the same packed NVFP4
weights for grouped decode, grouped prefill, and the indexed fallback.

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Exact V64 parent `f6aaf93ce06ccfce9461e06bc3f71dea05dd901c01497c8be31bd2c515da689a`
- Patch SHA-256 `1032101b0fbdf8ccc8a5e0b858b6c507a1b9958ac7a0996344a7ca29a8d2c6fd`
- Patch size 61,072 bytes across 11 allowed files. The delta touches six files.
- Flags `--flag moe-runner-backend=humming --flag speculative-moe-runner-backend=marlin --flag enable-layernorm-sp --flag swa-full-tokens-ratio=0.25`
- Hopper option `SGLANG_ENABLE_MUSE_SPARK_HUMMING_GROUPED_DECODE` defaults on

The one-CTA planner applies the layer's original expert map, counts and
stably sorts slots, and returns the inverse map used by finalization.
If the batch has `S` slots and `R` live local routes, the planner puts
the live rows in `[S-R, S)`. Humming's grouped scheduler uses the full
matrix row count as the last expert's end. A suffix lets the fixed
`S` allocation supply that boundary with no host read of routing data.
Rows outside every group are ignored. The workspace keeps the inherited
48 MiB of BF16 arenas for 2,048 slots with a 4,096-wide expert. The planner
adds 24,708 bytes of metadata at TP8 with 32 local experts. The optimized
path admits shapes with up to 2,048 slots and up to 64 experts on a rank,
and uses the indexed path for ineligible decode shapes.

The two grouped GEMMs enable TMA for activations and packed weight tiles.
They keep Stream-K disabled and use FP32 accumulation. Output stores
mask rows after each expert boundary. A CUDA kernel applies the first
scale, rounds to BF16, and runs the same fast-math SiLU helper and
multiply as the original activation. Finalization rounds the second
scale product to BF16 before per-route RMSNorm and sums in original slot
order. The expert all-reduce and draft path keep their inherited dispatch.
[Humming 0.1.12 source release](https://pypi.org/project/humming-kernels/0.1.12/)

With one shared indexed alignment, the normal EP expert path uses about
11 launches for expert mapping, alignment, GEMMs, scaling, activation and
finalization. V66 uses six for planning, gather, two GEMMs, activation and
finalization. At 256 local live routes, the removed scale sweeps account
for 12 MiB of logical reads and writes per layer. Staging the live W13
inputs adds about 4 MiB per layer. The tuned M tiles are 8, 16 and 24 at
TP8 over the supported decode/verify range. Both modes use identical
geometry at each tier, so the grouped path reads the same logical packed
weight bytes and runs the same number of padded matrix tiles. The grouped
tiers use 640 threads and an estimated 82,944–123,904 bytes of shared
memory per CTA on SM90. There are three extra compiled tiers per
projection for TP8 and TP4.

Under uniform top-8 routing across 256 experts, the expected unique
packed-weight reads at M8/32/128/256 are about 194/551/849/864 MiB per rank
per layer. At M256, 62 routed layers imply 52.3 GiB, or 11.7 ms at the
H200's 4.8 TB/s peak. Those are logical volumes and an ideal bandwidth
scale. TMA transport, dispatch, caching, and routing skew determine
performance. V66 is a decode component optimization whose end-to-end
gain still needs measurement.
[H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)

Pinned-base forward and reverse application, whitespace and paths passed.
The official local `okc serve diff` export from an isolated sparse
checkout returned the identical patch. The executable [CPU harness](cpu_check.py)
passed 600 mapped planner cases, 48 complete grouped-and-finalizer
cases, 56,157 route-value comparisons and 71 dispatch cases. A separate
review exercised 3,036 exact planner cases and audited the Humming
scheduler, grouped output tail, CUDA SiLU and BF16 boundaries. The
[receipt](proof.json) records details. H200 JIT, cooperative resource use,
graph replay, numerical agreement, peak memory, and performance remain
unverified.

The closed `okc-v66-suffix-BK5wjM` staging directory and
`okc-v66-export-iPp02U` sparse clone were removed after inspecting their
contents, Git state, and open files. The two copies occupied 48,136 KiB
in total. The export matched the frozen patch and held no new commits
or stashes. Available space on the shared volume changed from
55,840,928 KiB to 55,889,044 KiB.
