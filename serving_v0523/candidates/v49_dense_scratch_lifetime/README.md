# V49 dense prefill scratch lifetime

V49 is V44 with two earlier releases in `_dense_expert_outputs_impl`. The up
projection drops `w13` after its last GEMM reads it. The activation drops
`gate_up` after SiLU has been queued. Both consumers use the allocation stream
and the activation writes to its own output. The CUDA caching allocator can
reuse the released storage in stream order. Capture still skips the eager
body and records the same buffers. [PyTorch stream lifetime](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html),
[TVM FFI stream handling](https://tvm.apache.org/ffi/concepts/tensor.html).

At 16,384 local routes and 4,096-wide BF16 experts, the five named dense scratch
tensors peak at 608 MiB in V44 and 512 MiB in V49. The calculation excludes
allocator overhead, transient library workspaces, routing metadata, and the
unchanged graph buffers. Judge peak memory and throughput are unmeasured.
Routing, launch count, GEMM shapes, precision, and operation order are identical
to V44.

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`.
- V44 patch SHA-256 `0822e1279e1710bbb30f2e0e541dcf0055a105e7bc1032ae5b081638c1d27b4c`.
- V49 patch SHA-256 `59b1852ec241787c797345e3c74bc365b6856ecdaf2e77f2b85d75ce256467b6`.
- Local verification passed for CLI export, forward apply to the pinned index,
  reverse apply, whitespace, source restrictions, and syntax compilation of
  the three Python files. The other three patch entries match V44 byte for
  byte. The changed Python file matches V44's AST after removing the two
  `del` statements, and neither tensor is referenced after its release.
- No official check or scored run has used this payload. The comparison flag
  is `--flag swa-full-tokens-ratio=0.25`.

## Scratch cleanup

Removed two temporary directories owned by this experiment after verifying the
paths, Git state, saved patch, and open files. The first used a shallow-clone
fallback that copied the Git pack. The second used an alternate object store
and checked out only four source paths.

| Removed directory | Observed volume free-space change |
| --- | --- |
| `/private/tmp/okc-v49.i1aWLO` | +43,088 KiB |
| `/private/tmp/okc-v49.W0YIV4` | −1,047,272 KiB across the deletion window on the shared volume. The inspected directory occupied 3,192 KiB. |
