# KPERM3-only WGS candidate

Status: CPU packing/static checks pass; CUDA compilation, oracle correctness and
performance are untested. One isolated standalone candidate, with no decoder or
attention changes. Root owns official checks and any later promotion.

* Source: `native_wgs_kperm3_v9.py`, SHA256
  `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
* Embedded CUDA SHA256:
  `85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c`.
* Frozen parent: `ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a`.
* Donor: keell0renz/muse-open-kernel-challenge, commit
  `890809cee421500b00126d084e81b338ef9181ab`, decoder source SHA256
  `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
  Apache-2.0 LICENSE and NOTICE are preserved under `third_party/public_scout/`.
  The candidate retains the complete donor NOTICE as comments and the parent's
  complete existing license text. The donor's reported score is unverified.

The candidate applies the reviewed within-stage K permutation to WGS weights and
runtime activations. Paired W13 gate/up rows place SwiGLU outputs in the matching
order for W2. One eight-byte scale record and four exact BF16 scale pairs serve
each lane's entire stage. The existing pipeline, stage/job costs, ring depths,
joined barriers, FCI3 flush, GS2 commit order, NTT1 dual accumulators, FP32 ybuf,
final norm/reduction, PDL and every route remain intact. No SC6, MFIN, B4 finalizer
or BF16-ybuf changes are included.

The intended benefit is fewer scale-load/address/decode instructions. Unique
shared-memory scale bytes and HBM bytes remain unchanged; the old loads use
four-lane broadcasts. Gather now issues four coalesced 32-bit loads rather than
one 128-bit load, with the same bytes and destination layout. Register liveness,
gather issue overhead, compiler serialization, and ordinary FP32 reassociation
may erase any benefit. There is no gain forecast or claimed occupancy increase.

Only prepared weight entries 12/13 change. The source AST and actual byte proof
preserve all original natural skinny/fat/unit formats at entries 0..11. All
unchanged dispatch flags leave WGMIN=9, so the non-WGS alias of entries 12/13
cannot enter the natural-fat WGMMA path (`is_nat` requires WGMIN<=8). The
nonbatched GATHB=0 gather is also correctly permuted. WGSSC2=0 is explicitly
rejected by a static assertion; every incumbent build uses its default value 1.
General-scale fallback and M<=16 dispatch are unchanged. WGS shapes, including
nearby sizes and chunk tails, consume the matching permuted pair of weight banks.

Both parent prepared layouts remain: the tiny layout with `_POS` and the larger
layout with `_POS_MUL`. Persistent WGS weight sizes are unchanged (27 MiB per
expert, 864 MiB per 32-expert bank, with both parent banks retained). Preparation
adds exact nibble indexing within each stage and a paired row-index tensor;
it moves no activation/input computation outside `kernel_fn`. W13 permutation
can transiently hold two 32 MiB nibble buffers before returning a 16 MiB code
copy; W2 is half those sizes. The returned copy also stays live during subsequent
existing packing. These are allocation estimates, not measured whole-prepare
peak memory or time; allocator retention and original packing temporaries matter.
No extra main extension or second fallback WGS copy is introduced. Record actual
prepare duration and compiler resources from the official result before promotion.

CPU validation:

```
/tmp/okc-kperm-cpu-20261104/bin/python tools/test_wgs_kperm3_cpu.py
```

The temporary CPU environment contains NumPy 2.5.4 only; no Torch or GPU code is
loaded. The verifier executes AST-whitelisted candidate integer-packing helpers
through a NumPy adapter, then independently decodes their actual packed bytes.
It passed complete W13/W2 experts for both bit layouts: 100663296 FP4 codes and
6291456 scale bytes, all 4096 hidden channels, all 16 K stages, every warp/lane,
and all 14161 qualifying PRMT scale pairs. It also checks paired row coverage,
the exact runtime gather mapping, unchanged 256-column B-readiness groups, and
AST equality of dispatch, flags, fallback packers, scratch and entries 0..11.
Verifier SHA256: `98cc9c5e611e0348c3ad20ea3b2af355e6ad44a87710c617110d86fee72f1d59`.

Next authorized validation, performed only by root, is one official private MoE
check of this exact source. The official five tests include M4/M33/M257/M128-skewed/M1,
and fresh timed inputs cover all six scored sizes. A passing CPU proof does not
establish numerical or synchronization correctness. Retain failures and stop on
serialization, correctness failure, preparation timeout or no useful measured gain.
No independent GPU allocation or submission was made by the implementation agent.
