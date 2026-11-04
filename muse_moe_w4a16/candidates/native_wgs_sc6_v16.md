# Lossless six-bit WGS scales over the scored KPERM parent

2026-11-04. Isolated candidate. Independent source review approved this exact
hash for one private check after root reproduced the CPU report byte for byte.
GPU outcome remains pending. Active submissions are unchanged.

Frozen source: `8044835232a26e8188471ac5296cd598339e00518641d03f055c129502aa8cc5`.
Parent: `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`;
public receipt `d24dd50196cb4158`, private `bcb761dfca264db0`.
The fixed plan is `tools/SC6_KPERM_PLAN.md` (660633ee).

The hypothesis is that reducing packed WGS weight requests from 27 to 26.25 MiB
per complete job can outweigh the extra scale reconstruction instructions.
This saves 2.7778% of packed weight bytes, not a predicted percentage of runtime.
The packing/decoder mechanism comes from the Apache-2.0 keell0renz commit
`890809cee421500b00126d084e81b338ef9181ab`; existing attribution and licensed
source/NOTICE remain preserved. No other donor scheduling or math changes enter.

Both actual local W13/W2 scale banks must pass the existing PRMT predicate and
contain only bytes 0x40 through 0x7e. That prepared larger layout owns its SC6
flag and compact entries 12/13. Its MAIN WGS extension alone receives the SC6
define. Ineligible banks and the tiny layout keep the original representation;
general-scale fallback remains available. The dormant non-WGS natural-fat
aliases of entries 12/13 remain excluded by unchanged WGMIN9 and build flags.

Four 8960-byte pieces form a contiguous 35840-byte global stage. Producer copy
and expected transaction sizes both use 35840; shared slot stride remains
36864, including 1024 unused tail bytes. Activation-ring base, 231168 dynamic
shared bytes, 576 threads, GS2/wait1, three weight/activation slots, KPERM,
NTT1 dual accumulators, FCI3, costs, all precision and reduction ordering stay
the parent implementation. Additional 32/16-bit scale loads and integer work
can hurt; actual compiler resources and runtime are unknown.

The bounded SC6R host helper reports exact BIG=false/true attributes and
occupancy, after the original max-dynamic attribute setters. API failure or
zero admitted CTAs throws before caching initialization success. Local memory
and compiler serialization are reported evidence, not automatic rejection.
One SC6W preparation record states the layout actually selected.

CPU verification passed with verifier
`ca7766ecd5d0eb0ea9ec6498831a9f2f6cfc18368ea0a47f7be9e51d1a85b0fd`;
report `experiments/iterations/20261104-wgs-sc6-cpu.json`, SHA
`3caa7461ea40db76d5b19bb8255982285ab2e06862ce7af9a2fb86a6caa9650f`.

- Restores the entire CUDA parent after the declared delta; 38 other Python
  nodes and 16 preprocessed non-WGS builds match. All post-scale decode operands,
  MMA ordering, producers other than the three byte-count expressions, and
  reduction/finalizer paths are preserved.
- Actual WGS pack helpers cover complete W13/W2 experts in both bit layouts:
  100663296 unchanged FP4 codes and 6291456 reconstructed scale bytes. Ordinary
  fallback matches parent bytes, including arbitrary eight-bit scales.
- Actual packed bytes feed the native candidate scale decoder: every qualified
  63-by-63 pair in each position with four backgrounds, plus all64-by-64 values
  in both cross-packed fields as an algebraic superset. This includes intentional
  uint32 wrap; qualification separately rejects 0x7f.
- Complete actual preparation and kernel functions execute through CPU mocks:
  ten layout preparations, five eligible/ineligible banks, eight alternations,
  72 calls and 80 extension launches, including M385's partial final batch.
  Heavy unchanged ordinary packing/LUT arithmetic and CUDA calls are mocked in
  this integration check; full WGS packing is checked independently above.
- Six negative controls reject wrong multiplier/bias/cross-bit mask, old TMA
  payload/piece stride and an SC6 mode leaking between prepared states. Both
  SC6 modes exercise native API-error/occupancy/admission retry checks.

Reproduce with `/tmp/okc-kperm-cpu-20261104/bin/python tools/test_wgs_sc6_cpu.py
--expected-sha 8044835232a26e8188471ac5296cd598339e00518641d03f055c129502aa8cc5`.
CPU evidence does not establish CUDA compilation, GPU tensor results, memory
ordering or performance. No GPU run was made by the author.
Independent source verdict: `experiments/iterations/20261104-wgs-sc6-source-critic.json`
(240d3a37), GO for the exact frozen source; no requested changes.

The predeclared screen compares M32/128/256 against both exact parent receipts:
affected GM relative speed at least 1.01, at least two of three affected cases
positive against both parents, and every affected normalized latency at most
1.02. M1/8/16 control GM normalized latency must lie in [0.98,1.02]. All official
correctness, evaluator and provenance checks must be clean. This is a consistency
screen, not a statistical significance claim. Failure closes this fixed attempt;
a useful standalone result only permits a separately reviewed decoder port.
