# V58 single-bank native MoE feasibility

The V27 serving prototype stores one packed WGS bank per sparse layer. It couples the exact V26 small-batch kernel to Muse's existing post-expert norm. A separate inverse-layout dequantizer can feed V50's dense BF16 prefill from the same bank. This is a source design with an exact CPU layout check. There is no V58 serving patch or GPU measurement.

## Why the existing kernel stops at 384 tokens

V26's official component run `274c69cbc0954d76` recorded 231,168 B of dynamic shared memory per CTA for both variants. It uses one cooperative CTA per SM. The hard limit is not a dispatch preference.

- With `MMAX=16384`, `EMAX` is 131,072 slots. `Smem.eid`, `ent`, and `pos` alone would need 655,360 B per CTA. The Hopper block has about 227 KiB before any WGS work rings or job tables. `ent` and `pos` are signed shorts and overflow on the widened slot domain. Increasing the cap from 384 to 512 already adds 5,120 B from those three arrays alone.
- With 64 routes per job, `JMAX` grows from 88 to 2,088. The host uses two fixed 512-counter regions for `doneA` and `xready`. `doneAs` has only 136 × 16 slots. These cannot be fixed by widening one field.
- Worst-case packed activation workspaces `xA` and `xB` would total 2,421,161,984 B. The route output is another 1,073,741,824 B, plus 524,288 B of local IDs and roughly 40 MiB of fixed split-K state. Approximately 3.29 GiB would be dedicated before allocator effects. Stock also has route temporaries, so this total alone does not determine the judge's memory delta.

A full-prefill cooperative WGS design would need global routing metadata, a new scheduling and lifetimes contract, and compact workspaces. Increasing `MMAX` is insufficient.

## Exact layout inverse

The frozen V27 packer source SHA256 is `7310dd760cee4ae4802b1290d46d3dbb2e871a1cde131b78bb6e69743c23d0c1`. One bank for 32 local experts occupies 864 MiB per layer, equal to the original FP4 and E4M3 payload. Its two arrays contain contiguous 9,216-byte groups indexed by 256-row block, 256-column stage, and warpgroup. Each group stores 8,192 code bytes and 1,024 scale bytes.

For an original K coordinate within a stage, write `k = 16*(4*t+j) + 4*group + 2*q + parity`. For physical row `r` within a 256-row block, `wg = r//64`, `warp = (r%64)//16`, `h = (r%16)//8`, and `g = r%8`. The 32-bit code word and scale byte inside that warpgroup's group are

```text
lane = 4*g + t
word = ((group*4 + warp)*32 + lane)*4 + j
scale = 8192 + (((warp*8 + g)*4 + t)*4 + j)*2 + h
reg = 2*q + h
```

The four source bit positions are `(6,7,8,15)[bit] + 16*parity + 3*reg` modulo 32 for `reg < 3`. For `reg == 3`, use `(4,0,1,3)[bit] + 16*parity`. W2 rows remain in original order. W13 packs the gate/up halves in alternating eight-row groups after the channel permutation `k` above. For a logical channel, invert its within-256 permutation as `16*(4*group+j) + 2*t + ((i&2)<<2) + (i&1)` where `i = 2*q+parity`, then restore the eight-row gate/up interleave. This maps both halves back to canonical gate-first, up-second BF16 rows.

A CPU round trip executed six unmodified V27 packer functions through a NumPy shim. A separately written inverse recovered all 9,437,184 code nibbles and 589,824 scale bytes from four gate-up or down slabs at K=4096. It covered the complete address pattern through several 256-row blocks. A CUDA loader could consume one contiguous 9,216-byte group and unpack into a shared 64×256 BF16 tile before coalesced row-major writes. It should reuse V50's exact nonnegative E4M3 scale transform and BF16 operand arithmetic. Its GPU performance is unknown.

The [CPU proof receipt](proof.json) records both the packer hash and the per-slab comparison counts.

## Integration decision

A narrow implementation would rebase V27's validated source contracts to the current loader and configuration API, keep the V26 decode core and a single model-owned 123.03 MiB workspace, then use the new inverse for V50's larger prefill. Native and dense forward paths share the same packed weights and prepared BF16 globals. The decode finalizer retains Muse's stock per-route norm and slot order. No full-model dual bank is retained. The old module paths, execution flags and loader hooks need revalidation on SGLang `297e54fd`; V27's original adapter did not complete a GPU forward.

Medium batches have a material regression risk. Extrapolating the V26 128/256-token component times gives about 254 μs at 384 tokens. Eight serial chunks at 3,012 tokens would then cost about 2.0 ms versus V40 Marlin's measured 1.49 ms. At 8,192 tokens, 22 chunks would cost about 5.6 ms versus 3.4 ms. A lower switch to dense prefill around 6,000 tokens could reduce that penalty but has no matched evidence. The same layout's CUDA unpack bandwidth, model-scale loader startup and current cooperation/graph behavior remain open.

The V26-like decode component benefit alone is about 1.056× in an illustrative V48b latency budget, or score 1.166 starting from 1.1037. Prioritize the smaller V53 Humming prototype for a first check. Keep this one-bank design as a fallback until exact current-core and dense-layout measurements can justify a serving implementation. The standalone public 2.37× MoE and 2.247× decoder scores do not establish an end-to-end serving multiplier.
