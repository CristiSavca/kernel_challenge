# SF Tensor PTX-in-Lean release assessment and first implementation boundary

2026-11-04, public read-only review. **No public release of SF Tensor's Lean PTX model or proofs was located. We can implement a meaningful, source-bound Lean proof of our active packed decoder now, without representing it as their model or as whole-kernel verification.** No toolchain was installed, no proof/code was built, no kernel source was edited, and no GPU/paid resource was used in this task.

## Resolved links and release evidence

The user's shortlinks returned HTTP301 to:

- `https://t.co/kDA2911agg` → [SF Tensor engineering article](https://sf-tensor.com/engineering/formalizing-ptx).
- `https://t.co/WW2GwZ4Ak9` → [SF Tensor paper PDF](https://sf-tensor.com/papers/formalizing-ptx.pdf).

The paper is also [arXiv2611.04447v1](https://arxiv.org/abs/2611.04447v1), submitted November3, by Luk, Tom and Ben Koska. At this review, the [verified official GitHub organization](https://github.com/sf-tensor) and its [public repositories API](https://api.github.com/orgs/sf-tensor/repos?per_page=100&type=public) show exactly one public repository, [`sf-tensor/tcgen05`](https://github.com/sf-tensor/tcgen05). Its observed master commit is `d96b82180ac9041c09295f3a6a3c2c4e592ddce6`, September27; license MIT. That is the Blackwell tensor-core arithmetic model, not the Lean PTX semantics/proof package. The engineering article links the paper and earlier work but no source repository, Lean dependency, artifact archive or code license. Thus there is no identified Lean release to pin/install. This is a bounded public-discovery result, not proof that private code or an unindexed personal repository does not exist.

Compact public fetch evidence at approximately18:11UTC: article HTML117499B SHA256 `1f897812bb66df85b21f824ddb90b6036f9d7603578b968f5060b011f7cc7a13`; organization repository JSON5155B SHA256 `449385c1805c5d9d4c5622a403b2c00611321f605db7251e8beab057d9c33bfc`. The paper PDF response advertises455518B; its body was not downloaded/hash-verified here. GitHub metadata and article were read directly without login, clone or execution. The paper's arXiv license is the perpetual non-exclusive distribution license; that is not an open-source software grant. Do not inherit tcgen05's MIT license for unreleased Lean code.

## What the paper supplies, and what it does not

The [paper](https://arxiv.org/html/2611.04447) describes Lean4 v4.35.0 plus Mathlib, execution events and ordering relations for selected PTX instructions, asynchronous tensor operations and sequential kernels. Its guarantees concern completed executions; polling termination is not proved. Floating-point values are modeled as exact reals, excluding IEEE rounding and float-bit reinterpretation. Applying results assumes ISA undefined-behavior freedom and correct ptxas/hardware. Concurrent kernels, host races, multiple GPUs and noncoherent loads are outside the reported scope. Those are material boundaries for our BF16 packed operands, WGMMA pipeline and serving graph/EP8 paths. A formal proof of a hand-written event model would still need a faithful connection to our source and machine instructions.

Do not build a pretend SF Tensor API, copy mathematical prose as if it were runnable semantics, or claim their handwritten98-instruction Hopper example certifies our much larger compiled kernel. A complete PTX memory-model reimplementation is not the first bounded deliverable.

## Selected first proof: the live MULDEC packed operand transform

Pin `muse_moe_w4a16/submission.py` SHA256 `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0` and embedded CUDA SHA256 `85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c`. MAIN is built with `use_mul=True`; the active `dq4` branch at lines717–723 is **MULDEC**, not the inactive DQ7 branch. Its exact unsigned32-bit program is:

```text
M  = 0x81C081C0
r0 = q & M
r1 = rotateRight32(q, 3) & M
r2 = rotateRight32(q, 6) & M
r3 = ((q & 0x001B001B) * 4228 mod 2^32) & M
```

The two identical funnel operands make the fixed3/6 shifts rotations. Multiplication is modulo2^32, including wrap between the low and high16-bit halves; neither arbitrary-precision multiplication nor signed right shift is an acceptable replacement.

Build an **independent reference specification** from the source's `_stored_pos_mul`/`_POS_MUL` mapping, lines3970 onward. For each register R0..3 and half0/1, extract code bits0..3 from their packed positions, then scatter them to output bits6,7,8,15 plus16 for the high half. The desired sparse16-bit expansion of a four-bit code c is `((c & 7) << 6) | ((c & 8) << 12)`. Do not define the reference through the same rotate/multiply trick being proved.

Required theorems:

1. The32 `(register,half,codebit)` positions form a bijection of packed bits0..31; no code bit is missing or overwritten.
2. **For every q : BitVec32, all four implementation outputs equal the independent bit-placement specification.** This covers all2^32 packed words, not a random sample or just the16 code values.
3. A pack/decode corollary for all eight4-bit codes uses the actual positions and recovers the four expected operand bit patterns. Register/half ordering must match `_pack_codes_nat`'s `(h,q)` order, not only a list with eight unconstrained labels.

This certifies a nontrivial live optimization: the multiply-based fourth-register reconstruction and two rotations recover every code bit exactly. It is useful as a regression guard for future packed formats/decoder rewrites. It does **not** prove BF16 multiplication/rounding, scale decoding, KPERM's tensor coordinates, tensor MMA, race freedom, barrier progress, output numerics or performance. Treat sparse values as bit patterns in this first theorem; interpreting those patterns as scaled BF16 values is a separate future specification, not an implied result.

## Implementation and evidence contract for root

Use a pinned Lean4 toolchain with its standard `BitVec` library and a checked bitvector tactic such as `bv_decide`; Mathlib is unnecessary for this integer-only boundary. [Lean's official BitVec reference](https://lean-lang.org/doc/reference/latest/Basic-Types/Bitvectors/) documents fixed-width operations and bitvector automation. Verify the exact chosen toolchain/tactic interface during implementation rather than assuming the paper's environment is installed locally.

A small source extractor/checker must bind the proven expression and packing positions to the pinned real files. Fail closed on different source/CUDA hashes, a changed active preprocessor branch, unrecognized expressions/types, or a changed position map. Either extract the recognized arithmetic into generated Lean definitions or verify the exact reviewed snippets before building; report which link is trusted. The source-to-Lean mapping and independent specification still need review—Lean checking a theorem about the wrong translation would not protect CUDA. Keep the theorem universal, with no assumptions stating its own desired result.

Run a real Lean build, retain exact theorem/toolchain/source hashes and `#print axioms` results, and reject `sorry`, `admit`, `sorryAx` or custom axioms that bypass the property. Use the proof-producing tactic/checker rather than reporting a native executable assertion as a Lean theorem. No new GPU timing is needed for this proof.

Meaningful negative checks are inexpensive: changing rotate3→4 must fail (q=512 distinguishes r1: expected64, mutant0); changing multiplier4228→4224 must fail (q=16 distinguishes r3: expected64, mutant0); changing one packing position must fail the bijection/specification link. Preserve these as rejected mutants, not alternate candidate kernels. Altering source without refreshing the proof binding must also fail before the theorem can be advertised as current.

A suitable completion claim is: **“Lean checked the active decoder's32-bit packed operand reconstruction for every input word, with the proof tied to current source.”** Not “SF Tensor installed,” “CUDA formally verified,” “GPU races eliminated,” or “proved faster.” After this real first artifact, a separately reviewed event-order theorem can target a concrete current async invariant; it should not silently adopt an unsupported PTX model or replace runtime correctness/progress checks.

## Implemented local fragment (same-day update)

Root installed the official Lean4.35.0 toolchain; this task implemented `formal/OKC/Dq4.lean`, its Lake package and toolchain pin. The four universal `decodeN_correct` theorems and `positions_bijective` passed a real build. `masked3` and `multiply3_codes` prove the fourth output by isolating its eight influential bits, then kernel-checking all256 two-code assignments. No theorem contains an unproved assumption, custom axiom or native-checking dependency; printed dependencies are only `propext`, `Classical.choice`, and `Quot.sound` (the finite multiply lemma uses a subset).

Actual Lean4.35.0 inspection found that stock `bv_decide` adds a native Boolean-check axiom, extending the trusted base to the Lean compiler. A trial certificate-replay wrapper was abandoned and removed. The retained implementation instead uses direct output-bit extensionality, `simp`/`omega`, and kernel `decide`; it does not use `bv_decide` or `native_decide`. The requested full pack/decode corollary was deferred: all four output identities already quantify over every packed word, and the position permutation is separately proved. Root owns the actual-source binder, generated candidate checks and negative mutations; their status is separate from this local proof build.
