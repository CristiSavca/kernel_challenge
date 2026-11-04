# Source-bound packed-decoder proof

This is our independent, paper-inspired first implementation of machine-checked
GPU arithmetic reasoning. It is **not SF Tensor's unpublished Lean model, a PTX
interpreter, or a certificate for the complete CUDA kernel**. The upstream
discovery and modeling limits are recorded in
[`../tools/PTX_LEAN_UPSTREAM_ASSESSMENT_20261104.md`](../tools/PTX_LEAN_UPSTREAM_ASSESSMENT_20261104.md).

The useful first property is exact: for **every 32-bit input word**, the current
MoE decoder's four `OKC_MULDEC` output words equal an independent bit-placement
specification. This includes the fourth output's multiply trick with unsigned
32-bit wraparound. The 32 source bit positions are proved to be a permutation of
0 through 31. Floats are not approximated by reals here: the claim is about raw
bits, before any floating-point interpretation.

`OKC/Dq4.lean` supplies the specification and proofs. The source checker in
`../tools/verify_ptx.py` parses the candidate's Python AST without importing it,
extracts a narrowly recognized unsigned32 `dq4` branch, and generates separate
Lean definitions using the actual constants. Lean must establish equality with
the reviewed theorem; a changed constant cannot inherit an old certificate.
The packing functions and position assignments must match their reviewed ASTs
in immutable candidate `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
AST identity is correspondence evidence, not a proof of Python/Torch execution.

Each verification builds in a fresh temporary directory, checks reviewed proof
file and toolchain executable hashes, and audits every expected theorem's
axioms. Only Lean's standard `propext`, `Classical.choice` and `Quot.sound` are
accepted. No `sorry`, custom axiom, `native_decide`, or native bitvector-check
axiom is accepted. Temporary proof builds are removed after completion. The
reviewed contract is `proof-contract.json`; changing it requires source review.

## Running it

The current Mac installation uses official Lean 4.35.0 for Darwin arm64 under
`.toolchains/lean-4.35.0-darwin_aarch64` at the repository root. Its release
archive was checked against SHA256
`c0c6f34f35ba1dcc5c4a5fd106fefab9eb5e8469712624dd2c868d51b63177c2`
and size 570422530 bytes before extraction. No global Lean/profile setup is
required. A different platform needs a separately reviewed toolchain entry;
the checker deliberately does not silently download or switch toolchains.

From the repository root:

```sh
.venv-modal/bin/python tools/verify_ptx.py muse_moe_w4a16/submission.py \
  --report experiments/formal/current-muldec.json
.venv-modal/bin/python tools/test_verify_ptx.py
.venv-modal/bin/python tools/test_verify_ptx.py --integration
.venv-modal/bin/python tools/test_experiments_formal_gate.py
```

The integration test runs real Lean against the incumbent and rejects three
deliberately broken decoders: rotation 3→4, an extra output-mask bit, and
multiplier 4228→4224. Separate tests reject packing-map changes, unsupported
expressions, duplicate assignments, incomplete axiom logs and native/custom
proof dependencies. These mutants are temporary files, never official uploads.

For a future authorized experiment, add `--verify-ptx` to the existing
`tools/experiments.py check` or `submit` command. It freezes the candidate, runs
Lean on those bytes before any upload, then attaches the exact proof-report
hash to the job receipt. A failed proof prevents the remote call. A transient
status error still leaves the accepted job's receipt and proof intact. This
flag is explicit because most other kernel families do not contain the modeled
fragment. It does not waive the private-check-before-score requirement.

## What this protects, and what it leaves open

It protects the five recognized arithmetic expressions and their independent
bit-layout contract. The source-to-Lean translation, packing correspondence and
specification remain reviewed trusted components. Ordinary builtin unsigned32
and funnel-shift semantics, immutable packing maps, and distinct output
registers are assumptions. The checker rejects direct map mutations and macros
overriding modeled identifiers, and applies C++ line splicing before stripping
comments. It does not certify the entire Python/C++ environment or indirect
mutations through aliases.

The proof does not cover tiny/non-MULDEC dispatch, tensor coordinates, scales,
BF16 rounding, MMA, out-of-bounds accesses, races, asynchronous barriers,
termination, compiled PTX/SASS, graph concurrency, or model outputs. The
pack/decode tensor pipeline is not formally verified. Official numerical,
concurrency and performance checks remain necessary.

This tool has zero runtime speed effect by itself. Its purpose is to reject
wrong bit manipulations before spending a GPU check, and to give future decoder
rewrites a precise contract. A new expression family requires a new reviewed
translation/proof; unsupported syntax is rejected rather than guessed.
