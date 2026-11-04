# V29: exact BF16 register repair in small-token SK2

Parent is the public decoder `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`. This isolated candidate changes only its SK2 compression/consumer path and dense extension namespace. Public attention, MoE, other dense kernels, compiler flags, tuner, routing and active submission remain unchanged.

The parent encodes 512 BF16 weights in 768 bytes using one matrix-wide eight-value exponent-high window, then applies out-of-window FP32 corrections in separate warps. V29 retains that primary encoding and unchanged `decode8`, but appends one 32-bit descriptor per unit to each stage. Mode 0 needs no repair. Mode 1 stores the exact exceptional BF16 bits and nine-bit fragment position; only its owning lane replaces the appropriate 16-bit half of one statically unrolled register word before MMA. Mode 2 loads an immutable 1024-byte exact BF16 sidecar for a unit containing at least two exceptions. No activation is read for correction.

Descriptor fields are **bits31:30 mode; single mode bits24:16 position and bits15:0 original BF16; raw mode bits29:0 sidecar index**. All primary unit addresses within a stage remain unchanged. Stage stride includes the aligned descriptor tail. Entire matrices use existing alternative backends when actual `772 * units + 1024 * raw_units >= 1024 * units`; this is an immutable weight-byte eligibility rule, not a test-shape or seed heuristic. All bit patterns, including zeros, subnormals, infinities and NaN payloads, have an exact representation or full fallback.

Compressed SK2 now has zero correction warps/planes. Its ring requires only the original regular consumer arrivals, with the same loads-before-handback positions, producer schedule, ring depth and geometry. Noncompressed router behavior is retained. Fused router four-way reduction, UV physical rows, BND actual occupancy queries and PDL waits/triggers retain their original logic. The descriptor and raw-weight reads complete before the existing handback; neither adds a producer/consumer barrier. Compile-time `COMP && CL==1` covers all generated compressed paths. The old macro `SK2_NESC=3` was inactive; actual parent correction count was `min(NSTAGE,4)`.

This changes floating-point grouping: original BF16 bits now enter MMA, replacing approximate MMA plus an FP32 correction sum. Exact operand restoration is not a proof of identical output rounding or official numerical tolerance. New header reads, branches, rare global weight reads and register live ranges can lose. No resource or performance forecast is asserted.

`build.py` verifies the exact parent hash and reverses every source edit byte-for-byte. `test_cpu.py` executes actual Python pack/stream helpers with a NumPy adapter and actual extracted C++ `decode8`, repair and raw-address expressions. It checks 70,784 units / 36,241,408 BF16 comparisons, all 65,536 bit patterns, all 512 positions, three wrong-owner/half/raw-index source mutations, 2,808 full stage/unit mappings, 17 generated template geometries and 1,120 ring interleavings. Two ring-model negative controls detect missing/extra handbacks. The model is not GPU memory-order or CUDA-compiler proof. No tensor MMA computation is emulated.

Run the frozen CPU suite with:

```sh
.venv-modal/bin/python muse_decode_layer/candidates/decoder_exact_patch_v29/test_cpu.py --expected-sha bbbc37f80aa3a5263dd0a66611bae62b2376c8b77ed67c14ba8ffd8db78e8a2c --output /tmp/v29-independent-cpu.json
```

Only root may issue one official private check after independent source GO and CPU reproduction. The fixed screen compares exact 97af public `7a8c4941741645ee` and private `b2a773e2d7194de8`: whole official six-load score ratio ≥1.025 against both; affected four weighted-load speed GM ≥1.04; every affected/control case normalized latency ≤1.03; control two-load latency GM in [0.97,1.03]. Loads use the official 46:16 sliding/full weights. Require clean 13 tests, 12 benchmarks and 12 profiles, exact provenance, and all expected compressed SK2 template instances renamed to `sk2_exact_patch_v29` in all eight affected cases. `COMP` is the eighth template argument. A compile/tuner fallback without activation is not a successful experiment. No repeat, nearby variant or promotion follows a failed fixed screen.

Preparation logs print actual per-matrix `(units, single_units, raw_units, requested_bytes)` when compressed. The proxy distribution used to prioritize this experiment is not judged-weight evidence. Published source attribution and licensing remain those of the exact parent.
