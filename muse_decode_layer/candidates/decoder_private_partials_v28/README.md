# Decoder private partials V28

Frozen experiment from exact public decoder `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.

Source: `submission.py`, SHA256 `d1b1bcf5c241710697bc0d2fb39836b1d7064c262bff7f820b564085e2136d9b`.

Replace only MAIN WGS split-unit vector atomics and payload clearing with immutable private boundary stores and an ordered FP32 owner sum. Explicit RN/FTZ addition preserves the public global reduction policy; arrival order changes and requires GPU numerical validation. Extra scratch is 20 MiB. Public weights, arithmetic, PDL, fused finalizer and small-token kernels are retained. The V26 serving component is only a reduction-code donor; its performance is not a forecast for this public vector-atomic parent.

`build.py` and `edits.json` provide deterministic generation and complete parent restoration. `manifest.json` pins the source/donor and scope. `test_cpu.py` executes actual source snippets on CPU and preserves its explicit GPU/progress limitations in `cpu_result.json`.

```sh
python3 muse_decode_layer/candidates/decoder_private_partials_v28/test_cpu.py --expected-sha d1b1bcf5c241710697bc0d2fb39836b1d7064c262bff7f820b564085e2136d9b --output /tmp/okc-v28-independent-cpu.json
```

Fixed screen and quantitative costs are in `tools/DECODER_V28_HYPOTHESIS_20261104.md`. Affected indices 4/5/8/9/10/11; controls 0/1/2/3/6/7. Aggregate gates use the official 46:16 weighted loads, not an unweighted case GM. Affected three-load speed GM ≥1.03, whole six-load speed GM ≥1.015, all case latency ratios ≤1.03, control three-load latency GM in [0.97,1.03], each against both exact-parent public/private receipts. All 13 tests, 12 benchmarks/profiles and integrity checks must pass.

CPU status: PASS. GPU/source review status: pending. No promotion or performance claim.
