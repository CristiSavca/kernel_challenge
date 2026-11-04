# Public source scan for the decoder goal

November 4, 2026. Bounded primary GitHub metadata/tree/README comparison; no external code execution, model weights or GPU use. The current public decoder is `97af3994…179c2`, 2.181183x. The saved 21:51 UTC board leader is 2.576163x. A repository claim or local timing is not an official score reproduced by us.

| Repository and checked HEAD | Date | Finding relevant to this incumbent |
|---|---|---|
| [vjeux/okc-2026](https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393) | Oct28 | Existing pinned donor; no new HEAD mechanism. |
| [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0) | Oct29 | Existing reviewed implementation; unchanged. |
| [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge/tree/890809cee421500b00126d084e81b338ef9181ab) | Nov2 | Existing pinned source; prefill is separately relevant to TTFT but does not improve the four-board overall score. |
| [zimo0110/muse-spark-open-kernel-challenge](https://github.com/zimo0110/muse-spark-open-kernel-challenge/tree/fe1c70bb3a1348096ebe6c9f6d93b960fccfad09) | Nov4 06:30UTC | MoE README claims 2.382249x (jobc913470f59834b10), close to our2.373691x. Newer text includes negative fixed-N32, separate-epilogue and private-partial experiments. Forwarded to the MoE owner; this is not a stronger released whole-decoder result. |
| [AutoclickerI/muse-spark-open-kernels](https://github.com/AutoclickerI/muse-spark-open-kernels/tree/85aed1fb1365877b615542dd44980646816d6b26) | Nov3 | Readable final decoder sources exist; README claims decoder1.889x, attention1.670x and MoE2.032x, below the current corresponding incumbents. Source organization is useful but no validated wholesale replacement identified. |
| [trxphxle/okc-2026-muse-kernels](https://github.com/trxphxle/okc-2026-muse-kernels/tree/72db638fa2718bc0ea5645fdd5c52a5511594579) | Oct31 | Claims decoder2.1756x, attention2.5673x, MoE2.3249x. Its lower-bit dense/deeper-pipeline negatives are constraints, not proof that all alternate layouts fail. No measured stronger replacement identified. |

The anencore94 fork's newer push metadata still points to the exact vjeux f35a41 HEAD; it is not a new kernel. The queried `ZhimaoL/sglang` matching refs under `heads/okc` were empty; that only establishes absence under that prefix. Parent already contains the relevant m64n72 FP8 PV / shared QK attention mechanisms, so the previously reported N72 idea is not new here. Root independently checked Rohan's018b7 HEAD and public leader-account repositories; see `experiments/iterations/20261104-number-one-public-discovery.json` for that separate scope.

No repository found in this bounded pass supplied verified public code that closes the 18.1% decoder score gap. This is a scoped search result, not a claim that stronger public code does not exist. Licenses/provenance must be rechecked before any new code import; this scan imports none.

The actionable next implementation is therefore the independently proposed exact BF16 register-repair codec, recorded in `DECODER_WINDOW_V29_HYPOTHESIS_20261104.md`. It removes an entire correction execution path while keeping the existing cheap primary decode. Its success is unknown until the fixed official whole-decoder check. Public source announcements and prior failed local alternatives do not substitute for that test.
