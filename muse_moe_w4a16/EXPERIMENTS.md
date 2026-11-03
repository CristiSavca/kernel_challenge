# Expert routing experiment — 2026-10-24

Private check `1cc96c4a472b470a` tests `candidates/local_routing_v1.py`.
The immutable upload and full status are recorded under `../experiments/`.
Private H200 validation passed correctness and static checks. Measured speedup:
**1.0526377681×**, 95% interval **[1.0368482016, 1.0684273346]** relative to SGLang.
Scored submission sent November 3: `8f1f69d433364789`; confirmed queued in
mode `run`, result pending. It uses the same frozen source as the passing check.

This candidate retains SGLang's NVFP4 Marlin path, FP32 accumulation, BF16
intermediates, SwiGLU and per-expert normalized weighted sum. It changes only
the sorting bucket count and the shape-based tile heuristic:

1. Map global expert IDs to local IDs exactly as stock does, then align using
   the local count 32. The alignment wrapper adds its own bucket for -1.
   This avoids 224 impossible expert IDs and enables the existing native
   small-bucket alignment path.
2. Estimate routes per local expert using the global expert count 256.
   Non-local routes are not executed by this rank. At 128 tokens, block M
   changes from 48 to 8; at 256/257 tokens it changes from 64 to 16. A hot
   expert still receives as many complete tiles as its actual routes need.

The pinned wrapper/CUDA implementation was inspected to verify the sentinel
contract, and local syntax compilation passed. The official private H200 check
also passed numerical correctness and established the measured gain above.

If the combined change regresses, test a count-only control before changing
the math. Do not pre-dequantize the persistent NVFP4 weight bank.

The equivalent serving change is staged in the separate pinned checkout and
exported to `../serving/patch.diff`. It is an unvalidated draft and cannot be
checked by the official serving service until that track opens November 5.
