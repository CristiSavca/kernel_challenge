# V174 multirow projection replay

The final V174 source produced the reference BF16 gate activations and down outputs for 54 projection pairs. Six additional tests used cancellation to distinguish ascending full-K FP32 accumulation from a reassociated reduction. They preserved the expected nonzero down result, 5.4424162954092026e−9, across 49 routed slots per test. The reference uses independent Humming weight unpacking and checks the rounded intermediate before checking the down projection. [Numerical results](independent_warp.json)

The fixture executes the production unpadded planner, the producer and each consumer as CPU coroutines. It covered empty lists, 1–8 rows, the W13 16-row seam, the W2 24- and 32-row seams, multiple tile rotations, mixed per-projection fallback and a nine-route case at width 4,096. Every plan ends immediately before a protected page. Its unused slots retained 136,917 positive, in-range destination IDs across three asynchronous schedules. All 2,515,968 live BF16 elements matched, and 3,381,504 per-phase checks preserved untouched values. The 17–24 row W2 branch instantiates three MMA groups. Its final group therefore always has at least one live row before the epilogue subtracts the group offset. [Executable fixture](independent_warp.py)

The simulated copy engine counted 22,494 tensor transfers and 74,334 activation-row transfers, totaling 122,681,856 bytes. Each count independently matched the live row total, K slabs, output bands and number of weight planes. Both kernels use eight consumer warps and one producer. Each consumer warp returns one arrival per slab, with the phase advancing across row tiles. The two W13 stages require 45,600 bytes including the barriers. The W2 stages require 35,872 bytes. Smaller lists retain the maximum shared allocation per CTA, and W2 has two stages compared with three in V173’s small-list path. Hopper run time and occupancy need hardware measurement.

The complete tested patch is `7dcbcb538043c14a0c525a9a419c1d92b25dfa23666faae109b82b4df3cd1374`. Run the fixture with the repository NumPy environment and a checkout reconstructed from the frozen patch.

```sh
.venv-modal/bin/python -B serving_v0523/candidates/v174_multirow_expert_tma/independent_warp.py /path/to/reconstructed/sglang
```
