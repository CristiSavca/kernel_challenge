# V109 calibrated DFLASH proposal trees

V109 tests whether a request benefits from a broader or deeper draft tree at the same number of target verification rows. The current DFLASH selector already computes FP32 conditional scores between successive candidate positions. V109 builds two additional trees from those scores on the target tensor-parallel group’s root rank. One multiplies the scores by 0.5 before the row softmax. The other multiplies them by 2. Within each tree, the proposer fills the remaining verification slots in descending order of prefix probability. This can introduce a second-level branch that none of V101b’s adaptive native-spine trees contains.

The ordinary V101b choice serves a new request. After each eligible greedy verification, the updater walks the observed target-committed prefix against the two alternatives and the V101b choice for that step. It records the sign of each pairwise result in three bounded integer scores. A comparison in which both proposals still match at the observation boundary is censored when more verification capacity remains. Two confirmed wins from a neutral score can promote an alternative. If both qualify, the direct comparison between them decides first. Numeric scores follow the existing request ownership, including reorder and slot reuse.

No new proposal network or target call is required. Proposals consume the current draft scores. The committed prefix is read only during the existing commit step. The request carries twelve extra bytes of numeric feedback. The extra tree buffers on the tensor-parallel root use 16 KiB at 32 requests and width eight. The proposal kernel dispatches three blocks per eligible request and the feedback kernel evaluates two additional trees. Both reuse V101b’s kernel launches and its broadcast of the chosen tree. These costs may outweigh any acceptance gain.

## Offline results

| Synthetic target relative to the draft | V109 / V101b committed tokens per step |
| --- | ---: |
| Broader | 1.0060× |
| Matches draft calibration | 0.9987× |
| Sharper | 1.0048× |
| Alternates between broader and sharper | 1.0006× |

The table uses 32 constructed selector lattices, 64 paired requests per lattice and 48 steps per request. The same synthetic study compared promotion thresholds. It cannot establish prevalence in real requests or predict throughput. A separately selected favorable lattice yielded 1.0773× in its broad regime and 0.9803× in its calibrated regime. Local results support retaining this as a research candidate. End-to-end H200 latency, correctness, peak memory and score remain unmeasured. See [promotion sensitivity](promotion_study.json), [selected example and family oracle](replay.json) and [independent three-continuation example](independent_lattice.json).

- The CPU suite exercised 1,134 proposal trees, 378 selection/acceptance cases and 1,134 comparison updates across widths two through eight. It included exceptional scores, ties, 64-bit token IDs and padding. The actual Triton CPU interpreter checked 90 proposals, 30 selected trees and 90 comparison updates. [CPU results](cpu_check.json) and [interpreter results](interpreter.json).
- The worker-mode fixture exercised 128 combinations spanning root and peer ownership, graph eligibility and tensor shapes. The production request carriers passed 28 merge/filter cases, 168 shuffled relay resolutions and seven slot reuse resets. [Worker results](worker_modes.json) and [request state results](lifecycle.json).
- All 21 proposed, selected, feedback and opt-out kernel specializations compiled for SM90 with Triton 3.8. Offline assembly reported zero spills and at most 49 registers per thread among the tested specializations. [SM90 signatures and code hashes](sm90.json).
- The official full export is 232,031 bytes. Its SHA-256 is `c8fa4e84afde3e6a2cf0f62b1388a82db38c94debb3820eccc7522be24ad50e4`. Strict forward and reverse application and V101b-plus-delta reconstruction passed. See the [full patch](patch.diff), [three-file candidate change](delta.diff) and [flag, source and export validation](validation.json).

## Scope

The optional switch is `--speculative-dflash-temperature-tree`. It requires V101b’s greedy tree verification and adaptive spine, a supported graph-captured native selector and a single pipeline-parallel stage. Other paths use the existing proposal and verification behavior. This flag changes the proposal distribution only. The target still evaluates and verifies every selected row and decides every committed token. The patch introduces no prompt or output lookup, benchmark input, network call, or request-sharing mechanism. Files and launch flags pass the published structural rules. The judge owns final output, TTFT and memory checks.
