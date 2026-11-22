# V120 phase-specific draft context projection

**November 18 dispatch correction.** The prompt-projection startup check in this frozen patch rejects the registered Muse logits processor. The earlier CPU fixture used the base class and missed that distinction. V120b enables the prompt optimization for the native model and tests both classes. [Correction and regression](../v120b_native_muse_prompt/README.md)

V120 combines V111’s long-prompt path with a tensor-parallel projection for decode verification and V104’s direct draft KV writes. On the eight-GPU configuration, each GPU projects 1,024 of the 8,192 output channels for the replicated verify rows. At 32 users and width eight, this reduces the projection from 171.80 to 21.47 GFLOP per GPU per pass. H200 throughput and the official four-load score are unmeasured. [Submission patch](patch.diff) · [Changes from V111](delta.diff) · [Export and flags](validation.json)

## Decode and prompt processing

All five target hidden-state taps enter each dot product at the original 40,960-feature width. The worker gathers the BF16 output channels in draft-group rank order, then applies the existing hidden RMS normalization once. It uses the target-confirmed lengths, positions and translated draft cache slots. The fused draft KV writer can commit the five layers’ normalized and rotated keys and their values in one kernel, saving five launches per eligible append. [Integrated decode fixture](verify_decode.py) · [Direct KV implementation and SM90 compilation](../v104_v64_direct_draft_kv/README.md)

V120 retains the resident 640 MiB weight for V111’s prompt path. During verify, each GPU references an 80 MiB view of that weight. During an eligible long prompt, each GPU uses the full weight only on its own sequence rows and retains V111’s smaller prompt gather. The high-load verify gather receives 3.5 MiB from peers per GPU. Its local result, rank-major receive buffer and contiguous gathered result can occupy 8.5 MiB together, 4.5 MiB above a single full output. These figures describe tensor sizes. The H200 memory allocator and communication library determine the observed peak. [Independent source and memory review](independent_review.md)

Startup binds the group inside the draft placement and checks that every member has the native two-dimensional BF16 projection on Hopper. The optimized helper runs only for the worker’s prefix-valid verification append. It completes the collective on the original inputs before a rank with hooks or an overridden projection takes its native path. This avoids a per-step host vote. The support contract assumes replicated checkpoint weights and replicated verification features within each draft group. Replacing a live weight can retain its original storage until worker reinitialization. [Asymmetric hook tests](independent_hooks.json)

## Validation

| Check | Result |
| --- | --- |
| Integrated TP2, TP4 and TP8 decode | 24 cases, including 256 verify rows and a full 40,960-wide reduction. BF16 normalized features, positions, commit lengths and physical slots matched the native path exactly. [Receipt](decode.json) |
| Integrated prompt processing | 48 ordinary and already-projected append cases, plus 69 V108 distributed prompt cases. Six empty worker calls also returned without a collective. [Append coverage](decode.json) · [Full prompt fixture](prefill.json) |
| Asymmetric instrumentation | All 52 outcomes matched in a separate two-rank Gloo run, including changed weights, hooks and mixed-dtype autocast. [Receipt](independent_hooks.json) |
| Direct KV storage | 32 BF16/FP8 cases matched 1,483,968 cache bytes. Setup and worker dispatch passed 62 additional cases. [Bytes](prefix_bytes.json) · [Dispatch](prefix_dispatch.json) |
| Export | 96,445 bytes, 20 permitted Python paths, 121-byte longest added line. The full patch applies and reverses on the judge pin. V111 plus the delta reproduces the official export. All 11 flags clear the fixed-pattern list. [Receipt](validation.json) |

The intended comparison is the 32-user private check against V64’s scored 20,547.4 tokens/s/GPU and 43.3 median decode tokens/s/user. The private check covers one load and one repetition. The judge must also confirm the first- and later-turn outputs, exact requested lengths, TTFT and peak memory. V64’s measured four-load score is 1.4119×. The 4× objective requires another 2.83× over V64. [Official result](../../runs/abd217baf0ed49fd.json) · [Serving rules](../../RULES.md)
