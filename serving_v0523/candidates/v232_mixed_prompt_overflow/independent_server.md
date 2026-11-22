# V232 server resolution and graph capacity review

Seventeen scenarios ran the native msgspec CLI, `argparse` actions, graph-priority resolver, DFLASH checkpoint parser, both prefill buffer ceiling accessors and target runner’s aligned shape selection. The tests read 18 source files directly from the frozen V232 tree and verify any materialized copies. The platform fixture supplies CUDA and a checkpoint config with native width eight. A width-four override also runs through the same production DFLASH resolver. [Executable replay](independent_server.py) · [Frozen result](independent_server.json)

| Public-size configuration | Extra captured target token keys |
|---|---|
| Existing 16,384-token limits or either 16,640 override alone | None |
| Both 16,640 limits with width eight | 16,392, 16,400, 16,416, 16,448, 16,512, 16,640 |
| Existing buffer provider returns 16,384 | None |
| Both limits 16,402 with alignment 16 | 16,400 |
| Explicit prefill shape list, full prefill, disabled decode, draft runner, different algorithm or disabled mixed prefill | None |

The archived scored server reports a 16,384-row chunk, 16,384 maximum prefill tokens and 32 running requests. The DFLASH resolver supplies 48 requests only if the caller leaves that field unset. Prefill phase `max_bs` names a **token** cap, while the runner's `max_bs` is the size of its **request** table. Thus a 31-tail cohort beside a full prompt can fit the 16,640 graph with 16,632 real rows and 32 real requests. A 32-tail cohort with that prompt would require 33 requests under a 32-slot request limit. [Scored server log](../../runs/abd217baf0ed49fd.log) · [Resolution cases](independent_server.json)

The original proposal added mixed keys before post-publication buffer providers and runner alignment were known. A 16,384-row provider left all six larger shapes eligible for startup capture while later admission rejected them. The frozen implementation selects generated shapes in target runner initialization, applies the real alignment, and filters each key against the effective limit before warmup/capture. Explicit user-provided grids retain their native semantics. The zero-generated-key provider result and the one-key aligned case verify both edges. [Source delta](delta.diff) · [Capture scenarios](capture.json)

Native slot allocation is based on the live prompt rows and `tails × width` for the verified proposals. The mixed worker concatenates the actual prompt and reserved tail slot arrays. At the Breakable attention boundary, the original callback narrows cache locations to the live row count and zeroes padded output rows. The scheduler and worker verify the request-table and padded-body capacities separately and require all three configured graph keys to exist. The standard proposal/standalone verification path remains reachable when admission declines. [Scheduler exercise](scheduler.json) · [Worker and physical-slot exercise](bridge.json)

Peak CUDA graph workspace, cache headroom and the cost of capturing the six new bodies still require the official H200 measurements. Increasing the physical limit adds up to 256 token rows of scratch to relevant buffers even though the ordinary prompt chunk stays at 16,384.
