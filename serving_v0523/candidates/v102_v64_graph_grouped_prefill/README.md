# V102 graph-captured expert prefill on measured V64

V102 tests the cost of running the large-prefill experts outside V64’s CUDA graph. On an eligible chunk, V64 reads a route count back to the CPU in each of 62 layers to allocate its workspaces. V102 keeps the plan, grouped Humming matrix multiplications and expert normalization on the device. Its full patch is 62,165 bytes over 12 permitted sources, SHA-256 `ee8bf58ecb98548d236b54f774e948139fff0659c3479b6586b3243142daa5b6`. [Full patch](patch.diff) · [Six-file change from V64](delta.diff) · [Official export checks](validation.json)

The ordinary model forward sets sequence parallelism for `EXTEND`, and the expert call receives the gathered rows. The new branch requires this mode, BF16 contiguous input, compatible Humming kernels and at least 256 routes per global expert. For Muse top-eight over 256 experts, eligibility begins at 8,192 rows. Each token’s local routes stay in the same wave. The planner keeps every selected expert, including skewed and repeated expert assignments. The two GEMMs retain full-K accumulation, the activation retains the original BF16 scaling and rounding, and each output token combines slots in its original order. Target verification and decode use V64’s dispatch. Specializing the shared finalizer with expert scaling disabled reproduces V64’s per-token arithmetic exactly. [V98 wave design](../v98_graph_grouped_prefill/README.md) · [Validation](validation.json)

Use the four overrides from the scored V64 run.

```text
--flag swa-full-tokens-ratio=0.25
--flag moe-runner-backend=humming
--flag speculative-moe-runner-backend=marlin
--flag enable-layernorm-sp
```

The V64 server used 16,384-token chunks. At 32 users its reported first-turn p90 was 110 seconds. V102 replaces up to 62 host synchronizations per eligible chunk. At 16,384 rows the capture includes nine possible waves, including launches for empty waves. Its two BF16 arenas total 384 MiB and the output occupies 128 MiB. V64 also used about 384 MiB of eager arenas at balanced routing. Graph allocator reservations, sorting workspaces, routing skew and launch overhead can change the outcome. V64’s scored peak was 1.9 GB over stock against the 2 GB limit. A matched H200 check must establish throughput, peak, TTFT and output agreement. [V64 scored run](../../runs/abd217baf0ed49fd.json)

The production dispatch selected waves for full-size 8,192- and 16,384-row inputs under eligible capture and eager conditions. Eleven dispatch cases and six two-rank Gloo startup votes passed. The numerical rerun covered 50 adapter cases, 8,240 BF16 outputs and 151 empty waves. The planner covered 1,168,433 live slot mappings and wrote every token once. V64’s 54 sequence-parallel integration cases passed. The three device source files match V98 exactly, covering its 13 prior Hopper Triton compilations and scaled-SiLU CUDA compilation with zero spills. Physical execution of V102 remains unmeasured. [CPU results](cpu_result.json) · [Dispatch fixture](dispatch_check.py) · [Triton compilation](../v98_graph_grouped_prefill/sm90.json) · [CUDA compilation](../v98_graph_grouped_prefill/sm90_nvcc.log)

The November 19 runner selects V102 at 16 users while the exact V93 check is pending and the scored V64 parent passes. This isolates expert prefill on V64’s decoder. A passing V93 selects the integrated V98 comparison, subject to its shared dependencies. [Private window](../../runs/nov19_decode_runner.launch.json)
