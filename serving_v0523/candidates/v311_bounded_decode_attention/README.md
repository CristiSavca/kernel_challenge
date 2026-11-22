# V311 reduces attention work during DFlash decode verification

V311 changes one file on V204, the build that scored 1.5487×. It keeps V204’s six flags. For short key spans, the host caps the partition grid using the largest number of key tiles the batch could contain. If a partition is removed, its first tile was already beyond the last live tile. At higher batch sizes, each merge block handles up to four adjacent output rows and computes the request boundaries once. The merge keeps each row’s original FP32 reduction order. These two changes adapt the V301 grid bound and the V236 grouped merge to V204’s ordinary verification path. [Full patch](patch.diff) · [Delta from scored V204](delta.diff) · [V204 result](../../runs/ad275bf67d9a4a0a.json)

| Requests in the verification batch | V204 work eliminated per GPU over 62 eligible layers |
| ---: | --- |
| 1 | 2,944 partition blocks, with 1,064,960 fewer scratch bytes per sliding-window call |
| 4 | 15,872 merge blocks |
| 16 | 95,232 merge blocks |
| 32 | 190,464 merge blocks |

Both versions issue two kernels per eligible layer. Each surviving grouped merge block can do more work. The change in block count alone cannot establish a latency gain. The CPU replay checked 61,727 live partition intersections across 3,000 geometries and 40 parent/candidate comparisons through the full serving attention entry. Explicit ragged layouts continue to use native attention. [Dispatch and allocation replay](dispatch.json)

The Triton interpreter matched all 1,946,880 FP32 partial and log-sum-exp words and all 236,544 BF16 output words on 22 fixtures. An independent dense FP64 calculation covered 192,768 live components, with a maximum absolute difference of 0.00920. The 50 offline Hopper builds had zero reported spill bytes in the aligned compile matrix. Serving latency, token agreement and whole-server peak memory still require a physical run. [Interpreter](interpreter.json) · [Hopper compilation](sm90.json)

The official export uses 264,300 bytes over 42 allowed paths. Its delta changes only the verification attention implementation. Both exports apply and reverse on the published source, and the six flags match the V204 scored receipt. [Validation](validation.json) · [Flags](flags.json)
