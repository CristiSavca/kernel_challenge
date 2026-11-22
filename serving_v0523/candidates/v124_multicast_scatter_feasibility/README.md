# V124 multicast token reduce-scatter feasibility

The proposed custom SM90 kernel cannot accelerate V120b's full 16,320-token prompt chunks under the scored V64 communication settings. A BF16 expert sum at that size is 127.5 MiB per rank. The attention output sum is 255 MiB. The pinned custom all-reduce selects neither. The often-quoted 128 MiB entry is the unconstrained tuning range. SGLang clips that range to its configured 16 MiB pull workspace before selecting the collective. V120b inherits these files byte-for-byte from the pinned source. [Executable dispatch and size inventory](dispatch.py) · [Recorded output](dispatch.json)

The scored V64 server logged 18.01 MiB of symmetric memory. Its allocation is 16 MiB of pull storage, 16 push slots of 128 KiB and 8 KiB of pull semaphores. The 8 MiB local buffer logged separately contains registration bookkeeping. The live group constructor supplies only its group and device. The test reads the immutable `297e54fd` Git objects and checks the V120b patch hash before executing the real Hopper configuration clipping and selection code. [V64 official server log](../../runs/abd217baf0ed49fd.log) · [V120b export](../v120b_native_muse_prompt/validation.json)

## Eligible geometry

For contiguous BF16 input split into equal rows over eight ranks, these are the inclusive intervals selected by the native custom all-reduce tuning. The proposed scatter also needs the custom communicator to win over higher-priority backends, valid multicast registration and no algorithm override. The graph column uses the registered-input graph heuristic. A captured backend that chooses the eager heuristic follows the eager column. Source replay checked all 8,192 combinations of row count, width and heuristic up to 16,384 rows.

| Projection output width | Eager input rows | Graph input rows | Largest eligible payload |
| ---: | ---: | ---: | ---: |
| 4,096 expert channels | 24–2,048 | 64–2,048 | 16 MiB |
| 8,192 attention channels | 16–1,024 | 32–1,024 | 16 MiB |

The medium-size mechanism is real. At 1,024 padded input rows the two outputs are 8 and 16 MiB. An eligible direct kernel could stage each input once, issue the incumbent `ld_multimem_16B` for its rank's segment, and write only its reduced rows locally. It could remove the rebroadcast and full output copy. Correct reuse would have to preserve both peer barriers and release the PDL successor only after the final peer rendezvous. The pinned NVLink primitive has internal reduce-scatter code but exports only all-gather through its Python wrapper. Its direct pull form requires a multicast alias of the producing allocation. The existing generic all-reduce instead stages arbitrary input through the bounded registered workspace.

V64’s retained output contains two startup prefill sizes, 151 and 1. It does not report scored graph bucket frequencies by load. Later-turn prompts and residual chunks can enter this interval, while the largest first-turn chunk and its 8,192-row predecessor are outside it for both projections. There is no measured fraction of prefill latency to attribute to this medium-size optimization. The default `NCCL_NVLS_ENABLE` setting also does not prove NCCL executed the same reduction instruction or accumulation order for an individual large all-reduce. Tiling it into custom reductions would change the incumbent algorithm and introduce extra barriers. [Logged request geometry](dispatch.json) · [V64 timestamps](../../runs/abd217baf0ed49fd.log)

## Historical capacity change

V48b raised the workspace on V41 and logged 130.01 MiB, an additional 112 MiB per rank. Its four-load score was 1.1037×, versus 1.0934× for V41. At 16 and 32 users the respective frontier ratios changed from 1.1046 to 1.1084 and from 1.1651 to 1.1746. These separately measured runs changed the underlying all-reduce capacity on an earlier model path. They do not measure a token reduce-scatter or establish memory headroom for V120b. V64 reported +1.9 GB/GPU against the +2 GB limit. V120b has no measured peak. No workspace increase or runtime change was made for V124. No compiler VM or challenge allocation was used. [V41 measurement](../../runs/67e2364305c041ae.json) · [V48b measurement](../../runs/8ebed2bced8942ff.json) · [V64 measurement](../../runs/abd217baf0ed49fd.json)
