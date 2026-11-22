# Serving-track results

**V204 earned the highest official score in this repository, 1.5487× stock SGLang.** The organizer measured the patch on eight H200 GPUs at 1, 4, 16 and 32 concurrent users. It passed the correctness and latency checks and used 1.9 GB more peak memory per GPU than stock, within the 2 GB limit. The official serving leaderboard was still private when these results were checked on November 22, 2026. Final placement is unknown. [Official rules](https://openkernelchallenge.dev/rules) · [V204 patch](candidates/v204_mixed_prompt_weight_pipeline/patch.diff) · [Machine-readable results](results.json)

## Scored runs

| Candidate | Official score |
| --- | ---: |
| V204 | 1.5487× |
| V64 | 1.4119× |
| V44 | 1.1847× |
| V48b | 1.1037× |
| V41 | 1.0934× |
| V40 | 1.0512× |

Six scored runs completed. An earlier V44 attempt timed out and the organizer refunded it. The score uses the geometric mean of throughput relative to stock at stock’s per-user speed across all four loads. The table below gives V204’s raw end-to-end measurements alongside those four scoring ratios. [Scoring definition](RULES.md#score) · [Full run inventory](results.json)

| Users | Measured tokens/s/GPU | End-to-end tokens/s/user | Throughput ratio at stock’s per-user speed |
| ---: | ---: | ---: | ---: |
| 1 | 8,146.0 | 176.2 | 1.4126× |
| 4 | 12,973.1 | 75.3 | 1.4396× |
| 16 | 18,472.5 | 26.4 | 1.6802× |
| 32 | 21,302.6 | 15.4 | 1.6836× |

## Single-load checks

Of 25 private checks, 21 passed, three crashed and one failed correctness. The final three all passed after submissions closed at 08:00 UTC on November 20. Each received an indicative comparison with stock at its tested load. Only a scored run evaluates the full four-load curve. [Run inventory and patch hashes](results.json)

| Candidate | Users | Finished Nov 20 (UTC) | Tokens/s/GPU | End-to-end tokens/s/user | Indicative ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| V129 | 1 | 11:49 | 4,299.0 | 93.4 | 0.432× |
| V141 | 32 | 13:36 | 15,646.3 | 11.6 | 1.239× |
| V154c | 16 | 14:59 | 19,898.7 | 28.4 | 1.613× |

## Final local candidate

V312 combines long-prompt expert routing with a grouped merge for decode attention on the V204 codebase. Its local tests matched 376,268 routed keys and 236,544 BF16 attention output words against their respective references. The patch was never submitted and has no official serving score or eight-H200 timing. [V312 evidence](candidates/v312_bounded_prefill_and_decode/README.md) · [Patch](candidates/v312_bounded_prefill_and_decode/patch.diff) · [Launch flags](candidates/v312_bounded_prefill_and_decode/flags)

The JSON inventory covers all 32 organizer-listed runs. It includes their job IDs, patch hashes, flag overrides and aggregate measurements. File paths in the inventory are relative to this directory. Raw traces and private run records stay in the local workspace. The owner can retrieve an individual receipt using `okc serve status <job-id> --json` while organizer access remains available.
