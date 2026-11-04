# Serving track: current rules and concrete next work

Read-only verification at **2026-11-04 14:11 UTC**. **Preparing a narrow serving integration is a concrete next avenue toward the prize; it does not replace the remaining kernel effort or establish a serving speedup.** No submission, initialization, checkout download, GPU allocation, payment, rules acceptance or source edit was performed. Raw public responses, hashes and CLI observations are in [the evidence record](../experiments/iterations/20261104-serving-current-readonly.json).

| Item | Current official evidence |
|---|---|
| Kernel cutoff | November 4, 11:59 p.m. Pacific / November 5, 2:59 a.m. Eastern. API exclusive close is November 5 at midnight Pacific. |
| Serving opening | November 10, 9 a.m. Pacific / noon Eastern. API currently says `open:false`. |
| Serving cutoff | November 19, 11:59 p.m. Pacific / November 20, 2:59 a.m. Eastern. API exclusive close is November 20 at midnight Pacific. |
| Limits | One scored run and three checks daily. Checks use concurrency 16, one repetition, two conversations per slot, with a 35-minute deadline. Scored loads are concurrency 1, 4, 16 and 32 on an 8×H200 node. |
| Patch | Base SGLang `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`; source/JSON under `python/sglang/`, excluding `srt/model_loader/`; **409,600 bytes maximum**, added lines ≤400 characters. |
| Queue/board | Serving priority group is still `null`; authenticated CLI also returns `priority:null, serving_priority:null`. Public serving board returns `hidden:true, rows:[]`. The global queue sample is 546 pending / 137 running kernels; it does not disclose a serving backlog. |

The [live serving API](https://openkernelchallenge.dev/api/v1/serving), [serving board API](https://openkernelchallenge.dev/api/v1/serving/leaderboard) and [queue API](https://openkernelchallenge.dev/api/v1/queue) support those values. The empty CLI board is not evidence of zero competitors. Own serving jobs returned `[]`.

**Eligibility and prizes.** The current official terms explicitly allow either track or both. Top-50 kernel placement gives serving queue priority, not entry eligibility; the CLI says that priority applies daily to scored and check runs, after final kernel runs finish, including cutoff ties. General entry is individual, age 18+, one account, subject to jurisdiction and sponsor-affiliation exclusions; this review does not verify the user's personal eligibility. The pool is **$100,000 total**, with **$50,000 plus an H200 for first**, then $30,000/$10,000/$3,000/$2,000 and $1,000 each for sixth–tenth. Top-25 serving entries receive at least three held-out reruns, with median scores determining final placement subject to review. The current terms now agree with the November 19 deadline, resolving the older November 18 discrepancy. [Official terms, sections 3–6 and 8–9](https://openkernelchallenge.dev/terms.pdf).

The November 3 update says the serving board stays private until completion and SGLang maintainers' assessment of whether a patch could be merged affects final selection. Serving must retain the model, required precision, all routes/norms and DFlash weights; outputs must agree with stock, requests return 220 tokens, and time to first token cannot exceed twice stock at a load that improves the score. These are end-to-end conditions, not consequences of a passing isolated kernel. [Current rules and update text](https://openkernelchallenge.dev/llms.txt).

**The workspace already has a starting point.** Contrary to the initial task premise, `serving/sglang` exists at the exact API base. It has one dirty file, `python/sglang/srt/layers/muse_spark_fp4_moe.py`, and an existing **1,549-byte patch** (`4a861b0a…`). That patch adjusts route-count-based Marlin tile selection and local expert alignment. It was preserved, not judged correct or fast here. Local `serving/RULES.md` is stale: it says 720 KB, whereas the current API says 400 KB.

The current decoder source is **749,201 bytes** (`97af3994…`), so a wholesale copy cannot fit the patch cap even before diff overhead. The current MoE file is **204,388 bytes** (`247888ac…`); a selected component could fit, but needs substantial integration review and removal of unused experiment/tuning machinery for maintainability. Its isolated score is not an end-to-end forecast.

A bounded next task is to map one serving MoE fast path against the existing `_forward_marlin`/post-expert-norm boundary, retaining the stock path for unsupported regimes. Specifically establish:

1. How to repack after weight loading without changing registered parameters or Marlin's gate/up global-scale normalization semantics. The current serving code explicitly relies on post-expert RMSNorm to cancel its scale ratio; blindly feeding transformed Marlin buffers to our raw-NVFP4 entry would be unsafe.
2. Which real token batches, prefill/verify modes and expert mappings are supported, with all TP/EP communication and remote-route behavior preserved. A load level is not a safe dispatch shortcut.
3. How prepared buffers, scratch, counters and CUDA graphs are owned across layers and overlapping requests. Existing kernel notes explicitly limit validation to serial prepared-state reuse. Also account for persistent repacked-weight memory across the model.
4. A compact, reviewable patch plus an end-to-end validation plan ready before November 10, beginning with the existing small patch's correctness audit. Actual serving correctness and speed remain pending until official checks open.

This work has a direct connection to the prize-bearing track and a known integration surface. It is therefore more concrete than indefinitely continuing weak isolated parameter bets. A focused remaining kernel hypothesis may still be worthwhile before tonight's cutoff; kernel improvements can inform the later component integration. No winning placement or throughput gain is predicted.
