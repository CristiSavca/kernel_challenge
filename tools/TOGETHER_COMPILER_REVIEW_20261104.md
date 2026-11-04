# Together Compiler: primary-source review, 2026-11-04

**Accept the measurement and search-design lessons; no drop-in TKCC implementation is publicly established.** This is relevant to our serving work, but does not establish a new Muse/H200 kernel speedup or justify changing the frozen V26 experiment.

All three user short links were resolved with read-only HTTP requests:

| Short link | Verified destination |
| --- | --- |
| `https://t.co/FYRuKtXs7E` | [Together Compiler: Continual Compilation for AI Inference](https://www.alphaxiv.org/abs/2611.together-compiler-continual-ai-inference) |
| `https://t.co/0Zj8omrGTA` | [Compiler release article](https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference) |
| `https://t.co/W9pRerBCP7` | [Polymorphic Inference roadmap](https://www.together.ai/blog/polymorphic-inference) |

The paper is dated **November 4, 2026**, authored by Reyna Abhyankar and Dylan Lim (equal contribution), Drew Wadsworth, Venkata Hasith Vattikuti, YuTong, Julia Zhang, Jung Hwan Heo, Taeksang Kim, Kesavan Ramakrishnan, Evangelene Stanley, Xinyu Fang, Daniel Y. Fu and Simran Arora, Together AI Research. The landing page labels it alphaXiv Exclusive; its metadata has `arxivId:null`. A BibTeX publisher label is insufficient to invent a numeric arXiv identifier. The 38-page [PDF](https://www.alphaxiv.org/abs/2611.together-compiler-continual-ai-inference.pdf) was read directly, SHA-256 `c1ce4a8a7a3772f31d7135626ad443ae2d1bce844f8fab25213403097698a3ef`.

## What the evidence establishes

The system keeps weights loaded, hot-reloads forward implementations, recaptures graphs and restores the baseline afterward. A smaller-layer engine screens candidates before a higher-fidelity engine. Its cost model distinguishes hardware estimates, measured primitives and measured forward passes. The paper reports 1,354 versus 168 candidates for two tiers versus one slow tier, with a 7.2% better final solution in that experiment. Appendix A uses 24-hour optimization runs with twice the serving replica resources. These are vendor-reported measurements, not our independent reproduction. [Paper §§4.2–4.3, Appendix A](https://www.alphaxiv.org/abs/2611.together-compiler-continual-ai-inference.pdf).

The release article supplies a useful practical picture: many initialization steps can be amortized, while graph capture remains necessary. Its startup table is one workload, not a universal 467-to-9-second promise. It describes both full-engine and low-latency results; the latter can also change parallelism. Thus a TPS headline does not isolate kernel code or predict TTFT. [Release article](https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference).

The roadmap says limited beta is available now and self-service is future work. Its adaptive speculation, heterogeneous routing and model/harness post-training discussion is a broader research agenda, not a downloadable implementation of all these features. The public paper and articles do not link a TKCC implementation. Bounded GitHub repository searches for `org:togethercomputer compiler` and `org:togethercomputer tkcc` found no corresponding compiler source repository; the former returned an unrelated wheel repository. Consequently there is **no verified TKCC source commit or code license to import**. This is a scoped search result, not proof that no public copy exists anywhere. [Roadmap](https://www.together.ai/blog/polymorphic-inference).

Important scope corrections from the paper: the 13–38% change in active-expert count and the 1.3× engine slowdown are **different examples**; the latter concerns a quadratic sparse-attention top-k path. Scientific-model bit-exactness results do not certify every LLM kernel. The H100 megakernel comparison uses Llama-1B, while frontier results use other models/platforms; no Muse-on-H200 result is supplied. [Paper §§3, 5 and Appendix C](https://www.alphaxiv.org/abs/2611.together-compiler-continual-ai-inference.pdf).

## Apply these ideas to our actual work

| Decision | Concrete application to pinned SGLang/Muse |
| --- | --- |
| **Accept: two validation levels.** | Continue using actual GPU component checks for compilation, numerical failures and gross regressions. A passing component becomes eligible for representative full-serving validation; it does not prove eight-rank correctness, TTFT, throughput or graph/workspace lifetime. V25/V26 already demonstrate why these labels matter. |
| **Accept: workload-sensitive evidence.** | For a future serving comparison, retain actual batch size, expert counts/skew, empty-rank incidence, request lengths and prefix-cache state. Test the unchanged computation on ordinary and adverse route distributions. Never specialize to recorded IDs/seeds or remove routed work. |
| **Accept: fewer research constraints.** | Give an implementer the actual source, correctness contract, resource budget and measured failures. Use a short note and exact source/result receipts; avoid building a new knowledge graph or requiring a predetermined fusion/megakernel strategy. Preserve independent numerical/concurrency review. The paper's context ablations support reducing prescription, not removing verification. |
| **Reject now: build a two-engine service.** | Our remaining bounded free-credit environment cannot reproduce a 24-hour, twice-replica optimization campaign or hold two full eight-H200 engines. No model download, beta enrollment, provider allocation or persistent-engine implementation is authorized by this review. |
| **Reject: transfer isolated tricks automatically.** | Their PDL, graph and prefetch examples do not reopen our closed prefetch experiment. Their source-access example concerns MLA metadata planning, whereas our current attention domain differs. Reopen only when a new, source-specific mechanism and evidence exist. |

The inference is practical: optimize the selected forward path, not a preferred kernel architecture. Our existing pinned `muse_spark.py` already contains custom communication and conditional fusions, so a naive PyTorch/NCCL comparator would overstate gains. Full-serving ownership and exact BF16/FP4 boundaries remain mandatory. [Pinned selected-boundary source](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/models/muse_spark.py#L750).

## One bounded proposal and its falsifier

**Proposal for the next useful serving candidate, not an allocation request:** after its existing component screen passes, compare the frozen candidate and exact stock forward boundary on the same representative route/input sets in one owned process. Load immutable weights once, keep separate prepared state and scratch per implementation, warm each and capture independent graphs, then alternate the paired measurements. Include baseline-after-candidate checks to expose state leakage. Record compile/setup cost separately; preserve every required cast, norm, route and the fixed evaluator tolerance. The component stage should report only its measured chain; subsequent full-engine testing must also include the actual prefill/decode mix, cache conditions and tail TTFT.

Reject promotion if a component win reverses on the representative serving distribution, if the selected production backend differs from the tested component, if graph/state contamination occurs, or if any required numerical/rank case fails. Do not relax a failed fixed gate or infer speed from the number of proposals generated. This can improve experiment usefulness without constructing a general compiler platform. The exact implementation is deferred until a useful candidate and authorized hardware are available; **this review implements none of it**.

Read-only scope: public text/metadata and the local pinned source only. No external repository code was executed, no model weights were downloaded, and no candidate, runner, budget, GPU job or scored submission was changed.
