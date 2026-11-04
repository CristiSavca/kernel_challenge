# EAT: released method and an actionable local adaptation

Read-only review, November 4, 2026. No upstream runtime was installed or executed. The useful transfer is a supervisor that changes the work assignment when the evidence changes. We already have independent critics and persistent artifacts; adding another critic by itself does not implement this idea.

## What is actually verified

The [paper v1](https://arxiv.org/html/2610.36848v1), submitted October 27, describes supervisor reflection, team restructuring and persistent knowledge. Its headline decode experiment is DeepSeek MLA on H100, not the Muse H200 OKC decoder: reported speedup is 1.21× production FlashInfer, but 0.96× the run-only comparison, with 36/47 workloads won. Main runs report best-of-two results, while several comparisons use single runs. The paper does not establish a general gain from restructuring or contain the later OKC result.

The [author's November 2 announcement](https://x.com/shiyi_c98/status/2117240310736515510) is real: X's primary oEmbed/syndication responses identify Shiyi Cao and 13:00:01 UTC, and mention OKC. Both retrieved responses truncate before the numerical claim. A [secondary reproduction](https://digg.com/ai/ntb2pfpt) attributes H200 decode-layer **2.449× SGLang, #4 of 1,180 as of November 1**. Those historical numbers remain unverified from the retrieved primary text; they are not a paper result. Our saved official November 4 18:44 snapshot independently shows team `caoshiyi` at decoder rank 7, speedup 2.5021, job `3338748e39a74d1f`. It neither proves the historical rank nor attributes that submission to EAT.

The framework is released at [EAT commit 0a90500](https://github.com/caoshiyi/EAT/tree/0a90500b5f87c26d2fb11013b95591dcf81ed0f8), dated November 2, under Apache-2.0 with NOTICE. However, **no H200 OKC decoder source or trace was found** in the bounded advertised release:

- [eat-kernels, pinned 5b9e5d3](https://github.com/caoshiyi/eat-kernels/tree/5b9e5d3f4d21fb7b814f927c34d522bddc912b45): complete 213-entry tree; released KDA work targets MI350X/B200.
- [Evolution traces](https://huggingface.co/datasets/eva98/eat-evolution-traces): AMD KDA, B200 KDA and RoboDojo root directories; traces are CC-BY-4.0, with third-party code retaining its own licenses.
- [Task gym](https://huggingface.co/datasets/eva98/eat-gym): corresponding KDA/RoboDojo tasks, not Muse; Apache-2.0 subject to third-party notices.

This is a scoped absence finding, not a claim about unadvertised branches or private artifacts. The released KDA kernels do not supply a compatible Muse decoder port.

## What the released supervisor actually does

[SUPERVISOR.md](https://github.com/caoshiyi/EAT/blob/0a90500b5f87c26d2fb11013b95591dcf81ed0f8/agent_team/SUPERVISOR.md) assigns reflection, role/workflow changes, launching and monitoring. It tells the supervisor to improve team organization rather than supply kernel techniques. Claims of hardware limits should be challenged against evidence. This is an LLM diagnosis policy, not a numeric stall detector, a trained performance model or a genetic search engine.

[launch_stateless.sh](https://github.com/caoshiyi/EAT/blob/0a90500b5f87c26d2fb11013b95591dcf81ed0f8/agent_team/launch_stateless.sh) starts fresh supervisor sessions by default, supports persistent sessions, and parses a team configuration with bounded roles and phases. It accounts API spending after work; that is not our allocation reservation contract. It also automatically stages all changes and commits. We should not execute it in this shared workspace.

[launch_team.sh](https://github.com/caoshiyi/EAT/blob/0a90500b5f87c26d2fb11013b95591dcf81ed0f8/agent_team/launch_team.sh) runs agents concurrently within each phase, phases sequentially, then invokes a separate evaluator. Timeouts are opt-in. Evaluation output and agent tails feed the next reflection. Failed evaluator commands are captured rather than necessarily stopping the loop; score extraction alone is weaker than our frozen source/evaluator/receipt checks.

[parse_team_config.py](https://github.com/caoshiyi/EAT/blob/0a90500b5f87c26d2fb11013b95591dcf81ed0f8/agent_team/parse_team_config.py) validates structural configuration, caps roles and repairs incomplete workflow membership. Working-directory/isolation headings do not establish separate working trees. File ownership and evaluator protection still need real enforcement. We should retain our existing local tools, source pins, independent review and spending guard rather than import these weaker operational defaults.

## Narrow change to our research loop

After a terminal result or a concrete implementation blocker, root should write a short decision record before assigning the next work. A queued run is not a stall. Inputs should be the exact receipt/source/gate, at most three related attempts, the relevant profile or failure pair, and the unresolved assumption—not the entire chronological history.

The record should contain:

1. **Observation:** what passed or failed, with scope and counterevidence.
2. **Diagnosis:** the proposed structural cause, confidence and one observation that could refute it. Separate measured bottlenecks from source-count hypotheses.
3. **Work change:** one different assignment or coordination rule that obtains the missing information. Name its writer, reviewer, file boundary and deliverable.
4. **Experiment contract:** the next observable, fixed correctness/performance comparison and stop/revisit condition. Preserve budgets and evaluator/source identity.

For a numerical failure, assign the exact failing-pair diagnosis rather than another optimizer. For a measurement mismatch, assign the comparator/coverage owner. For a cross-module lifecycle gap, assign an integration specialist. For an exhausted parameter neighborhood, require a different amount of work or dependency structure, not renamed tuning. Use one implementation writer and one independent reviewer; add a specialist only for a named unresolved boundary. Keep CPU checks proportional to the actual delta, then obtain device evidence.

Small validated gains remain worthwhile. A failed attempt does not prove a hardware ceiling, but it also does not authorize unchanged reruns. Reopen a closed mechanism only with new source evidence, a changed prerequisite or a measurable causal question. Do not weaken a frozen threshold after seeing the result.

**Immediate application:** V25 failed the fixed stock-relative screen; V26 changed split accumulation into private partials plus ordered owner reduction. CAS cost was a source-based hypothesis, not a measured causal attribution. Root now reports V26 passes at 1.68511007× stock with every case faster. The next uncertainty is serving ownership: replacing the original weight bank, sharing scratch safely across layers/graphs, and selecting the actual runtime path. Assigning that integration specialist is a substantive change of work. Repeating numerical audits would not answer it. This component result still does not establish whole-model throughput or a leaderboard win.

Evidence hashes, exact source URLs, release limits and the compact machine-readable decision contract are in `experiments/iterations/20261104-eat-research.json`. Only methodology is adapted; no upstream code bodies are retained or imported.
