# Start here

Read [HANDOFF_CLAUDE.md](HANDOFF_CLAUDE.md) before doing work. It is the current
handoff; older research notes contain superseded next steps.

The user paused optimization on November 4, 2026. The later request authorized
this handoff and GitHub push, not further GPU experiments. The goal and Codex
monitor are paused. Wait for an explicit instruction to resume optimization.

Read [AGENTS.md](AGENTS.md), and each problem's AGENTS.md and TASK.md before
editing its submission. Preserve official evaluators, exact passing sources,
licenses, raw results and the Modal reservation journal. Never accept updated
challenge terms for the user or expose credentials.

The public decoder is now 2.2469798527× (V35, see HANDOFF_CLAUDE.md); before that 2.1811831350×. V33 passed correctness but regressed;
it was rejected. The unfinished credit guard and V34 controller are archived
under `handoff/20261104/` as **draft patches**, not active validated tooling.
V34's worker and tests do not exist. The active Modal guard still blocks new
allocation under its conservative ledger. No paid GPU use is authorized.
