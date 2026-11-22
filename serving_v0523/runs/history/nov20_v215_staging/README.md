# Optional V215 private comparison at 16 users

The staged worker can select V215 for the existing final private check on November 20. The selection window is 04:00–08:00 UTC, with 08:00 exclusive. V129/1 and V141/32 must already have consumed the first two checks in order, V120b/32 must still be queued or running, and the judge must show two checks used and one left. The final remote observations remain the quota followed by V120b. The V198, V215 and earlier standby paths use the same process lock, arbitration lock and exclusive claim. [Frozen plan](nov20_v215.plan.json)

V215 requires the authenticated November 19 V204/16 submission and its exact completed private job `7fcdf9abf39b486a`. The worker verifies its patch, flags, first- and later-turn checks, memory allowance, and both first- and later-turn p90 TTFT against that job's node-matched stock measurements. It requires matching benchmark shape, model weights, image, dedicated cluster, GPU count and repetitions. The candidate's node can differ from V68's because each check provides its own node-matched stock comparison. [V204 submission](../../v168_or_v204_nov19_check16.submit.txt) · [V68 report](../../04b01f958aa5487c.json)

V68 measured 17,512.9 tokens/s/GPU and 1.424 at the stock interactivity point. Both V204 values must exceed the corresponding V68 value by more than 1.63%, the observed three-repetition spread on V64's scored 16-user rung. That translates to more than 17,798.36027 tokens/s/GPU and more than 1.4472112 at the stock point. It is a selection margin for a single-repetition check. Pending, failed or insufficient V204 evidence retains V198. If the optional choice changes during preflight, the current poll yields. A selected V215 rebinds the V204 CLI receipt through the final authorization. [V64 report](../../abd217baf0ed49fd.json) · [Selector](v215_selection.py)

A preview reads the local artifacts. This command does not start a worker.

```sh
python3 -B serving_v0523/runs/history/nov20_v215_staging/submit_nov20_v187_check.py --candidate v215
```

The staged executable imports the established gating modules from the canonical runs directory. Its run-mode entry point is the same command with `--run`. The V198-only selection still produces its previously frozen plan and skips all V204 queries. [Regression receipt](nov20_v215.cpu.json) · [Staged worker](submit_nov20_v187_check.py) · [Local tests](test_v215.py)
