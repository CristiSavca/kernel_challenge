# Optional Nov19 pending V93 branch

The original full worker is `serving_v0523/runs/submit_nov19_integrated_verify.py --run`. Its archived launch records PID 38959, caffeinate PID 38962, and exclusive ownership of `nov19_decode_runner.lock`. This isolated sibling uses the canonical run directory and the two existing Nov19 locks. Omitting the selection flag reproduces the original plan byte for byte.

Only an authenticated pending V93 selects V204/16. Both parent polls require the exact completed V68/16 result with the measured and stock rungs, and the frozen V64 scored parent and private comparator must agree. A third V93 read immediately precedes the shared receipt decision. A change the external judge makes after that observation cannot be atomic with submission. Confirmed V93 pass, failure and unknown keep their prior branches. One parseable private CLI line must bind the expected patch, width and unique fresh job ID. An unconfirmed timed-out or ambiguous attempt consumes the local claim until manual reconciliation.

The durable claim is `v168_or_v204_nov19_check16.submit.txt`. The original V168 worker recognizes the filename during rollback. The new header and CLI identify V204, which prevents a claim from qualifying as a V168 measurement on Nov20. If V204 is actually chosen, the matched V173 and V180 branches have no V168 parent and the V176 branch has neither of its two required Nov19 alternatives. Other Nov20 alternatives retain their own allocation rules.

## Reproduce without contacting the judge

From the repository root run the following commands.

```sh
runs="$PWD/serving_v0523/runs"
stage="$runs/history/nov19_v204_staging"
(cd "$stage" && PYTHONPATH="$runs" python3 -B -m unittest -q test_v204 test_nov19_decode_checks test_nov19_integrated_verify)
PYTHONPATH="$runs" python3 -B "$stage/submit_nov19_integrated_verify.py" --pending-choice v204
```

The 109 test methods and 51 independent mocked histories used only temporary receipts and simulated judge responses. Frozen file and plan digests are in `nov19_v204.cpu.json`.

## Optional activation by campaign owner

1. Reproduce both plans and compare the staged digests with the CPU receipt. Verify the launch receipt, PID command and uniquely held lock using `ps` and `lsof`. Compare inode and full path. Verify the intended private slot has no local attempt. Save the old launch receipt, plan and log under bounded history before replacing the process.
2. Stop only the exact verified original process, wait for its lock to release and its bound caffeinate to exit. Re-read the attempt receipts and the lock ownership. Abort replacement if another worker claimed the slot while these checks ran.
3. Using the same interpreter as the old process, run the following from the repository root after the owner approves. Give the new worker a unique log path, preserving the original log.

```sh
runs="$PWD/serving_v0523/runs"
stage="$runs/history/nov19_v204_staging"
PYTHONPATH="$runs" nohup python3 -Bu "$stage/submit_nov19_integrated_verify.py" --pending-choice v204 --run > "$runs/nov19_integrated_v204_runner.log" 2>&1 < /dev/null &
runner_pid=$!
nohup caffeinate -i -w "$runner_pid" > /dev/null 2>&1 < /dev/null &
```

4. Check the new PID owns the same exclusive runner lock, its plan stdout matches the frozen selection, and the new caffeinate process is bound to that PID. Record the real PIDs, UTC time, source digests, log path, peer lock states and receipt inventory in the launch record. Rollback uses the same one-worker handoff with the original canonical executable and its default plan. Preserve any ambiguous shared receipt across this change.
