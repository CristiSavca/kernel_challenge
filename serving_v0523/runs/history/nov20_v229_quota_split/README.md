# Separate the scored and private-check quotas

This release gives the existing November 20 private standby its original opportunity after a separately accepted scored run. In the frozen worker, a current-day submission receipt caused the standby to return before consulting the judge, even when the receipt belonged to a scored run. The official service budgets scored runs and private checks separately. [Official serving contract](https://openkernelchallenge.dev/api/v1/serving)

The entry is `submit_nov20_v187_check.py --candidate v229`. Its preview is byte-identical to the frozen plan. The dedicated lock, shared private-allocation lock and durable claim match the previous worker. The permitted window is 04:00–08:00 UTC on November 20, ending before 08:00. V129/1 and V141/32 must already hold the first two private slots in that order. V120b/32 must still be queued or running. The worker requires the official aggregate to show exactly two private checks used and one left, then verifies V120b again. V229 or V198 selection follows the original V204/V68 comparison. [Frozen plan](nov20_v229.plan.json)

For the independent scored receipt, the worker binds one successful CLI acceptance to the patch under the candidate directory, its SHA-256, flags, submission time and job ID. The authenticated job must belong to the same team on the serving board, have mode `run`, matching flags and a compatible creation time. A completed job must also report the matching patch digest and flags. The queued response can omit the patch digest, so the worker binds it through the local accepted CLI record. A pending local claim, private job, conflicting evidence, duplicate ID or unavailable confirmation keeps the standby waiting. Legacy receipts lacking a recorded time, digest or successful CLI exit cannot establish this exception.

Every other submission receipt is bound by path and bytes through the two status observations, the final private quota and control reads, and the pre-delivery check. A conflict discovered after the durable claim preserves that claim for reconciliation. The existing candidate and comparison-job duplicate guards remain active.

Run the following from this directory for the host replay:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verify.py
```

The frozen suite passed all 145 tests against the earlier source. Those tests and six new test methods passed against this release. The new tests include the same-history old/new ablation, ambiguous receipts, false official identities, competing jobs and mutations during the initial or repeated scored read, final quota read, final control read and post-claim dispatch. The original judge transport and process locks are simulated in temporary directories for this validation. [Frozen suite](frozen_suite.json) · [Replacement suite and source hashes](validation.json)
