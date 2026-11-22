"""Exercise the actual Nov19 alternative, V163 submission and conditional V176 comparison."""
import copy
import json
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import submit_nov20_v176_check as r
import test_nov20_v163_check as v163

ALT_JOB = "5555555555555555"
CHILD = "6666666666666666"
IDS = (*v163.IDS, v163.ACCEPTED, ALT_JOB)


class Judge(v163.Fake):
    def __init__(self, directory, alternate):
        super().__init__(directory)
        self.alt = alternate
        self.child = CHILD
        self.events = []
        self.now = r.nov19.START
        self.quota["resets_at"] = r.nov19.RESET
        # Use the original Nov19 writer and branch evidence. Only its dependency outcomes are emulated.
        with patch.object(r.nov19, "dependency", return_value=("passed", "mocked Nov19 comparator")):
            marker = r.nov19._offer(alternate[0], self.runs, r.ROOT, "mock", lambda: self.now, self.execute)
        assert marker == "submitted " + ALT_JOB, marker
        self.set_state(alternate, ALT_JOB, "pending")
        self.now = r.START + timedelta(minutes=7)
        self.quota["resets_at"] = r.RESET
        assert v163.Fake.offer(self) == "submitted " + v163.ACCEPTED
        self.set_state(r.fallback.OWN, v163.ACCEPTED, "pass")
        self.jobs[v163.ACCEPTED]["created_at"] = self.now.timestamp()
        self.now += timedelta(minutes=1)
        self.quota["limits"]["checks"] = r.slots.THIRD.copy()
        self.attempts.clear()
        self.queries.clear()
        self.events.clear()
        self.reads = 0

    def execute(self, cmd, **kw):
        self.events.append(cmd[1:])
        response = super().execute(cmd, **kw)
        if cmd[1:3] == ["serve", "check"]:
            directory = Path(cmd[3]).parent.name
            job = {self.alt[0].directory: ALT_JOB, r.V176.directory: self.child}.get(directory, v163.ACCEPTED)
            response.stdout = response.stdout.replace(v163.ACCEPTED, job)
        return response

    def offer(self):
        return r.run_once(okc="mock", runs=self.runs, clock=lambda: self.now, execute=self.execute)

    @property
    def receipt(self):
        return self.runs / (r.NAME + ".submit.txt")


class Standby(unittest.TestCase):
    def fixture(self, alternate=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Judge(temp.name, alternate or r.ALTERNATIVES[0])

    def held(self, f):
        self.assertFalse(f.offer().startswith("submitted "))
        self.assertFalse(f.attempts or f.receipt.exists())

    def update_header(self, f, req, key, value):
        receipt = f.runs / (req[2] + ".submit.txt")
        first, _, tail = receipt.read_text().partition("\n")
        header = json.loads(first)
        header["evidence"][key] = value
        receipt.write_text(json.dumps(header) + "\n" + tail)

    def test_each_authentic_nov19_alternate_opens_exact_private_slot_once(self):
        for alternate in r.ALTERNATIVES:
            f = self.fixture(alternate)
            with patch.object(r.measured, "submit_once", wraps=r.measured.submit_once) as submit:
                self.assertEqual(f.offer(), "submitted " + CHILD)
                self.assertEqual(submit.call_args.kwargs["reserve_checks"], 0)
            self.assertEqual(r.slots.bound_receipt(f.runs, r.OWN), CHILD)
            evidence = json.loads(f.receipt.read_text().splitlines()[0])["evidence"]
            self.assertEqual((evidence["nov19_selected"], evidence["nov19_job"]), (alternate[0].name, ALT_JOB))
            self.assertEqual((evidence["v129_job"], evidence["comparison_job"]), (IDS[0], IDS[3]))
            self.assertEqual(evidence["comparison_rung"], v163.REAL_PASS["result"]["rungs"][0])
            self.assertEqual(evidence["comparison_node"], v163.REAL_PASS["result"]["node"])
            self.assertEqual(f.events[-2], ["serve", "status", "--json"])
            patch_path = str(r.ROOT / "serving_v0523/candidates" / r.V176.directory / "patch.diff")
            command = ["mock", "serve", "check", patch_path, "--users", "16"]
            for key, val in sorted(r.V176.flags.items()):
                command.extend(["--flag", key[2:] + ("=" + val if val else "")])
            self.assertEqual(f.attempts, [command + ["--no-wait"]])
            self.assertEqual(f.offer(), "attempt recorded")

    def test_reservation_never_uses_absence_or_crosses_v168_v173(self):
        for alt in r.ALTERNATIVES:
            f = self.fixture(alt)
            (f.runs / (alt[2] + ".submit.txt")).unlink()
            self.held(f)
            self.assertFalse(f.queries)
            f = self.fixture(alt)
            other = next(a for a in r.ALTERNATIVES if a != alt)
            f.write(other, "7"*16)
            self.held(f)
        for competitor in (r.peer.PARENT, r.peer.OWN, *r.fallback.PEERS):
            for name, text in ((competitor[2]+".submit.txt", ""), (competitor[2]+".skip.json", "{}"),
                               ("other.submit.txt", competitor[0].sha256), ("8"*16+".json", competitor[0].sha256)):
                f = self.fixture()
                (f.runs / name).write_text(text)
                self.held(f)
        for old in r.nov19.LEGACY_PREFILL_ALTERNATIVES:
            f = self.fixture()
            (f.runs / (r.nov19.prefix(old)+".submit.txt")).touch()
            self.held(f)
        for state in ("pending", "pass", "crash", "timeout"):
            f = self.fixture()
            f.write(r.peer.PARENT, "7"*16)
            f.set_state(r.peer.PARENT, "7"*16, state)
            self.held(f)
        f = self.fixture()
        self.assertIsNone(r.peer.local_position(f.runs))

    def test_orphaned_alternate_submissions_and_judge_results(self):
        for selected in r.ALTERNATIVES:
            for conflicting in r.SLOT_HISTORY:
                for form in ("receipt", "official", "orphan"):
                    f = self.fixture(selected)
                    job = "7"*16
                    if form == "receipt":
                        # Preserve the canonical selected proof while making a separate authentic competing job.
                        old = f.runs / (conflicting[2]+".submit.txt")
                        retained = old.read_bytes() if old.exists() else None
                        f.write(conflicting, job)
                        old.rename(f.runs / "renamed.submit.txt")
                        if retained is not None:
                            old.write_bytes(retained)
                    elif form == "official":
                        f.set_state(conflicting, job, "pass")
                        (f.runs / (job+".json")).write_text(json.dumps(f.jobs[job]))
                    else:
                        (f.runs/"orphan.submit.json").write_text(conflicting[0].sha256)
                    self.held(f)
            f = self.fixture(selected)
            other = next(req for req in r.ALTERNATIVES if req != selected)
            prior = (other[0], 16, "historical", other[3]-timedelta(days=1), other[4]-timedelta(days=1))
            f.write(prior, "7"*16)
            f.set_state(prior, "7"*16, "pass")
            (f.runs / ("7"*16+".json")).write_text(json.dumps(f.jobs["7"*16]))
            self.assertEqual(f.offer(),"submitted "+CHILD)

    def test_receipt_rotate_during_read_defers_and_recovers(self):
        for bad in ("", "[]", "1234", "???", None):
            f = self.fixture()
            p = f.runs / (r.fallback.NAME + ".submit.txt")
            original = p.read_bytes()
            def rotating(runs, req):
                if req == r.fallback.OWN:
                    if bad is None:
                        p.unlink(missing_ok=True)
                    else:
                        p.write_text(bad)
                return reader(runs, req)
            reader = r.receipt_evidence
            with patch.object(r, "receipt_evidence", side_effect=rotating):
                self.held(f)
            p.write_bytes(original)
            self.assertEqual(f.offer(), "submitted "+CHILD)

    def test_both_governed_receipts_must_bind_their_actual_decisions(self):
        for field, value in (("exclusive_alternatives", ["v98","v95"]), ("check_load", True),
                             ("comparison_job", "7"*16), ("comparison_receipt", "missing")):
            f = self.fixture()
            self.update_header(f, f.alt, field, value)
            self.held(f)
        for field, value in (("slot", True), ("slot", 3), ("v129_job", ALT_JOB),
                             ("failed_controls", {"v120b": IDS[2], "v139": IDS[1]})):
            f = self.fixture()
            self.update_header(f, r.fallback.OWN, field, value)
            self.held(f)

    def test_distinct_canonical_and_live_identities_for_every_job(self):
        for i, job in enumerate(IDS):
            for key, value in (("id","7"*16), ("board","other"), ("mode","run"), ("created_at",0),
                               ("users", True), ("users","16"), ("serving_flags",{}), ("duplicate_of","7"*16)):
                f = self.fixture()
                f.jobs[job][key] = value
                self.held(f)
            for change in ("missing", "same_job_elsewhere", "bad_load", "bad_exit", "skip"):
                f = self.fixture()
                req = (*r.PRECEDING, f.alt)[i]
                p = f.runs / (req[2]+".submit.txt")
                if change == "missing":
                    p.unlink()
                elif change == "same_job_elsewhere":
                    (f.runs / "extra.submit.txt").write_text("job "+job)
                elif change == "skip":
                    (f.runs / (req[2]+".skip.json")).touch()
                else:
                    original = p.read_text()
                    bad = original.replace("CLI exit code: 0", "CLI exit code: 1") if change == "bad_exit" else original.replace("as a check run at ", "as a check run at 7")
                    p.write_text(bad)
                self.held(f)
        for job in (IDS[0], IDS[3], IDS[4]):
            f = self.fixture()
            f.jobs[job]["refunded"] = True
            self.held(f)
        f = self.fixture()
        f.jobs[IDS[0]]["created_at"] = f.jobs[IDS[3]]["created_at"]+1
        self.held(f)
        f = self.fixture()
        f.jobs[ALT_JOB]["status"] = "withdrawn"
        self.held(f)

    def test_v163_pass_requires_real_complete_private_metrics(self):
        for i in (1, 2):
            for state in ("pending", "pass"):
                f = self.fixture()
                f.set_state(r.PRECEDING[i], IDS[i], state)
                self.held(f)
        for state in ("pending", "incorrect", "timeout", "crash"):
            f = self.fixture()
            f.set_state(r.fallback.OWN, IDS[3], state)
            self.held(f)
            self.assertEqual(f.queries, [IDS[3]])
        for key, value in (("passed", False), ("serving_flags", {}), ("patch", {}), ("concurrency", [True]),
                           ("short_outputs", 1), ("later_turns", {"ran": True, "passed": False}),
                           ("correctness", {"passed": False}), ("rungs", [])):
            f = self.fixture()
            f.jobs[IDS[3]]["result"][key] = value
            self.held(f)
        for key, value in (("reps", True), ("concurrency", 32), ("truncated", True),
                           ("tok_s_per_gpu", float("nan")), ("tok_s_per_user_e2e", 0),
                           ("tok_s_per_user_p50", False), ("ttft_first_p90_s", float("inf"))):
            f = self.fixture()
            f.jobs[IDS[3]]["result"]["rungs"][0][key] = value
            self.held(f)
        for i in (1,2):
            f=self.fixture()
            f.set_state(r.PRECEDING[i],IDS[i],"timeout")
            self.assertEqual(f.offer(),"submitted "+CHILD)

    def test_quota_reserve_rechecks_late_markers_and_window(self):
        for t in (r.START - timedelta(microseconds=1), r.END):
            f = self.fixture()
            f.now = t
            self.held(f)
            self.assertFalse(f.queries)
        for count in (1, 2):
            for event in ("used", "left", "cap", "reset", "v168", "peer", "cutoff", "refunded", "truncated", "lost"):
                if count==2 and event in ("refunded", "truncated", "lost"):
                    continue
                f = self.fixture()
                def change(n):
                    if n != count: return
                    if event == "used": f.quota["limits"]["checks"]["used"] = 1
                    if event == "left": f.quota["limits"]["checks"]["left"] = True
                    if event == "cap": f.quota["limits"]["checks"]["per_day"] = 4
                    if event == "reset": f.quota["resets_at"] = "wrong"
                    if event == "v168": f.write(r.peer.PARENT,"7"*16)
                    if event == "peer": (f.runs / (r.peer.NAME+".skip.json")).touch()
                    if event == "cutoff": f.now = r.END
                    if event == "refunded": f.jobs[ALT_JOB]["refunded"] = True
                    if event == "truncated": f.jobs[IDS[3]]["result"]["rungs"][0]["truncated"] = True
                    if event == "lost": del f.jobs[IDS[1]]
                f.on_quota = change
                self.held(f)

    def test_shared_arbitration_excludes_fallback_and_ambiguous_submission_is_final(self):
        f = self.fixture()
        with r.slots.arbitrate(f.runs):
            self.held(f)
            self.assertFalse(f.queries)
        arrived, release = threading.Event(), threading.Event()
        answers = []
        def slow(n):
            if n==1:
                arrived.set()
                self.assertTrue(release.wait(5))
        f.on_quota = slow
        other = threading.Thread(target=lambda: answers.append(f.offer()))
        other.start()
        self.assertTrue(arrived.wait(5))
        self.assertEqual(f.offer(), "another allocation poll is active")
        self.assertEqual(r.peer.run_once(okc="mock",runs=f.runs,clock=lambda:f.now,execute=f.execute),
                         "another allocation poll is active")
        release.set()
        other.join(6)
        self.assertEqual(answers, ["submitted "+CHILD])
        for job in IDS:
            f=self.fixture()
            f.child=job
            self.assertTrue(f.offer().startswith("attempt needs manual reconciliation"))
            self.assertEqual(f.offer(),"attempt recorded")
        for name in ("bad_acceptance", "timeout"):
            f=self.fixture()
            setattr(f,name,True)
            self.assertTrue(f.offer().startswith("attempt "))
            self.assertEqual(f.offer(),"attempt recorded")
            self.assertEqual(len(f.attempts), 1)

    def test_real_hash_and_existing_private_plans(self):
        plan = r.plan()
        self.assertEqual(plan["tree"], r.TREE)
        self.assertEqual(plan["flags"], r.peer.V168.flags | {"--speculative-dflash-share-verify-schedule":"",
                                                           "--speculative-dflash-sp-prefill-context":""})
        self.assertEqual(plan["required_quota"], {"per_day":3, "used":2, "left":1})
        self.assertEqual((plan["mode"],plan["users"],plan["maximum_private_checks"]),("check",16,1))
        self.assertIn(r.peer.PARENT[2],plan["requires_unattempted"])
        self.assertEqual(plan["comparison"]["reference"],r.fallback.NAME)


if __name__=="__main__":
    unittest.main()
