"""Exercise the third selection of the existing final-slot worker with temporary, offline judge histories."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import submit_nov20_v187_check as r
import test_nov20_v187_check as parent
import test_nov20_v189_check as earlier


HISTORY = r.RUNS / "history/nov20_v189_before_v198"


class Judge(earlier.Judge):
    def offer(self, candidate=None):
        return super().offer(candidate or r.V198)


class Cases(parent.Cases):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Judge(temp.name)

    def test_positive_and_exact_command(self):
        for state in ("queued","running"):
            fake = self.fixture()
            fake.jobs[parent.IDS[2]]["status"] = state
            self.assertEqual(fake.offer(), "submitted "+parent.CHILD)
            proof = r.slots.bound_receipt(fake.runs, (r.V198, 16, r.NAME, r.START, r.END))
            self.assertEqual(proof, parent.CHILD)
            header = json.loads(fake.receipt.read_text().splitlines()[0])
            self.assertEqual((header["candidate"],header["sha256"],header["flags"],header["users"]),
                             (r.V198.name,r.V198.sha256,r.V189.flags,16))
            command=["mock","serve","check",str(r.ROOT/"serving_v0523/candidates"/r.V198.directory/"patch.diff"),"--users","16"]
            for key, value in sorted(r.V189.flags.items()):
                command.extend(["--flag",key[2:] + ("="+value if value else "")])
            self.assertEqual(fake.attempts,[command+["--no-wait"]])
            self.assertEqual(fake.events[-3:-1], [["serve","status","--json"],["serve","status",parent.IDS[2],"--json"]])
            self.assertEqual(header["evidence"]["comparison_rung"]["tok_s_per_gpu"],17512.9)
            self.assertEqual(header["evidence"]["final_control"]["status"],state)
            self.assertEqual(header["evidence"]["final_quota"]["checks"],r.slots.THIRD)
            self.assertIsNotNone(r.sibling.second.prior_attempt(fake.runs))
            self.assertIsNone(r.sibling.local_position(fake.runs))

    def test_every_selection_shares_a_single_attempt(self):
        for first in (r.V187, r.V189, r.V198):
            for timeout in (False,True):
                fake=self.fixture()
                fake.timeout=timeout
                fake.offer(first)
                for second in (r.V187, r.V189, r.V198):
                    self.assertEqual(fake.offer(second), "attempt recorded")
                self.assertEqual(len(fake.attempts),1)
                self.assertEqual(len(list(fake.runs.glob("*standby_slot_nov20_check16.submit.txt"))),1)

    def test_actual_archived_worker_before_and_after_new_selection(self):
        spec=importlib.util.spec_from_file_location("standby_189_fixture", HISTORY/"submit_nov20_v187_check.py")
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        self.assertEqual((old.NAME,old.PROCESS_LOCK,old.slots.LOCK),(r.NAME,r.PROCESS_LOCK,r.slots.LOCK))
        for new_first in (False,True):
            for timeout in (False,True):
                fake=self.fixture()
                fake.timeout=timeout
                def old_offer():
                    return old.run_once(okc="mock",candidate=old.V189,runs=fake.runs,clock=lambda:fake.now,execute=fake.execute)
                actions=[fake.offer,old_offer] if new_first else [old_offer,fake.offer]
                actions[0]()
                self.assertEqual(actions[1](),"attempt recorded")
                self.assertEqual(len(fake.attempts),1)

    def test_preserve_legacy_plan_bytes_and_one_allocation(self):
        for name, args in (("v187",[]),("v189",["--candidate","v189"]),("v198",["--candidate","v198"])):
            out=io.StringIO()
            with patch.object(sys,"argv",["offline",*args]), contextlib.redirect_stdout(out):
                r.main()
            if name != "v198":
                self.assertEqual(out.getvalue(), (HISTORY/("nov20_"+name+"_runner.plan.json")).read_text())
            else:
                plan=json.loads(out.getvalue())
                old=r.plan(candidate=r.V189)
                self.assertEqual({k for k in set(plan)|set(old) if plan.get(k)!=old.get(k)},
                                 {"candidate","sha256","tree","alternative_candidates"})
                self.assertEqual(plan["alternative_candidates"],["v187","v189","v198"])

    def test_existing_lock_blocks_all_cli_selections_before_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs=Path(tmp)
            with (runs/r.PROCESS_LOCK).open("a") as holder, patch.object(r,"RUNS",runs):
                r.fcntl.flock(holder, r.fcntl.LOCK_EX | r.fcntl.LOCK_NB)
                for selected in (r.V187, r.V189, r.V198):
                    with patch.object(sys,"argv",["offline","--candidate",selected.name,"--run"]), \
                            patch.object(r.first,"utc_now",side_effect=AssertionError("entered armed worker")):
                        r.main()
                self.assertEqual(sorted(p.name for p in runs.iterdir()), [r.PROCESS_LOCK])

    def test_other_alias_during_final_quota(self):
        for target in (r.V187,r.V189,r.V198):
            for token in (target.name,target.sha256):
                fake=self.fixture()
                def concurrent(n):
                    if n==2:
                        (fake.runs/"renamed_private.submit.txt").write_text("unreconciled "+token)
                fake.on_quota=concurrent
                self.held(fake)
                self.assertFalse(fake.receipt.exists())

    def test_damaged_v198_evidence(self):
        loads=json.loads
        for field, value in (
            ("candidate","V189"), ("tree","0"*40), ("patch_sha256","0"*64), ("parent","V195"),
            ("parent_sha256","0"*64), ("parent_tree","0"*40), ("attention_donor","V192"),
            ("common_parent","V191"), ("common_tree","0"*40), ("launch_flags",{}),
            ("strict_apply_reverse_full_delta",1), ("strict_both_donor_orders",1), ("source_and_modes_verified",1)
        ):
            def damaged(raw,*args,**kwargs):
                obj=loads(raw,*args,**kwargs)
                if isinstance(obj,dict) and obj.get("candidate")=="V198":
                    obj[field]=value
                return obj
            fake=self.fixture()
            with self.subTest(field=field), patch.object(r.json,"loads",damaged):
                with self.assertRaises(ValueError):
                    fake.offer()
                self.assertFalse(fake.receipt.exists() or fake.attempts)
                self.assertEqual(r.plan(candidate=r.V189)["candidate"],"v189")


if __name__ == "__main__":
    import unittest
    unittest.main()
