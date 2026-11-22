import fcntl
import json
import shutil
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
import unittest

import submit_nov19_integrated_verify as runner


class Allocation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.now = runner.START
        location = Path("serving_v0523/candidates",runner.V139.directory,"patch.diff")
        self.patch = self.root / location
        self.patch.parent.mkdir(parents=True)
        shutil.copyfile(runner.ROOT/location,self.patch)
        name = runner.control.PARENT_JOB + ".json"
        shutil.copyfile(runner.RUNS/name,self.runs/name)
        self.status = dict(resets_at=runner.RESET,limits=dict(checks=dict(per_day=3,used=2,left=1)))
        self.events = []
        self.on_status = None

    def execute(self, command, **kwargs):
        self.events.append(command)
        if command[2] == "status":
            if self.on_status:
                self.on_status()
            return subprocess.CompletedProcess(command,0,json.dumps(self.status),"")
        self.assertEqual(command,["okc","serve","check",str(self.patch),"--users","4",*runner.control.FLAGS,"--no-wait"])
        return subprocess.CompletedProcess(command,0,"submitted · job 0123456789abcdef","")

    def offer(self):
        return runner.offer(okc="okc",runs=self.runs,root=self.root,clock=lambda:self.now,execute=self.execute)

    def test_complete_and_repeat(self):
        self.assertEqual(self.offer(),"submitted 0123456789abcdef")
        receipt=self.runs/(runner.NAME+".submit.txt")
        self.assertEqual(json.loads(receipt.read_text().splitlines()[0])["sha256"],runner.V139.sha256)
        self.assertEqual(self.offer(),"attempt recorded")
        for c in runner.previous.TREE_ALTERNATIVES:
            self.assertEqual(runner.previous.run_candidate(c,okc="okc",runs=self.runs,root=self.root,
                             clock=lambda:self.now,execute=self.execute),"attempt recorded")
        self.assertEqual(len(self.events),2)

    def test_retained_allocations(self):
        new=runner.checked_plan()
        self.assertEqual(runner.previous.plan()["checks"][:2],new["checks"][:2])
        self.assertEqual(new["checks"][2]["measured_tokens_per_s_gpu"],12531.4)
        self.assertEqual(new["maximum_private_checks"],3)
        self.assertTrue({c.name for c in runner.RETAINED}.isdisjoint(c.name for c in runner.OLD_MEMBERS))

    def test_clocks(self):
        for now in (runner.START-timedelta(microseconds=1),runner.END,runner.END+timedelta(hours=1)):
            self.now=now
            self.assertEqual(self.offer(),"outside submission window")
        self.assertFalse(self.events)

    def test_late_close(self):
        self.on_status=lambda:setattr(self,"now",runner.END)
        self.assertIn("reconciliation",self.offer())
        self.assertEqual(len(self.events),1)

    def test_stale_or_invalid_quota(self):
        for used,left,limit,reset in ((3,0,3,runner.RESET),(0,0,3,runner.RESET),(-1,4,3,runner.RESET),
                                    (True,2,3,runner.RESET),(0,3,4,runner.RESET),(1,2,3,"stale")):
            self.status=dict(resets_at=reset,limits=dict(checks=dict(used=used,left=left,per_day=limit)))
            self.assertEqual(self.offer(),"quota request failed")
        self.assertTrue(all(c[2]=="status" for c in self.events))

    def test_each_legacy_allocation_blocks(self):
        for c in runner.OLD_MEMBERS:
            p=self.runs/(runner.previous.prefix(c)+".submit.txt")
            p.touch()
            self.assertEqual(self.offer(),"attempt recorded")
            p.unlink()
        self.assertFalse(self.events)

    def test_duplicate_aliases(self):
        for name in ("other.submit.txt","other.submit.json","1111111111111111.status.json"):
            p=self.runs/name
            p.write_text(runner.V139.sha256)
            self.assertEqual(self.offer(),"attempt recorded")
            p.unlink()
        self.assertFalse(self.events)

    def test_competing_tree_lock(self):
        with (self.runs/"nov19_tree_alternative.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self.assertEqual(self.offer(),"waiting for alternative lock")
        self.assertFalse(self.events)

    def test_source_and_control(self):
        original=self.patch.read_bytes()
        self.patch.write_bytes(b"changed")
        self.assertEqual(self.offer(),"unconfirmed patch or parent")
        self.patch.write_bytes(original)
        (self.runs/(runner.control.PARENT_JOB+".json")).write_text("{}")
        self.assertEqual(self.offer(),"unconfirmed patch or parent")
        self.assertFalse(self.events)

    def test_racing_legacy_attempt(self):
        self.on_status=lambda:(self.runs/(runner.previous.prefix(runner.previous.V126)+".submit.txt")).touch()
        self.assertIn("reconciliation",self.offer())
        self.assertEqual(len(self.events),1)

    def test_full_quota(self):
        self.status["limits"]["checks"]=dict(per_day=3,used=0,left=3)
        self.assertTrue(self.offer().startswith("submitted "))


if __name__ == "__main__":
    unittest.main()
