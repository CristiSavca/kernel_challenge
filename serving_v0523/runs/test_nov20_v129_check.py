"""Offline Nov20 submission, quota, clock and archived-receipt checks."""
import concurrent.futures
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path

import submit_nov20_v129_check as runner


class Fixture:
    def __init__(self, tmp):
        self.root = Path(tmp)
        self.runs = self.root/"runs"
        self.runs.mkdir()
        self.parent = self.runs/(runner.PARENT_JOB+".json")
        shutil.copyfile(runner.RUNS/self.parent.name,self.parent)
        for candidate in (runner.V64,runner.V129):
            target = self.root/"serving_v0523/candidates"/candidate.directory
            source = runner.ROOT/"serving_v0523/candidates"/candidate.directory
            target.mkdir(parents=True)
            shutil.copyfile(source/"patch.diff",target/"patch.diff")
            if candidate == runner.V129:
                shutil.copyfile(source/"validation.json",target/"validation.json")
        self.patch = self.root/"serving_v0523/candidates"/runner.V129.directory/"patch.diff"
        self.now = runner.START
        self.quota = {"resets_at":runner.RESET, "limits":{"checks":{"per_day":3,"used":0,"left":3}}}
        self.commands = []
        self.after_quota = lambda: None
        self.status_rc = 0
        self.check_rc = 0
        self.check_text = "Submitted job 1234567890abcdef\n"
        self.timeout = False

    def execute(self, cmd, **kwargs):
        self.commands.append((cmd,kwargs))
        if cmd[1:]==["serve","status","--json"]:
            self.after_quota()
            return subprocess.CompletedProcess(cmd,self.status_rc,json.dumps(self.quota),"")
        assert cmd[1:3] == ["serve","check"]
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd,kwargs["timeout"])
        return subprocess.CompletedProcess(cmd,self.check_rc,self.check_text,"")

    def run(self):
        return runner.run_once(okc="dummy",root=self.root,runs=self.runs,clock=lambda:self.now,execute=self.execute)

    @property
    def checks(self):
        return [c for c,k in self.commands if c[1:3] == ["serve","check"]]


class Scenarios(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Fixture(tmp.name)

    def test_only_exact_private_command_restart_reserved_slots(self):
        f=self.fixture()
        self.assertEqual(f.run(),"submitted 1234567890abcdef")
        self.assertEqual(f.checks,[["dummy","serve","check",str(f.patch),"--users","1",
                                   "--flag","enable-layernorm-sp","--flag","moe-runner-backend=humming",
                                   "--flag","speculative-moe-runner-backend=marlin",
                                   "--flag","swa-full-tokens-ratio=0.25","--no-wait"]])
        self.assertTrue(f.run().startswith("attempt recorded"))
        self.assertEqual(len(f.commands),2)
        m=json.loads((f.runs/(runner.NAME+".submit.txt")).read_text().splitlines()[0])
        self.assertEqual((m["users"],m["candidate"],m["sha256"],m["flags"]),(1,"v129",runner.V129.sha256,runner.V64.flags))

    def test_clock_edges(self):
        for when in (runner.START-timedelta(microseconds=1),runner.END,runner.END+timedelta(days=1)):
            with self.subTest(when=when):
                f=self.fixture(); f.now=when
                self.assertEqual(f.run(),"outside submission window")
                self.assertFalse(f.commands)
        f=self.fixture(); f.now=runner.END-timedelta(milliseconds=20)
        self.assertTrue(f.run().startswith("submitted"))
        self.assertLessEqual(f.commands[-1][1]["timeout"],.02)

    def test_reset(self):
        for reset in (None,"2026-11-20T00:00:00Z","2026-11-22T00:00:00Z"):
            with self.subTest(reset=reset):
                f=self.fixture(); f.quota["resets_at"]=reset; f.run()
                self.assertFalse(f.checks)
                self.assertFalse(list(f.runs.glob("*.submit.txt")))

    def test_guard_fields(self):
        for key,values in (("left",(True,2,1,0,4,"3",None)),("used",(1,None,False)),("per_day",(4,True,None))):
            for val in values:
                with self.subTest(key=key,value=val):
                    f=self.fixture(); f.quota["limits"]["checks"][key]=val; f.run()
                    self.assertFalse(f.checks)

    def test_status_failure_and_malformed(self):
        for quota,rc in (({},0),({"limits":[]},0),([],0),(None,0),({},1)):
            with self.subTest(quota=quota,rc=rc):
                f=self.fixture(); f.quota=quota; f.status_rc=rc; f.run(); self.assertFalse(f.checks)

    def test_duplicate_identity_and_ambiguous_attempts(self):
        receipts = {"v129_old_check4.submit.txt":"",
                    "old_alias.submit.txt":json.dumps({"candidate":"V129","sha256":"wrong"}),
                    "old_different_name.submit.txt":json.dumps({"candidate":"legacy","sha256":runner.V129.sha256}),
                    "legacy.submit.txt":"opaque attempt of v129_routed_projection_pair; timed out",
                    "incomplete.submit.txt":'{"users":1}\nambiguous launch of v129_routed_projection_pair',
                    "conflict.submit.txt":'{"candidate":"other","sha256":"some"}\nchecking v129_routed_projection_pair/patch.diff',
                    "1234567890abcdef.status.json":json.dumps({"mode":"check","result":{"patch":{"sha256":runner.V129.sha256}}}),
                    "1234567890abcdef.json":'{malformed "sha256":"'+runner.V129.sha256,
                    "v129_previous_check1.json":"{broken", "v129_check1.submitted_at.txt":"00:00"}
        for name,raw in receipts.items():
            with self.subTest(name=name):
                f=self.fixture(); (f.runs/name).write_text(raw)
                self.assertTrue(f.run().startswith("attempt recorded"))
                self.assertFalse(f.commands)

    def test_unrelated_attempts(self):
        f=self.fixture()
        (f.runs/"v129b_old.submit.txt").write_text('{"candidate":"v129b","sha256":"other"}')
        (f.runs/"separate.submit.txt").write_text('{"candidate":"v140","sha256":"other","evidence":{"parent":"v129"}}')
        self.assertTrue(f.run().startswith("submitted"))

    def test_frozen_files_parent_and_rung(self):
        for field in ("parent_patch","candidate_patch","mode","receipt_id","flags","result_sha","load","valid_rung","parent_validation"):
            with self.subTest(field=field):
                f=self.fixture(); d=json.loads(f.parent.read_text())
                if field=="parent_patch": (f.patch.parent.parent/runner.V64.directory/"patch.diff").write_text("bad")
                elif field=="candidate_patch": f.patch.write_text("bad")
                elif field=="mode": d["mode"]="check"
                elif field=="receipt_id": d["id"]="1111111111111111"
                elif field=="flags": d["serving_flags"]={}
                elif field=="result_sha": d["result"]["patch"]["sha256"]="wrong"
                elif field=="load": d["result"]["concurrency"]= [4,16,32]
                elif field=="valid_rung": d["result"]["rungs"][0]["truncated"]=True
                elif field=="parent_validation":
                    val=f.patch.parent/"validation.json"; payload=json.loads(val.read_text())
                    payload["launch_flags"]={}; val.write_text(json.dumps(payload))
                f.parent.write_text(json.dumps(d))
                with self.assertRaises(ValueError): f.run()
                self.assertFalse(f.commands)

    def test_reconcile_after_quota(self):
        for change in ("clock","prior","parent","patch"):
            with self.subTest(change=change):
                f=self.fixture()
                def hook():
                    if change=="clock":f.now=runner.END
                    elif change=="prior":(f.runs/"old.submit.txt").write_text('{"candidate":"v129"}')
                    elif change=="parent":f.parent.write_text("{}")
                    else:f.patch.write_text("changed")
                f.after_quota=hook
                f.run()
                self.assertFalse(f.checks)

    def test_ambiguous_cli_blocks_retry(self):
        for text,rc,timeout in (("job 1111111111111111 and job 2222222222222222",0,False),
                                ("no response",0,False),("error",1,False),("",0,True)):
            with self.subTest(text=text):
                f=self.fixture(); f.check_text=text; f.check_rc=rc; f.timeout=timeout
                self.assertTrue(f.run().startswith("attempt "))
                self.assertTrue(f.run().startswith("attempt recorded"))
                self.assertEqual(len(f.checks),1)

    def test_two_concurrent_offers(self):
        f=self.fixture(); barrier=threading.Barrier(2)
        f.after_quota=lambda:barrier.wait(10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            responses=list(pool.map(lambda _:f.run(),range(2)))
        self.assertEqual(len(f.checks),1)
        self.assertEqual(responses.count("submitted 1234567890abcdef"),1)


if __name__=="__main__":
    unittest.main(verbosity=2)
