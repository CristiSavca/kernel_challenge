"""Replay the two private allocations against an isolated mutable fake judge."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import submit_nov20_v132_check as v132
import submit_nov20_v141_check as v141
from test_nov20_v141_check import Fixture, JOBS

COMP = "5555555555555555"
NEW = {v132.V132.directory: "6666666666666666", v141.V141.directory: "7777777777777777"}


class Arbitration(unittest.TestCase):
    def setup_batch(self, control_state="queued", now=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        f = Fixture(Path(tmp.name))
        f.docs[1].unlink()
        f.now = now or v141.slots.GRACE
        f.quota["limits"]["checks"] = v141.slots.SECOND.copy()
        cc = v132.V120b
        src = v132.ROOT / "serving_v0523/candidates" / cc.directory
        dst = f.patch.parent.parent / cc.directory
        dst.mkdir()
        for name in ("validation.json","patch.diff"):
            shutil.copyfile(src / name, dst / name)
        cstart = v132.START - timedelta(days=1)
        head = dict(candidate=cc.name, sha256=cc.sha256, flags=cc.flags, users=32, attempted_at=cstart.isoformat())
        f.control = f.runs / (v132.prefix(cc)+".submit.txt")
        f.control.write_text(json.dumps(head)+f"\nsubmitted /fixture/{cc.directory}/patch.diff as a check run at 32 users · job {COMP}\nCLI exit code: 0\n")
        official = json.loads((v132.RUNS / "04b01f958aa5487c.json").read_text())["result"]
        official.update(concurrency=[32], serving_flags=cc.flags)
        official["patch"]["sha256"] = cc.sha256
        official["rungs"][0]["concurrency"] = 32
        def control(status):
            f.jobs[COMP] = dict(id=COMP, users=32, board="serving", mode="check", serving_flags=cc.flags,
                                created_at=cstart.timestamp(), status=status, result=official if status=="done" else None)
        f.change_control = control
        control(control_state)
        ordinary = f.execute
        def execute(cmd, **kwargs):
            if cmd[1:3] != ["serve","check"]:
                return ordinary(cmd, **kwargs)
            f.calls.append((cmd,kwargs))
            directory = Path(cmd[3]).parent.name
            job = NEW[directory]
            c = v141.V141 if directory == v141.V141.directory else v132.V132
            f.jobs[job] = dict(id=job, board="serving", mode="check", users=32, serving_flags=c.flags,
                               status="queued", created_at=f.now.timestamp(), result=None)
            counts = f.quota["limits"]["checks"]
            counts["used"] += 1
            counts["left"] -= 1
            return subprocess.CompletedProcess(cmd, 0, f"submitted {cmd[3]} as a check run at 32 users · job {job}", "")
        f.execute = execute
        return f

    def offer(self, f, runner):
        return runner.run_once(okc="test-cli", clock=lambda:f.now, runs=f.runs, root=f.root, execute=f.execute)

    def test_nominal_after_grace_measured_control_keeps_second(self):
        f = self.setup_batch("done")
        for i in range(2):
            self.assertIn("priority", self.offer(f, v141))
            self.assertFalse(f.checks)
        for runner, reserve in ((v132,1),(v141,0)):
            with patch.object(runner,"submit_once",wraps=runner.submit_once) as offer:
                self.assertTrue(self.offer(f, runner).startswith("submitted "))
                self.assertEqual(offer.call_args.kwargs["reserve_checks"],reserve)
        self.assertEqual([Path(c[3]).parent.name for c in f.checks],list(NEW))
        self.assertEqual(f.quota["limits"]["checks"]["left"],0)
        for runner in (v141,v132):
            self.assertEqual(self.offer(f, runner),"attempt recorded")

    def test_reverse_order_after_control_passes(self):
        for status in ("queued","running","failed","canceled","withdrawn","missing"):
            with self.subTest(status=status):
                f = self.setup_batch(status)
                raw = f.control.read_text()
                if status == "missing": f.control.unlink()
                self.assertFalse(self.offer(f,v132).startswith("submitted "))
                with patch.object(v141,"submit_once",wraps=v141.submit_once) as offer:
                    self.assertTrue(self.offer(f,v141).startswith("submitted "))
                    self.assertEqual(offer.call_args.kwargs["reserve_checks"],1)
                self.assertFalse(self.offer(f,v132).startswith("submitted "))
                f.control.write_text(raw)
                f.change_control("done")
                with patch.object(v132,"submit_once",wraps=v132.submit_once) as offer:
                    self.assertTrue(self.offer(f,v132).startswith("submitted "))
                    self.assertEqual(offer.call_args.kwargs["reserve_checks"],0)
                self.assertEqual([Path(c[3]).parent.name for c in f.checks],list(reversed(NEW)))
                head=json.loads((f.runs/(v132.NAME+".submit.txt")).read_text().splitlines()[0])
                self.assertEqual(head["evidence"]["v141_job"],NEW[v141.V141.directory])
                self.assertEqual(head["evidence"]["comparison_job"],COMP)

    def test_grace_and_terminal_parent(self):
        f=self.setup_batch(now=v141.slots.GRACE-timedelta(microseconds=1))
        self.assertFalse(self.offer(f,v141).startswith("submitted ")); self.assertFalse(f.calls)
        f.now=v141.slots.GRACE
        f.jobs[JOBS[2]]["status"]="failed"
        self.assertEqual(self.offer(f,v141),"terminal V139 failure"); self.assertFalse(f.checks)
        # No V139 requirement is inherited by V132.
        f.change_control("done")
        self.assertTrue(self.offer(f,v132).startswith("submitted "))

    def test_unreconciled_peer_blocks_both_directions(self):
        for name,runner,peer in ((v132.NAME,v141,v132.V132),(v141.NAME,v132,v141.V141)):
            for kind in ("empty","nonzero","no_summary","orphan","skip","old","torn"):
                with self.subTest(peer=peer.name,kind=kind):
                    f=self.setup_batch("done" if runner==v132 else "queued")
                    path=f.runs/(name+".submit.txt")
                    if kind=="orphan":path=f.runs/"older";path.mkdir();path=path/"odd.submit.txt"
                    elif kind=="skip":path=f.runs/(name+".skip.json")
                    h=dict(candidate=peer.name,sha256=peer.sha256,flags=peer.flags,users=32,attempted_at=f.now.isoformat())
                    if kind=="old":h["attempted_at"]=(f.now-timedelta(days=1)).isoformat()
                    line=f"submitted /fixture/{peer.directory}/patch.diff as a check run at 32 users · job 8888888888888888"
                    raw=json.dumps(h)+"\n"+line+"\nCLI exit code: 0\n"
                    if kind in ("empty","skip"):raw=""
                    elif kind=="torn":raw='{"'
                    elif kind=="nonzero":raw=raw.replace("code: 0","code: 1")
                    elif kind=="no_summary":raw=raw.replace("as a check run","as a run")
                    path.write_text(raw)
                    self.assertFalse(self.offer(f,runner).startswith("submitted "))
                    self.assertFalse(f.checks)

    def test_reversed_last_slot_requires_exact_peer_identity_and_quota(self):
        for fault in ("flags","sha","user","date","live_user","live_flags","live_date","live_duplicate","live_id",
                      "live_mode","unknown","id_alias","count","reset","receipt_removed","control_flip","expired"):
            with self.subTest(fault=fault):
                f=self.setup_batch(); self.assertTrue(self.offer(f,v141).startswith("submitted "))
                f.change_control("done"); job=NEW[v141.V141.directory]
                doc=f.runs/(v141.NAME+".submit.txt"); raw=doc.read_text()
                line,_,tail=raw.partition("\n"); head=json.loads(line)
                if fault in ("flags","sha","user","date"):
                    head.update({"flags":{"flags":{}},"sha":{"sha256":"bad"},"user":{"users":1},
                                 "date":{"attempted_at":(v141.slots.GRACE-timedelta(microseconds=1)).isoformat()}}[fault])
                    doc.write_text(json.dumps(head)+"\n"+tail)
                elif fault.startswith("live_") or fault=="unknown":
                    f.jobs[job].update({"live_user":{"users":16},"live_flags":{"serving_flags":{}},
                        "live_date":{"created_at":v141.slots.GRACE.timestamp()-1},"live_duplicate":{"duplicate_of":COMP},
                        "live_id":{"id":COMP},"live_mode":{"mode":"run"},"unknown":{"status":"other"}}[fault])
                elif fault=="id_alias":doc.write_text(raw.replace(job,JOBS[0]))
                elif fault=="count":f.quota["limits"]["checks"]=v141.slots.SECOND.copy()
                elif fault=="reset":f.quota["resets_at"]="2026-11-20T00:00:00Z"
                elif fault=="receipt_removed":f.when_quota=lambda:doc.unlink()
                elif fault=="control_flip":f.when_quota=lambda:f.change_control("failed")
                elif fault=="expired":f.when_quota=lambda:setattr(f,"now",v141.END)
                self.assertFalse(self.offer(f,v132).startswith("submitted "));self.assertEqual(len(f.checks),1)

    def test_priority_changes_and_fallback_quota_races(self):
        for fault in ("pass","ambiguous","quota","deadline","score_parent","bad_first_identity"):
            with self.subTest(fault=fault):
                f=self.setup_batch()
                if fault=="pass":f.when_quota=lambda:f.change_control("done")
                if fault=="ambiguous":f.when_quota=lambda:(f.runs/(v132.NAME+".submit.txt")).write_text("unknown")
                if fault=="quota":f.when_quota=lambda:f.quota.update(limits={"checks":v141.slots.THIRD.copy()})
                if fault=="deadline":f.when_quota=lambda:setattr(f,"now",v141.END)
                if fault=="score_parent":f.when_quota=lambda:f.jobs[JOBS[2]].update(status="withdrawn")
                if fault=="bad_first_identity":f.jobs[JOBS[0]]["serving_flags"]={}
                self.assertFalse(self.offer(f,v141).startswith("submitted "));self.assertFalse(f.checks)

    def test_pending_load_uses_exact_receipt_and_unknown_control_defers(self):
        for fault in ("control_identity","control_error","control_torn","control_wrong_date","unknown_control","no_control_load",
                      "pending_parent_no_load","pending_first_bool","pending_first_no_load","missing_first_summary"):
            for runner in (v132,v141):
                with self.subTest(fault=fault,runner=runner.NAME):
                    f=self.setup_batch("done" if runner==v132 else "queued")
                    if fault=="control_identity":f.jobs[COMP]["id"]="9999999999999999"
                    elif fault=="control_error":
                        original=f.execute
                        def execute(cmd,**kwargs):
                            if cmd[1:4]==["serve","status",COMP]:
                                return subprocess.CompletedProcess(cmd,1,"","unavailable")
                            return original(cmd,**kwargs)
                        f.execute=execute
                    elif fault=="control_torn":f.control.write_text('{"')
                    elif fault=="control_wrong_date":f.jobs[COMP]["created_at"]=f.now.timestamp()
                    elif fault=="unknown_control":f.jobs[COMP]["status"]="other"
                    elif fault=="no_control_load":f.jobs[COMP].update(status="queued",users=None,result=None)
                    elif fault=="pending_parent_no_load":f.jobs[JOBS[2]].pop("users")
                    elif fault=="pending_first_bool":f.jobs[JOBS[0]]["users"]=True
                    elif fault=="pending_first_no_load":f.jobs[JOBS[0]].pop("users")
                    else:f.docs[0].write_text(f.docs[0].read_text().replace("as a check run","as a run"))
                    if fault in ("pending_parent_no_load","pending_first_no_load") or fault=="no_control_load" and runner==v141:
                        self.assertTrue(self.offer(f,runner).startswith("submitted "))
                    else:
                        self.assertFalse(self.offer(f,runner).startswith("submitted "));self.assertFalse(f.checks)

    def test_arbitration_is_kernel_flock_both_directions_one_poll_only(self):
        for winner,other,status in ((v141,v132,"queued"),(v132,v141,"done")):
            f=self.setup_batch(status)
            entered=threading.Event();release=threading.Event()
            def hold():
                entered.set()
                self.assertTrue(release.wait(5))
            f.when_quota=hold
            with ThreadPoolExecutor(1) as pool:
                pending=pool.submit(self.offer,f,winner)
                self.assertTrue(entered.wait(5))
                try:
                    self.assertEqual(self.offer(f,other),"another allocation poll is active")
                    cmd=[sys.executable,"-c",
                         "import fcntl,sys; fd=open(sys.argv[1],'a'); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)",
                         str(f.runs/v141.slots.LOCK)]
                    self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
                finally:release.set()
                self.assertTrue(pending.result().startswith("submitted "))
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            f.when_quota=lambda:None;f.change_control("done")
            self.assertTrue(self.offer(f,other).startswith("submitted "))
            self.assertEqual(len(f.checks),2)
