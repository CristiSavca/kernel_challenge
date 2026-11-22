"""Replay V163->V173 using registered source and captured judge schemas. All execute calls are in-memory."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
RUNS=HERE.parents[1]/"runs"
sys.path.insert(0,str(RUNS))
import submit_nov20_v173_check as r
import test_nov20_v163_check as older

FIRST,V120,V139=older.IDS
V163=older.ACCEPTED
V168="5555555555555555"
V173="6666666666666666"


def replay(change=lambda f:None, issued=V173, flip=None, flip_on=2):
    with tempfile.TemporaryDirectory(prefix="okc-v173-review-") as path:
        f=older.Fake(path)
        assert f.offer()=="submitted "+V163
        f.set_state(r.fallback.OWN,V163,"pending")
        f.jobs[V163]["created_at"]+=5
        f.write(r.PARENT,V168)
        f.set_state(r.PARENT,V168,"pass")
        f.quota["limits"]["checks"]=dict(r.slots.THIRD)
        f.reads=0
        f.queries=[]
        f.attempts=[]
        f.on_quota=lambda count: (flip(f) if flip and count==flip_on else None)
        change(f)
        def execute(cmd,**opts):
            if cmd[1:3] != ["serve","check"]:
                return f.execute(cmd,**opts)
            assert cmd[0]=="in-memory" and cmd[4:6]==["--users","16"]
            f.attempts.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, f"✓ submitted {cmd[3]} as a check run at 16 users · job {issued}","")
        def run():
            return r.run_once(okc="in-memory",runs=f.runs,clock=lambda:f.now,execute=execute)
        initial=run()
        receipt=f.runs/(r.NAME+".submit.txt")
        data=None
        if receipt.is_file():
            try:
                data=json.loads(receipt.read_text().splitlines()[0])
            except (ValueError,IndexError):
                data={"incomplete_attempt":True}
        return initial, len(f.attempts), data, run()


def main():
    cases={}
    def job(job,**fields):
        return lambda f:f.jobs[job].update(fields)
    def pr(**fields):
        return lambda f:f.jobs[V168]["result"].update(fields)
    def rung(**fields):
        return lambda f:f.jobs[V168]["result"]["rungs"][0].update(fields)
    def later(**fields):
        return lambda f:f.jobs[V168]["result"]["later_turns"].update(fields)
    def evidence(**fields):
        def edit(f):
            p=f.runs/(r.fallback.NAME+".submit.txt");head,tail=p.read_text().split("\n",1)
            h=json.loads(head);h["evidence"].update(fields);p.write_text(json.dumps(h)+"\n"+tail)
        return edit
    mutations={
        "v168_pending":lambda f:f.set_state(r.PARENT,V168,"pending"),
        "v168_wrong_flags":job(V168,serving_flags={}),
        "v168_duplicate":job(V168,duplicate_of=FIRST),
        "v168_refunded":job(V168,refunded=True),
        "v168_later_incomplete":later(incomplete=True),"v168_later_failed":later(passed=False),
        "v168_correctness_failed":pr(correctness={"passed":False}),
        "v168_rung_nan":rung(tok_s_per_gpu=float("nan")),"v168_rung_bool":rung(tok_s_per_user_e2e=True),
        "v168_rung_infinite":rung(tok_s_per_user_e2e_p50=float("inf")),
        "v168_ttft_infinite":rung(ttft_first_p90_s=float("inf")),
        "v168_rung_truncated":rung(truncated=True),
        "v168_wrong_board":job(V168,board="kernel"),"v168_wrong_moment":job(V168,created_at=r.START.timestamp()+15),
        "v129_refunded":job(FIRST,refunded=True),"v163_refunded":job(V163,refunded=True),
        "v163_wrong_users":job(V163,users="16"),"v163_earlier_than_first":job(V163,created_at=r.START.timestamp()),
        "v163_wrong_slot_proof":evidence(slot=2.0),"v163_srcref":evidence(v129_job=V163),
        "v163_missing_control":evidence(failed_controls={"v120b":V120}),
        "competing_control_recovered":lambda f:f.set_state(r.REQUIRED[1],V120,"pass"),
        "competing_control_running":lambda f:f.set_state(r.REQUIRED[2],V139,"pending"),
        "v163_unknown_result":job(V163,result={}),
        "orphan_v132":lambda f:(f.runs/"v132_nov20_check32.submit.txt").write_text("timeout after POST"),
        "archived_v173":lambda f:(f.runs/"v173_nov20_check16.submit.txt").write_text("timeout after POST"),
        "bound_parent_conflict":lambda f:(f.runs/"other.submit.txt").write_text("job "+V168),
        "quota_wrongday":lambda f:f.quota.update(resets_at="2026-11-20T00:00:00Z"),
        "quota_extra":lambda f:f.quota["limits"]["checks"].update(used=1,left=2),
    }
    pos,n,proof,again=replay()
    assert pos=="submitted "+V173 and n==1 and again=="attempt recorded"
    assert proof["evidence"]["comparison_job"]==V168
    assert proof["evidence"]["v163_job"]==V163 and proof["evidence"]["comparison_rung"]["concurrency"]==16
    cases["positive_replay_and_idempotency"]=True
    for name,change in mutations.items():
        state,n,proof,again=replay(change)
        expected_receipt=name=="archived_v173"
        assert n==0 and (proof is None if not expected_receipt else True),(name,state,proof)
        cases[name]=True
    for prior in (FIRST,V163,V168):
        state,n,proof,second=replay(issued=prior)
        assert "manual reconciliation" in state and n==1 and second=="attempt recorded",(prior,state)
        cases["ambiguous_returned_id_"+prior]=True
    for name,change in (("parent_fails",rung(tok_s_per_gpu=0)),("quota_flips",lambda f:f.quota["limits"]["checks"].update(left=0)),
                        ("second_receipt_disappears",lambda f:(f.runs/(r.fallback.NAME+".submit.txt")).unlink())):
        state,n,proof,_=replay(flip=change,flip_on=1 if name=="parent_fails" else 2)
        assert n==0 and proof is None,(name,state)
        cases["last_recheck_"+name]=True
    return dict(runner_sha256=hashlib.sha256((RUNS/"submit_nov20_v173_check.py").read_bytes()).hexdigest(),
                probes=len(cases),probed=list(cases),live_calls=0,mode="temporary directories and isolated fake status/check functions")

if __name__=="__main__":
    print(json.dumps(main(),indent=2))
