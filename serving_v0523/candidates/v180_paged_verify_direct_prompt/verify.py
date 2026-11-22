"""Check the export, H200-measured parent, prompt donor and phase isolation."""

import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from assemble import BACK, HELPER, HERE, PIN, V68, V178B, V178B_SHA, construct

stage = Path(sys.argv[1]).resolve()
root = HERE.parents[1]
parent = next(HERE.parent.glob("v68_*/patch.diff"))
prompt = next(HERE.parent.glob("v178b_*/patch.diff"))
job = json.loads((root / "runs/04b01f958aa5487c.json").read_bytes())
result = job["result"]
digest = lambda data: hashlib.sha256(data).hexdigest()
assert job["status"] == "done" and result["passed"] and result["correctness"]["passed"]
assert result["patch"]["sha256"] == digest(parent.read_bytes()) and result["short_outputs"] == 0
assert digest(prompt.read_bytes()) == V178B_SHA
full, delta = ((HERE / p).read_bytes() for p in ("patch.diff", "delta.diff"))
url = "https://openkernelchallenge.dev/api/v1/serving"
published = json.loads(subprocess.check_output(["curl","-fsSL",url]))
rules = published["rules"]
assert PIN == published["base"]["commit"]

with tempfile.TemporaryDirectory() as tmp:
    idx = Path(tmp) / "index"
    tree = construct(stage, idx, prompt)
    env = {**os.environ, "GIT_INDEX_FILE": str(idx)}
    def git(*args, data=None):
        return subprocess.check_output(["git","-C",stage,*args],env=env,input=data)
    def apply(base, patch, reverse=False):
        git("read-tree",base)
        git("apply","--cached","--whitespace=error-all",*(["-R"] if reverse else []),"-",data=patch)
        return git("write-tree").decode().strip()
    assert apply(PIN,full) == apply(V68,delta) == tree
    assert apply(tree,full,True) == git("rev-parse",PIN+"^{tree}").decode().strip()
    assert apply(tree,delta,True) == V68
    paths = git("diff","--name-only",PIN,tree).decode().splitlines()
    changed = git("diff","--name-only",V68,tree).decode().splitlines()
    assert set(changed) == {HELPER,BACK}
    for p in paths:
        assert p.startswith(rules["allowed_prefix"]) and p.endswith(tuple(rules["allowed_suffixes"]))
        assert not any(p.startswith(s) for s in rules["forbidden_prefixes"])
        mode = git("ls-tree",PIN,p).split()
        assert git("ls-tree",tree,p).startswith((mode[0] if mode else b"100644")+b" blob ")
        data = git("show",tree+":"+p)
        assert data == (stage/p).read_bytes() and data
        if p.endswith(".py"): compile(data,p,"exec")
    assert git("show",tree+":"+HELPER) == git("show",V178B+":"+HELPER)
    def method(t,name):
        nodes=ast.parse(git("show",t+":"+BACK)).body
        cls=next(n for n in nodes if getattr(n,"name",None)=="FlashAttentionBackend")
        return next(n for n in cls.body if getattr(n,"name",None)==name)
    def pick(nodes,predicate):
        node, = [n for n in nodes if predicate(n)]
        return node
    def written(n,name):
        return isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==name
    old,new,donor=[method(t,"__init__") for t in (V68,tree,V178B)]
    policy=lambda n: written(n,"self._direct_fp8_prompt")
    extra=pick(new.body,policy)
    assert ast.dump(extra)==ast.dump(pick(donor.body,policy))
    new.body.remove(extra)
    assert ast.dump(new)==ast.dump(old)
    old,new,donor=[method(t,"forward_extend") for t in (V68,tree,V178B)]
    init=pick(new.body,lambda n:written(n,"direct_prompt_kv"))
    assert isinstance(init.value,ast.Constant) and init.value.value is None
    new.body.remove(init)
    writer=lambda n:isinstance(n,ast.If) and ast.unparse(n.test)=="k is not None"
    w=pick(new.body,writer)
    assert ast.dump(w)==ast.dump(pick(donor.body,writer))
    new.body[new.body.index(w)]=pick(old.body,writer)
    parent_fa=pick(ast.walk(new),lambda n:isinstance(n,ast.If) and ast.unparse(n.test)=="result is None")
    attachment=parent_fa.body.pop(0)
    assert ast.dump(attachment)==ast.dump(pick(ast.walk(donor),lambda n:isinstance(n,ast.If)
        and ast.unparse(n.test)=="direct_prompt_kv is not None" and any(isinstance(v,ast.Call)
        and ast.unparse(v.func)=="complete_direct_prompt_kv" for v in ast.walk(n))))
    assert ast.dump(new)==ast.dump(old)
lines=full.decode().splitlines()
added=[l[1:] for l in lines if l.startswith("+") and not l.startswith("+++ ")]
assert not any(l.startswith(("rename from ","rename to ","old mode ","new mode ","Binary files","GIT binary patch")) for l in lines)
assert len(full)<=rules["max_patch_bytes"] and max(map(len,added))<=rules["max_line_chars"]
assert not any(bad in l for l in added for bad in rules["forbidden_words"])
flags = job["serving_flags"]
donor_flags=json.loads((prompt.parent/"validation.json").read_bytes())["launch_flags"]
assert flags==donor_flags
denied=[p for patterns in published["launch_flags"]["denied"].values() for p in patterns]
assert not any(fnmatch.fnmatchcase(f,p) for f in flags for p in denied)
print(json.dumps(dict(candidate="V180",parent="V68",parent_sha256=digest(parent.read_bytes()),parent_tree=V68,
    base=PIN,tree=tree,patch_sha256=digest(full),bytes=len(full),delta_sha256=digest(delta),delta_bytes=len(delta),
    donor="V178b",donor_sha256=V178B_SHA,donor_tree=V178B,measured_parent_job=job["id"],
    parent_users=result["rungs"][0]["concurrency"],parent_tok_s_per_gpu=result["rungs"][0]["tok_s_per_gpu"],
    launch_flags=flags,paths=paths,changed_from_parent=changed,feature_off_ast_equals_parent=True,
    prompt_writer_and_fa3_operand_ast_equal_donor=True,strict_apply_reverse_full_delta=True,
    official_rules=url,official_export="okc serve diff",checked_flag_patterns=len(denied),
    max_added_line_chars=max(map(len,added)),physical_H200=False,submitted=False),indent=2))
