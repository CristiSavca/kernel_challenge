"""Reproduce the V68 ablation, complete argument parser and native scheduler decisions."""
import ast
import fnmatch
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace as S

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
rules=json.loads(Path(sys.argv[2]).read_bytes())
pin=rules["base"]["commit"]
parent_dir=next(here.parent.glob("v68_*"))
parent=json.loads((parent_dir/"validation.json").read_text())["trees"]["v68"]
srt="python/sglang/srt/"
def git(*args,env=None):
    return subprocess.check_output(["git","-C",stage,*args],env=env)
def sha(data):
    return hashlib.sha256(data).hexdigest()
with tempfile.TemporaryDirectory() as scratch:
    obj=Path(scratch)/"objects";obj.mkdir()
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(scratch)/"index"),GIT_OBJECT_DIRECTORY=str(obj),
             GIT_ALTERNATE_OBJECT_DIRECTORIES=str(stage/".git/objects"))
    def apply(ref,patch,reverse=False):
        git("read-tree",ref,env=env)
        git("apply","--cached","--unidiff-zero",*(["-R"] if reverse else []),str(patch),env=env)
        return git("write-tree",env=env).decode().strip()
    tree=apply(pin,here/"patch.diff")
    assert apply(pin,parent_dir/"patch.diff")==parent
    assert apply(parent,here/"delta.diff")==tree
    assert apply(tree,here/"patch.diff",True)==git("rev-parse",pin+"^{tree}").decode().strip()
    assert git("diff","--no-renames","-U1",pin,tree,env=env)==(here/"patch.diff").read_bytes()
diff=(here/"patch.diff").read_bytes()
assert len(diff)<=rules["rules"]["max_patch_bytes"]
paths=git("diff","--name-only",pin,tree).decode().splitlines()
added=[line[1:] for line in diff.decode().splitlines() if line.startswith("+") and not line.startswith("+++")]
assert max(map(len,added))<=rules["rules"]["max_line_chars"]
assert all(word not in diff.decode() for word in rules["rules"]["forbidden_words"])
for p in paths:
    assert p.startswith(rules["rules"]["allowed_prefix"]) and all(not p.startswith(d) for d in rules["rules"]["forbidden_prefixes"])
    assert Path(p).suffix in rules["rules"]["allowed_suffixes"]
    if p.endswith(".py"):
        compile(git("show",tree+":"+p),p,"exec")
delta_paths=git("diff","--name-only",parent,tree).decode().splitlines()
assert delta_paths==[srt+p for p in ("arg_groups/field_order.py","arg_groups/fields/schedule.py","managers/scheduler.py")]
flags={"--enable-layernorm-sp":"","--moe-runner-backend":"humming","--speculative-moe-runner-backend":"marlin",
       "--swa-full-tokens-ratio":"0.25","--prefill-decode-interval":"1","--prefill-decode-interval-min-bs":"1",
       "--prefill-decode-interval-on-completion":""}
excluded=re.findall(r"--[\w*.-]+",rules["launch_flags"]["rules"])
assert all(not any(fnmatch.fnmatch(k,p) for p in excluded) for k in flags)
spec=importlib.util.spec_from_file_location("registration",next(here.parent.glob("v176_*/independent_registration.py")))
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
key="prefill_decode_interval_on_completion"
cli.FLAG_FIELDS["schedule"]=("Schedule",("swa_full_tokens_ratio","prefill_decode_interval","prefill_decode_interval_min_bs",key))
prev_order,*_=cli.parser(parent); order,parser,_,defaults,ns=cli.parser(tree)
assert order==(*prev_order,key) and defaults[key] is False
for names in itertools.permutations(flags):
    argv=[v for k in names for v in ((k,flags[k]) if flags[k] else (k,))]
    values=vars(parser.parse_args(argv))
    assert values[key] is True and values["prefill_decode_interval"]==1 and values["prefill_decode_interval_min_bs"]==1
    assert all(values[k]==v for k,v in cli.EXPECTED.items())
methods=("init_chunked_prefill","_arm_prefill_decode_interval","_should_defer_prefill")
def code(ref):
    raw=git("show",ref+":"+srt+"managers/scheduler.py").decode()
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=="Scheduler")
    natives={f.name:f for f in node.body if getattr(f,"name","") in methods+("get_next_batch_to_run",)}
    assert len(natives)==4
    return raw,natives

r,nodes=code(tree); before,base_nodes=code(parent)
assert ast.dump(nodes["get_next_batch_to_run"])==ast.dump(base_nodes["get_next_batch_to_run"])
assert 'self.chunked_req is None or len(can_run_list) != 1' in r
assert r.index("self.init_moe_gemm_config()") < r.index("self.init_chunked_prefill()")
assert "self.require_mlp_sync = require_mlp_sync()" in r
assert "req.prefix_len = req.extend_end" in git("show",tree+":"+srt+"mem_cache/common.py").decode()
from enum import Enum
class Mode(Enum):
    EXTEND=1; MIXED=2; DECODE=3; IDLE=4
    def is_extend(self):
        return self in (Mode.EXTEND,Mode.MIXED)
class Disagg(Enum):
    NULL="null"; PREFILL="prefill"; DECODE="decode"
ctx=dict(ForwardMode=Mode,DisaggregationMode=Disagg,ModelImpl=S(TRANSFORMERS="transformers"),get_resolved_model_impl=lambda x:"sglang")
def extract(ns,name):
    f=ns[name]
    f.decorator_list=[]
    exec("from __future__ import annotations\n"+ast.unparse(f),ctx)
    return ctx[name]
candidate={m:extract(nodes,m) for m in methods}; control={m:extract(base_nodes,m) for m in methods}
cases=0; diffs=0
for configured,minbs,size,model,global_extend,flag,partial,pp,disagg,sync in itertools.product(
        (0,1,3),(0,1,4),(0,1,4),(*Mode,None),(False,True),(False,True),(False,True),(1,2),(*Disagg,),(False,True)):
    cfg=S(prefill_decode_interval=configured,prefill_decode_interval_min_bs=minbs,
          prefill_decode_interval_on_completion=flag,chunked_prefill_size=16384,enable_mixed_chunk=False)
    ctx.update(get_parallel=lambda:S(pp_size=pp),get_schedule=lambda:cfg,get_disagg=lambda:S(disaggregation_mode=disagg.value))
    x=S(require_mlp_sync=sync,model_config=S(is_multimodal=False)); y=S(require_mlp_sync=sync,model_config=S(is_multimodal=False))
    candidate[methods[0]](x); control[methods[0]](y)
    batch=S(forward_mode=model,is_extend_in_batch=global_extend,contains_last_prefill_chunk=not partial) if model else None
    run=S(batch_size=lambda:size)
    candidate[methods[1]](x,batch);control[methods[1]](y,batch)
    observed=[candidate[methods[2]](x,run) for _ in range(4)]
    stock=[control[methods[2]](y,run) for _ in range(4)]
    suppressed=flag and pp==1 and disagg==Disagg.NULL and not sync and model==Mode.EXTEND and partial
    assert observed==([False]*4 if suppressed else stock)
    diffs+=observed!=stock; cases+=1
print(json.dumps(dict(candidate="V287",parent="V68",pin=pin,parent_tree=parent,tree=tree,paths=paths,delta_paths=delta_paths,
    patch_bytes=len(diff),max_added_line=max(map(len,added)),patch_sha256=sha(diff),delta_sha256=sha((here/"delta.diff").read_bytes()),
    source_sha256={p:sha(git("show",tree+":"+p)) for p in delta_paths}, native_argument_fields=len(order),unchanged_parent_argument_fields=len(prev_order),
    parser_permutations=5040,flags=flags,native_scheduler_cases=cases,native_changed_round_sequences=diffs,
    pinned_batch_run_path_identical=True,gpu_execution=False),indent=2))
