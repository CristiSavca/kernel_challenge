"""Exercise the existing Breakable session fixture at the exact V232 token keys."""
import ast
import inspect
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v222_packed_prompt_taps'))
import sessions

keys=(16384,16392,16400,16416,16448,16512,16640)
runner_classes=sessions.classes
def sized_classes(*args):
    rt, model, env=runner_classes(*args)
    rt.max_num_tokens=max(keys)
    return rt,model,env

# Only replace the fixture's four artificial key sizes, replay sequence and
# expected leading capacity. Its native model/backend capture and replay are untouched.
src=inspect.getsource(sessions.test)
for old,new in [('(264,129,17,7)','tuple(reversed(keys))'),('(7,129,17,264,7)','keys + (16384,)'),
                ('observed==[33,33,264]','observed==[2080,2080,16640]')]:
    assert src.count(old)==1
    src=src.replace(old,new)
scope=dict(vars(sessions),classes=sized_classes,keys=keys)
exec(compile(ast.parse(src),'<parameterized session fixture>','exec'),scope)
torch.set_num_threads(1)
out=scope['test'](sys.argv[1])
print(json.dumps(dict(**out,capture_keys=keys,model_hidden_width=16,public_rank_hidden_width=8192),indent=2))
