"""Run the inherited full mixed-scheduler fixture on the assembled V202 sources."""
import ast
from contextvars import ContextVar
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import torch


HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1])/"python/sglang/srt"
suite = HERE.parent/"v160_mixed_dflash_serving/independent_round.py"
spec = importlib.util.spec_from_file_location("joined_round",suite)
test = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = test
spec.loader.exec_module(test)
projection = src/"layers/dflash_context_projection.py"
defs = [node for node in ast.parse(projection.read_bytes()).body if getattr(node,"name",None)
        in ("dflash_sp_prefill_scope","project_dflash_prefill_shard")]
runtime = dict(contextmanager=contextmanager,_prefill_context=ContextVar("context",default=None),
               _draft_kv_helper=ContextVar("kv",default=None))
exec(compile(ast.Module(defs,[]),str(projection),"exec"),runtime)
test.settle.env["dflash_sp_prefill_scope"] = runtime["dflash_sp_prefill_scope"]
test.settle.DF._sp_prefill_context = None
test.settle.DF._target_logits_processor = None
test.ref.PAR.tp_group = object()
test.ref.env["project_dflash_prefill_shard"] = runtime["project_dflash_prefill_shard"]

if __name__=="__main__":
    cases = []
    with torch.inference_mode():
        for backend in ("FULL","BREAKABLE"):
            for tails in (1,2,4):
                for index in range(8):
                    for mirror in (True,False):
                        cases.append(test.replay(backend,tails,index,mirror))
    print(json.dumps(dict(rounds=len(cases),committed_tokens=sum(x[2] for x in cases),
                          backends=["Full","Breakable"],physical_h200=False),indent=2))
