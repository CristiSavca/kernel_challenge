"""Rerun V194's continuation suite with the two appended V202 config fields."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

import torch
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
parent = next(HERE.parent.glob("v160_*/scheduler_admission_cpu.py"))
continuation = next(HERE.parent.glob("v194_*/check_scheduler.py"))
original = parent.read_text()
old_order = "positional(old)+('speculative_dflash_mixed_prefill',)"
new_order = "positional(old)+('speculative_dflash_sp_prefill_context','speculative_dflash_mixed_prefill')"
assert original.count(old_order)==1

class JoinedLoader:
    def create_module(self,spec):
        return None
    def exec_module(self,module):
        module.__file__ = str(parent)
        exec(compile(original.replace(old_order,new_order),str(parent),"exec"),module.__dict__)

native_spec = importlib.util.spec_from_file_location
def get_spec(name, path, *args, **kwargs):
    if name == "parent_admission" and Path(path) == parent:
        return importlib.util.spec_from_loader(name,JoinedLoader(),origin=str(parent))
    return native_spec(name,path,*args,**kwargs)

v194 = ModuleType("joined_continuation")
v194.__file__ = str(continuation)
sys.modules[v194.__name__] = v194
with patch.object(importlib.util,"spec_from_file_location",get_spec):
    exec(compile(continuation.read_bytes(),str(continuation),"exec"),v194.__dict__)

def worker(rank,size,path):
    v194.worker(rank,size,path)

if __name__ == "__main__":
    local = v194.local()
    multi = {}
    for n in (2,4,8):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"continuation")
            mp.spawn(worker,(n,path),nprocs=n,join=True)
            multi[n] = json.loads(Path(path+".json").read_bytes())
    print(json.dumps(dict(single_rank=local,multi_rank=multi,torch=torch.__version__,physical_h200=False),indent=2))
