"""Run the production workspace constructor with the pinned build, alternate build and optional JIT failures."""
import ast
from contextlib import ExitStack, nullcontext
from hashlib import sha256
import json
from pathlib import Path
import sys
import threading
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import patch

import torch

stage=Path(sys.argv[1])
path="python/sglang/srt/layers/muse_spark/peer_prompt_pack.py"
src=(stage/path).read_bytes()
node=next(n for n in ast.parse(src).body if getattr(n,"name",None)=="PeerPromptInputs")
scope=dict(torch=torch,threading=threading)
exec(ast.unparse(node),scope)
constructor=scope["PeerPromptInputs"]
pin="5c4886908584029761b579af026dcfb627c84070"
module=ModuleType("sglang.srt.layers.muse_spark.muse_spark_humming_moe")
def failure():raise RuntimeError("injected JIT diagnostic")
cases=[]
for rows,width,world,mode in ((2048,4096,8,"native"),(2080,4096,8,"native"),(16,32,4,"native"),(16,36,4,"unaligned_width"),
        (16,32,4,"foreign"),(16,32,4,"unavailable"),(16,32,4,"jit_failure"),(16,32,4,"map_failure")):
    local=torch.empty(2*rows,width,dtype=torch.bfloat16)
    group=NS(rank_in_group=0,world_size=world)
    other=torch.empty_like(local)
    handle=NS(get_buffer=lambda idx,*a:local if idx==0 else other)
    identity=object()
    module._ranked_pool_module=failure if mode=="jit_failure" else lambda:None if mode=="unavailable" else identity
    dtype_map=patch.object(torch,"arange",side_effect=RuntimeError("identity allocation fault")) if mode=="map_failure" else nullcontext()
    with patch.dict(sys.modules,{module.__name__:module}),patch.object(torch.cuda,"Stream",lambda **kw:object()),\
            patch.object(torch.version,"git_version","other" if mode=="foreign" else pin),dtype_map:
        ws=constructor(local,handle,group,2)
        assert len(ws.banks)==2 and ws.storage.shape==(rows,width)
        if mode=="native":
            assert ws.norm_module is identity and ws.norm_rows.dtype==torch.int32
            assert ws.norm_rows.is_contiguous() and torch.equal(ws.norm_rows,torch.arange(rows,dtype=torch.int32).view(-1,1))
        else:assert ws.norm_module is None and ws.norm_rows is None
    cases.append(dict(rows=rows,width=width,world=world,mode=mode,normalization=ws.norm_module is not None))
print(json.dumps(dict(cases=cases,source=sha256(src).hexdigest(),scope="Real constructor and Torch views; only compiled-module supply, stream and peer handle modeled."),indent=2))
