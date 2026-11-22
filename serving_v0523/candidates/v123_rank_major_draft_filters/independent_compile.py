"""Trace the actual clean projection guard, then exercise a local forward override."""
import ast
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn


def audit(stage):
    code=Path(stage)/'python/sglang/srt/models/dflash.py'
    cls=next(n for n in ast.parse(code.read_text()).body if getattr(n,'name',None)=='DFlashGroupedConv')
    method=next(n for n in cls.body if getattr(n,'name',None)=='_can_use_rank_major')
    original=ast.unparse(method)
    lines=original.splitlines()
    alternatives={'as_submitted':original,
                  'dict_attribute':original.replace('vars(projection)','projection.__dict__'),
                  'outer_compile_inherits_parent':'\n'.join([lines[0],'    if torch.compiler.is_compiling():','        return False']+lines[1:])}
    class DFlashGatheredLinear(nn.Linear):
        pass
    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel_projection=DFlashGatheredLinear(32,8,bias=False)
            self.kernel_projection.tp_group=types.SimpleNamespace(world_size=8)
            self.taps=2
        def forward(self,x):
            return x+1 if self._can_use_rank_major(x) else x-1
    receipts={}
    for name,source in alternatives.items():
        glob={'DFlashGatheredLinear':DFlashGatheredLinear,'torch':torch}
        exec(compile(source,str(code),'exec'),glob)
        Conv._can_use_rank_major=glob['_can_use_rank_major']
        torch._dynamo.reset()
        model=Conv()
        with torch.inference_mode():
            inp=torch.zeros(8,32)
            compiled=torch.compile(model,backend='eager',fullgraph=True)
            try:
                first=compiled(inp)[0,0].item()
                model.kernel_projection.forward=lambda x:x
                second=compiled(inp)[0,0].item()
                receipts[name]={'fullgraph_traced':True,'clean':first,'override':second}
            except torch._dynamo.exc.Unsupported as err:
                receipts[name]={'fullgraph_traced':False,'reason':str(err).splitlines()[0]}
    return receipts


if __name__=='__main__':
    print(json.dumps({'torch':torch.__version__,'backend':'Dynamo fullgraph eager','results':audit(sys.argv[1])},indent=2))
