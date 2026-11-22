"""Whole-module Dynamo trace of production prepare/project/finish, including hook fallback."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
loader_spec = importlib.util.spec_from_file_location('v123_fixture', HERE.parent / 'v123_rank_major_draft_filters/check.py')
fixture = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(fixture)


class TpEight:
    world_size = 8
    rank_in_group = 0

    def all_gather(self, projected, dim):
        # The permutations in these original-sized representations are observable.
        return torch.cat([projected + rank*.125 for rank in range(self.world_size)], dim)


class Filter(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, hidden, tail):
        first, coefficients = self.conv.prepare(hidden)
        return first, self.conv.finish(tail, coefficients)


def run(stage):
    receipt=[]
    torch.set_num_threads(1)
    group=TpEight()
    namespace=fixture.production(stage,group)
    reference_conv=namespace['_grouped_conv']
    # Real Dynamo sees the nested compiled convolution callable inside the full outer graph.
    for name in ('_grouped_conv','_rank_major_grouped_conv'):
        namespace[name]=torch.compile(namespace[name],dynamic=True,backend='eager')
    for dtype in (torch.bfloat16,torch.float32):
        for hook in ('none','output','forward_override'):
            torch._dynamo.reset()
            conv=namespace['DFlashGroupedConv'](64,8,2,4,shard_filter=True).to(dtype)
            model=Filter(conv)
            with torch.no_grad():
                weight=conv.kernel_projection.weight
                weight.copy_((torch.arange(weight.numel()).reshape(weight.shape)%7-3)/4)
            if hook=='output':
                conv.kernel_projection.register_forward_hook(lambda module, inputs, out: out+.25)
            if hook=='forward_override':
                original=conv.kernel_projection.forward
                conv.kernel_projection.forward=lambda x: original(x)+.5
            graphs=[]
            def backend(gm, inputs):
                dim=[n.args[1] if len(n.args)>1 else n.kwargs.get('dim',0)
                     for n in gm.graph.nodes if n.target is torch.cat]
                graphs.append(dim)
                return gm.forward
            compiled=torch.compile(model, backend=backend, fullgraph=True)
            with torch.inference_mode():
                for rows in (8,16):
                    x=(torch.arange(rows*64).reshape(rows,64)%11-5).to(dtype)/4
                    tail=x.flip([1])
                    delta=conv.kernel_projection(x).view(rows,2,2,16)
                    raw=tuple(reference_conv(h,delta[:,side],conv.base_kernel[side],8,16,4,2) for side,h in enumerate((x,tail)))
                    got=compiled(x,tail)
                    assert all(torch.equal(a,b) for a,b in zip(raw,got)),(dtype,hook,rows)
            assert graphs and all(x==[-1] for x in graphs),(dtype,hook,graphs)
            receipt.append({'dtype':str(dtype),'preexisting_instrumentation':hook,'graphs':len(graphs),
                            'gather_dim':-1,'rows':[8,16],'matched_eager':True})
    return receipt


if __name__=='__main__':
    print(json.dumps({'torch':torch.__version__,'fullgraph_results':run(sys.argv[1])},indent=2))
