"""Compare both full-size BF16 matrices with rank-ordered full-K output slices on CPU."""
import json

import torch
from torch.nn import functional as F

torch.set_num_threads(4)
torch.manual_seed(228)
rows = []
with torch.inference_mode():
    for role, width, inner in (("pool",4096,8192),("unpool",8192,4096)):
        # Source is a live contiguous BF16 checkpoint, just as in the capture admission.
        weight = torch.empty((width, inner), dtype=torch.bfloat16).normal_(std=inner ** -0.5)
        for n in (8,32,128,256):
            data = torch.randn(n, inner + 8, dtype=torch.bfloat16)[:, :inner]
            full = F.linear(data, weight)
            for size in (4, 8):
                stripes = [F.linear(data, weight.narrow(0, i * (width // size), width // size)) for i in range(size)]
                actual = torch.cat(stripes, dim=-1)
                mismatches = (actual.view(torch.int16) != full.view(torch.int16)).sum().item()
                rows.append(dict(role=role,rows=n,width=width,inner=inner,tp=size,compared=actual.numel(),
                                 differing_bf16=mismatches,max_abs=(actual.float()-full.float()).abs().max().item()))
print(json.dumps(dict(torch=torch.__version__,device="CPU",method="PyTorch CPU linear; both stored matrices at real dimensions with full K and BF16 output",
    total_compared=sum(r['compared'] for r in rows), differing_bf16=sum(r['differing_bf16'] for r in rows),
    cases=rows,physical_h200=False),indent=2))
