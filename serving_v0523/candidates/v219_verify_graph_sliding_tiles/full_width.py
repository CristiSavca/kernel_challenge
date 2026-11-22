"""Compare the real row and channel dimensions of TP8 verify FC on CPU."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def check(stage, receipt):
    src = stage / "python/sglang/srt/layers/dflash_context_projection.py"
    code = src.read_text()
    cls = next(n for n in ast.parse(code).body if getattr(n,"name",None)=="DFlashVerifyFCGraph")
    env = dict(torch=torch)
    exec("from __future__ import annotations\n"+ast.unparse(cls),env)
    Graph = env["DFlashVerifyFCGraph"]
    payload = json.loads(receipt.read_bytes())["witness"]
    words = torch.tensor(payload["words"],dtype=torch.int32).to(torch.int16)
    assert hashlib.sha256(words.numpy().tobytes()).hexdigest()==payload["sha256"]
    torch.set_num_threads(4)
    generator=torch.Generator().manual_seed(219)
    with torch.inference_mode():
        # Fill in small slices, bounding conversion scratch for the full 640 MiB FC.
        weight=torch.empty((8192,40960),dtype=torch.bfloat16)
        for chunk in weight.split(128):
            chunk.copy_(torch.randn(chunk.shape,generator=generator)*.01)
        base=words.view(torch.bfloat16)
        tap_bank=base.repeat(160).view(8,40960)
        raw=torch.stack([row.roll(37*i) for i,row in enumerate(tap_bank)])
        ref=torch.nn.functional.linear(raw,weight)
        record=[]
        norm=torch.nn.RMSNorm(8192,dtype=torch.bfloat16)
        for rank,shard in enumerate(weight.chunk(8,dim=0)):
            graph=Graph.__new__(Graph)
            graph.local_weight=shard.clone()
            out=NS(hidden_states=raw,dflash_verify_fc_shard=None)
            graph.capture(out,NS(dflash_verify_raw_buffer=raw))
            assert torch.equal(out.dflash_verify_fc_shard.view(torch.int16),
                               ref[:,rank*1024:(rank+1)*1024].view(torch.int16))
            record.append(out.dflash_verify_fc_shard)
        def gather(local,dim):
            assert dim==-1 and local is record[7]
            return torch.cat(record,dim=dim)
        graph.context=NS(group=NS(all_gather=gather),draft=NS(hidden_norm=norm))
        graph.ready=lambda:True
        result=graph.finish(record[-1],raw)
        assert torch.equal(result.view(torch.int16),norm(ref).view(torch.int16))
        assert len(torch.unique(result.view(torch.int16)))>250
    return dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                input_witness_sha256=payload["sha256"],weight_shape=list(weight.shape),ranks=8,verify_rows=8,
                fc_elements=ref.numel(),normalized_elements=result.numel(),
                normalized_sha256=hashlib.sha256(result.view(torch.int16).numpy().tobytes()).hexdigest(),
                device="CPU",torch_version=str(torch.__version__),
                scope="Full BF16 matrix and production capture/finish math. Channel gather emulated on one host.")


if __name__=="__main__":
    print(json.dumps(check(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
