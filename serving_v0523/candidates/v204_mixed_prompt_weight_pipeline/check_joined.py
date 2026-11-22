"""Feed concrete grouped-expert results to both real mixed-head settlement paths."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
def fixture(name,pattern):
    path = next(HERE.parent.glob(pattern))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def expert_rows():
    exp = fixture("joined_experts","v199_*/independent_callback.py")
    layer = exp.layer
    rows = 8192
    x = exp.ns["rand"](rows,128)
    scores = exp.ns["rand"](rows,8).float()
    ids = torch.full((rows,8),-1,dtype=torch.long)
    ids[:80,:2] = torch.stack((torch.arange(80)%8,(torch.arange(80)+3)%8),dim=1)
    ids[80::73,0] = 2
    layer.humming_prompt_down = layer.humming_prompt_gate_up = None
    with torch.inference_mode(), patch.object(torch.cuda,"is_current_stream_capturing",return_value=False):
        ordinary = exp.forward(x,scores,ids)
        layer.humming_prompt_gate_up = exp.ns["pstate"]("paired")
        layer.humming_prompt_down = exp.bind()
        exp.events.clear()
        out = exp.forward(x,scores,ids)
    assert [e[0] for e in exp.events] == ["paired","W2 custom"]
    assert out.dtype==torch.bfloat16 and torch.equal(out,ordinary) and torch.isfinite(out).all()
    sample = out[:80].contiguous().clone()
    assert (sample.float().abs().sum(1)>0).all()
    return sample


def run_rank(rank,size,folder):
    suite = fixture("joined_vocab","v203_*/independent_gloo.py")
    rows = torch.load(Path(folder)/"expert.pt",weights_only=True).float()
    def projected(n,cols,valid,prompts,shard,world,with_nan):
        assert n<=len(rows) and not with_nan
        # A seeded full BF16 vocabulary projection maps the actual finalized
        # expert values to observable scores shared by both native branch tests.
        gen = torch.Generator().manual_seed(4737+cols)
        weight = torch.randn(cols,rows.shape[1],generator=gen).bfloat16().float()
        logits = rows[:n] @ weight.T
        logits[:,valid:] = float("nan")
        return logits
    suite.raw_logits = projected
    suite.worker(rank,size,str(Path(folder)/"gloo"),str(Path(folder)/"result.json"))


if __name__=="__main__":
    states = expert_rows()
    worlds = {}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(states,Path(tmp)/"expert.pt")
            mp.spawn(run_rank,args=(size,tmp),nprocs=size,join=True)
            worlds[size]=json.loads((Path(tmp)/"result.json").read_bytes())
    print(json.dumps(dict(expert_input_rows=8192,finalized_bf16_words_compared=8192*128,
        connected_bf16_words=states.numel(),connected_sha256=sha256(states.view(torch.int16).numpy().tobytes()).hexdigest(),
        worlds=worlds,physical_h200=False,
        scope="CPU-modeled expert matrix and finalizer operations, real source dispatch, seeded synthetic local LM head, real Gloo full/sharded verification."),indent=2))
