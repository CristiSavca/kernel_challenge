"""Drive a mixed DFLASH target with finalized grouped decode-expert values."""
from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.multiprocessing as mp


HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()


def prepare_parent():
    m=next(HERE.parent.glob("v212_*/validation.json"))
    manifest=json.loads(m.read_bytes())
    with tempfile.TemporaryDirectory() as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/"index"))
        def git(*a):return subprocess.check_output(["git","-C",STAGE,*a],env=env)
        git("read-tree",manifest["base"])
        git("apply","--cached","--whitespace=error-all",str(m.with_name("patch.diff")))
        assert git("write-tree").decode().strip()==manifest["tree"]


def actual_expert_bank():
    path=next(HERE.parent.glob("v191_*/dispatch_check.py"))
    source=path.read_text()
    capture="    summary[mode]=dict("
    assert source.count(capture)==1
    # Observe the fixture's already-checked outputs at its device boundary.
    source=source.replace(capture,'    if mode == "all":\n        save(env["actual"],env["expected"])\n'+capture)
    seen=[]
    def save(actual,expected):
        a=torch.tensor(actual.copy()).bfloat16()
        assert torch.equal(a.view(torch.int16),torch.tensor(expected.copy()).bfloat16().view(torch.int16))
        assert a.shape==(256,8) and torch.isfinite(a).all() and (a.abs().sum(1)>0).all()
        seen.append(a)
    scope=dict(__file__=str(path),__name__="decode_fixture",save=save)
    with redirect_stdout(io.StringIO()):
        exec(compile(source,str(path),"exec"),scope)
    assert len(seen)==1
    return seen[0],scope["summary"]


def worker(rank,size,folder):
    path=next(HERE.parent.glob("v214_*/mixed_transition.py"))
    spec=importlib.util.spec_from_file_location("mixed_transition",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    states=torch.load(Path(folder)/"rows.pt",weights_only=True)
    orig=module.previous.load
    def load(name, path):
        part=orig(name,path)
        if name=="v212_scheduler":
            assert part.ref is part.settle.ref
            def body(ids):
                x=states.index_select(0,ids.long().remainder(len(states)))
                return x,[x*i for i in (1,2,3,4,5)]
            part.ref.body_tokens=body
        return part
    module.previous.load=load
    module.worker(rank,size,folder)


if __name__=="__main__":
    prepare_parent()
    states,dispatch=actual_expert_bank()
    worlds={}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as folder:
            torch.save(states,Path(folder)/"rows.pt")
            mp.spawn(worker,args=(size,folder),nprocs=size,join=True)
            result=json.loads(Path(folder,"out.json").read_bytes())
            worlds[size]=dict(comparisons=len(result),rounds=2*len(result),
                physical_cache_words_per_rank=sum(x["cache_words"] for x in result),
                rejected_slots=sum(x["rejected"] for x in result),
                cases=sorted({x["case"] for x in result}),
                padded_full=[{k:x[k] for k in ("case","paired","base_paired","weight_sends","wire_bytes")}
                             for x in result if x["backend"]=="FULL" and x["case"].startswith("padded_")])
    print(json.dumps(dict(full_patch_sha256=sha256((HERE/"patch.diff").read_bytes()).hexdigest(),
        bf16_expert_bank_words=states.numel(),bank_digest=sha256(states.view(torch.int16).numpy().tobytes()).hexdigest(),
        decode_dispatch=dispatch,by_world=worlds,physical_H200=False,
        scope="Final source adapters and host weighted finalizer, CPU-modeled projection kernels. Actual joined scheduler "
              "with modeled transformer body and graph/PyNCCL boundaries, real four/eight-rank Gloo."),indent=2))
