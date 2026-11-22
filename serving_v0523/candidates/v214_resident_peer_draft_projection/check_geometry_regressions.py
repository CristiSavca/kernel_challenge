"""Run the original transport decline and recovery suite with exact live-prefix byte accounting."""
from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile

import torch.multiprocessing as mp

from transport_check import STAGE, v185


def worker(rank,size,folder):
    source=inspect.getsource(v185.worker)
    old='assert torch.equal(got, expected) and comm.sent-before==traffic*w.element_size()'
    assert source.count(old)==1
    accounting='''width=volumes[traffic]
                live=max(0,min(rows,n-rank*rows))
                cohort=max(0,min(width*rows,n-(rank//width)*width*rows))
                sent=(width-1)*live*hidden+(size//width-1)*(hidden*channels+cohort*channels)
                assert torch.equal(got,expected) and comm.sent-before==sent*w.element_size()'''
    ns=vars(v185).copy()
    exec(compile(source.replace(old,accounting),v185.__file__,"exec"),ns)
    ns["worker"](rank,size,folder)


if __name__=="__main__":
    reports={}
    for size in [4,8]:
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker,args=(size,folder),nprocs=size,join=True)
            reports[size]=json.loads((Path(folder)/"result.json").read_bytes())
    path=STAGE/"python/sglang/srt/layers/dflash_context_projection.py"
    print(json.dumps(dict(sha256=sha256(path.read_bytes()).hexdigest(), by_world=reports, physical_H200=False), indent=2))
