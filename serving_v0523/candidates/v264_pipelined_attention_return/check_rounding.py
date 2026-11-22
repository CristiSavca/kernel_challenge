"""Compare Gloo's native full and half-width SUM using cancellation-sensitive BF16 partials."""
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def reduce(x, kind, rank, size):
    if kind=="all_reduce":
        dist.all_reduce(x)
        return x.chunk(size, dim=0)[rank]
    dst=x.new_empty((x.shape[0]//size,x.shape[1]))
    dist.reduce_scatter_tensor(dst,x)
    return dst


def worker(rank, size, tmp):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(tmp)/"group"),rank=rank,world_size=size)
    # Includes cancellation across vastly different magnitudes and halfway additions.
    patterns=torch.tensor([
        [1024., 1., -1024., .5, .25, -.25, -.5, 1.],
        [8388608., 1., -8388608., .5, .25, -.25, -.5, 1.],
        [1., 1/256, 1/512, 1/1024, -1., -.5, .25, -.125],
    ],dtype=torch.bfloat16)
    records=[]
    for rows,width in ((128,256),(4096,512),(8192,1024)):
        ri=torch.arange(rows).view(rows,1)
        col=torch.arange(width).view(1,width)
        values=patterns[(ri+col)%3,(rank+ri%8+col%8)%8]
        for kind in ("all_reduce","reduce_scatter"):
            full=reduce(values.clone(),kind,rank,size)
            a=reduce(values[:,:width//2].contiguous(),kind,rank,size)
            b=reduce(values[:,width//2:].contiguous(),kind,rank,size)
            striped=torch.cat((a,b),dim=1)
            records.append(dict(rows=rows,width=width,kind=kind,words=striped.numel(),
                mismatches=int(torch.count_nonzero(full.view(torch.int16)!=striped.view(torch.int16))),
                max_abs=float((full.float()-striped.float()).abs().max())))
    (Path(tmp)/f"r{rank}.json").write_text(json.dumps(records))
    dist.destroy_process_group()


if __name__=="__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform=="darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker,args=(8,tmp),nprocs=8,join=True)
        ranks=[json.loads(p.read_bytes()) for p in sorted(Path(tmp).glob("r*.json"))]
    totals=[]
    for rows in zip(*ranks):
        rec=rows[0]
        totals.append(dict(rows=rec["rows"], width=rec["width"], collective=rec["kind"],
                           compared=sum(r["words"] for r in rows),different=sum(r["mismatches"] for r in rows),
                           max_abs=max(r["max_abs"] for r in rows)))
    print(json.dumps(dict(torch=str(torch.__version__),device="CPU/Gloo TP8",cases=totals,
                          scope="Native message-size accumulation sensitivity with identical per-rank partials. NCCL is untested."),indent=2))
