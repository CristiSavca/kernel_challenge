"""Test the serving-sized all-reduce and both O stripes on identical BF16 partials."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,size,tmp):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(tmp)/"sweep"),rank=rank,world_size=size,
                            timeout=timedelta(minutes=5))
    width=8192
    g=torch.Generator().manual_seed(267012+rank)
    # Dense, rank-specific BF16 values are sensitive to the order of additions.
    # Tiling bounds transient memory; every real matrix element participates in Gloo.
    tile=torch.randn((256,width),generator=g,dtype=torch.bfloat16)
    report=[]
    for rows in (8192,8193,8200,8256,16383,16384):
        x=tile.repeat(((rows+255)//256,1))[:rows]
        owner=(rows+size-1)//size
        start=rank*owner
        length=min(owner,rows-start)
        full=x.clone()
        dist.all_reduce(full)
        reference=full.narrow(0,start,length).clone()
        del full
        count=0
        max_abs=0.
        for part,ref in zip(x.chunk(2,1),reference.chunk(2,1)):
            part=part.contiguous()
            dist.all_reduce(part)
            got=part.narrow(0,start,length)
            count+=int((got.view(torch.int16)!=ref.view(torch.int16)).sum())
            max_abs=max(max_abs,float((got.float()-ref.float()).abs().max()))
        report.append(dict(tp=size,rank=rank,rows=rows,width=width,compared=length*width,different=count,max_abs=max_abs))
        del x, reference, ref, part, got
    (Path(tmp)/f"r{rank}.json").write_text(json.dumps(report))
    dist.destroy_process_group()


if __name__=="__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0" if sys.platform=="darwin" else "lo")
    out=[]
    for size in (4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker,args=(size,tmp),nprocs=size,join=True)
            records=[json.loads(p.read_bytes()) for p in sorted(Path(tmp).glob("r*.json"))]
        for case in zip(*records):
            out.append(dict(tp=size,rows=case[0]["rows"],width=case[0]["width"],
                    compared=sum(r["compared"] for r in case),different=sum(r["different"] for r in case),
                    max_abs=max(r["max_abs"] for r in case)))
    print(json.dumps(dict(device="CPU Gloo",torch=str(torch.__version__),cases=out,
                          scope="Full native message vs both stripe messages, actual 8192 channel width, otherwise identical synthetic BF16 input, one resulting owner per row. No GPU."),indent=2))
