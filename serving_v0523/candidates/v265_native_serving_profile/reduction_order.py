"""Isolate full versus striped CPU Gloo SUM with aligned rows and identical per-rank inputs."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def reduce(n, width, value, rank, size, mode):
    # Exactly one largest tensor per participant. Recreate the same scalar
    # payload for the narrow test after releasing the full-size operand.
    data=torch.full((n,width),value,dtype=torch.bfloat16)
    local=n//size
    if mode=="ar":
        dist.all_reduce(data)
        return data[rank*local:(rank+1)*local].clone()
    owner=torch.empty((local,width),dtype=data.dtype)
    dist.reduce_scatter_tensor(owner,data)
    return owner


def worker(rank, size, dir, widths):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(dir)/"rdv"),rank=rank,world_size=size,
                            timeout=timedelta(minutes=4))
    # These integer inputs are all exactly representable in BF16 and FP32.
    value=[2**24,1,-2**24,0,0,0,0,0][rank]
    data=[]
    for w in widths:
        for mode in ("ar","rs"):
            full=reduce(16384,w,value,rank,size,mode)
            # Both stripes have identical shapes and operands, but execute both.
            count=0
            examples=[]
            for half in range(2):
                slim=reduce(16384,w//2,value,rank,size,mode)
                expected=full[:,half*w//2:(half+1)*w//2]
                # Bound temporary comparison storage even for the full 8192 width.
                for r in range(0,len(slim),64):
                    diff=slim[r:r+64]!=expected[r:r+64]
                    delta=int(diff.sum())
                    count+=delta
                    if delta and len(examples)<2:
                        i,j=diff.nonzero()[0].tolist()
                        examples.append(dict(token=rank*(16384//size)+r+i,col=half*w//2+j,
                                             full=float(expected[r+i,j]),stripe=float(slim[r+i,j])))
                del slim
            data.append(dict(mode=mode,rows=16384,channels=w,owned=full.numel(),differences=count,examples=examples))
            del full
    Path(dir,f'{rank}.json').write_text(json.dumps(data))
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--widths",type=int,nargs="+",default=[256,8192])
    args=parser.parse_args()
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0" if __import__('sys').platform=="darwin" else "lo")
    with tempfile.TemporaryDirectory(prefix="okc-v265-gloo-order-") as folder:
        size=8
        mp.spawn(worker,args=(size,folder,args.widths),nprocs=size,join=True)
        ranks=[json.loads(Path(folder,f"{rank}.json").read_bytes()) for rank in range(size)]
    summaries=[]
    for cases in zip(*ranks):
        head=cases[0]
        summaries.append(dict(mode=head["mode"],rows=head["rows"],channels=head["channels"],
                              owned_compared=sum(c["owned"] for c in cases),differing=sum(c["differences"] for c in cases),
                              by_rank=[c["differences"] for c in cases],examples=[e for c in cases for e in c["examples"]][:4]))
    return dict(torch=torch.__version__,backend="CPU Gloo",tp=8,reductions=summaries,
                rank_BF16_constants=[2**24,1,-2**24,0,0,0,0,0], masks_or_roundoff_in_inputs=False,
                scope="Same exactly representable per-rank constant for full and half tensors. All token rows aligned. Comparison includes only the calling rank's owned rows. No H200 or NCCL timing or accuracy inferred.")


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
