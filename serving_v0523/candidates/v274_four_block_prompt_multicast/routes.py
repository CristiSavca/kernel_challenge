"""Reconstruct the standalone H200 synthetic-route draw and check the physically recorded local totals."""
import hashlib
import json
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
root=here.parents[2]
result=root/"experiments/modal_serving/20261113T065259Z/result.json"


def philox(seed, iteration, threads):
    a=np.full(threads,iteration,dtype=np.uint64)
    b=np.zeros_like(a)
    c=np.arange(threads,dtype=np.uint64)
    d=np.zeros_like(a)
    key0, key1 = seed, 0
    mask=0xffffffff
    for _ in range(10):
        x=a*0xd2511f53
        y=c*0xcd9e8d57
        a,b,c,d=(y>>32)^b^key0,y&mask,(x>>32)^d^key1,x&mask
        key0=(key0+0x9e3779b9)&mask
        key1=(key1+0xbb67ae85)&mask
    return np.concatenate([a,b,c,d]).astype(np.uint32)


def counts(tokens):
    size=tokens*256
    threads=min(size,132*2048)
    vals=np.concatenate([philox(77+tokens,i,threads) for i in range((size+4*threads-1)//(4*threads))])[:size]
    # curand_uniform4 maps to (0,1]. PyTorch maps the upper endpoint back to 0.
    vals=((vals.astype(np.float64)+0.5)/2**32).astype(np.float32)
    vals[vals==1]=0
    vals=vals.reshape(tokens,256)
    selected=np.argpartition(vals,8,axis=1)[:,:9]
    order=np.take_along_axis(vals,selected,axis=1).argsort(axis=1)
    ordered=np.take_along_axis(selected,order,axis=1)
    sorted_vals=np.take_along_axis(vals,ordered,axis=1)
    assert (sorted_vals[:,7]<sorted_vals[:,8]).all(), "ambiguous top-eight set"
    return np.bincount(ordered[:,:8].ravel(),minlength=256).reshape(8,32)


draws=[]
for case in json.loads(result.read_bytes())["cases"]:
    h=counts(case["T"])
    assert int(h[3].sum())==case["local_routes"],(case["T"],int(h[3].sum()),case["local_routes"])
    draws.append(dict(tokens=case["T"],measured_rank=3,measured_local_routes=case["local_routes"],
                      histograms=h.tolist(),marlin_us=case["stock_us"],fully_prepared_dense_us=case["dense_ceiling"]["us"]))
print(json.dumps(dict(archived_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),cases=draws,
    scope="Philox4x32-10/pinned worker synthetic torch.rand top-eight routing. Rank-3 aggregate checked against the archived H200; per-expert bins reconstructed on CPU."),indent=2))

