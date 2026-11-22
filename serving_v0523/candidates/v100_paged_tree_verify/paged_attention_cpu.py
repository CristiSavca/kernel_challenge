"""Run production partition/merge on fragmented physical KV pages against a dense tree oracle."""
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np


def main(source):
    v70 = runpy.run_path(str(Path(__file__).resolve().parents[1]/"v70_dflash_selector_tree/cpu_check.py"))
    pid, (_, _, partition, merge) = v70["program"](
        source / "python/sglang/kernels/ops/attention/paged_verify_attention.py",
        ("_tile_geometry", "_plan", "_partition_attention", "_merge_partitions"),
    )
    pointer = v70["Pointer"]
    rng = np.random.default_rng(24112026)
    tests, checked_rows, cross_pages, poisoned, worst = 0,0,0,0,0.
    grouped = {}
    for page in (1,3,16,127,128,160,192,256):
        count=0
        for width in (4,8):
            for window in (-1,5,64,2047):
                for prefix in (0,page-1,page,page+1,2*page-3,2048+page-3):
                    prefix=max(0,prefix)
                    prefixes=np.array([prefix,prefix+1],np.int32)
                    b=len(prefixes)
                    lengths=prefixes+width
                    total_pages=(int(lengths.max())+page-1)//page
                    # A prefix visible to this request never shares a physical page
                    # with another fixture request. Page zero is a reserved sink.
                    pages=(1+rng.permutation(b*total_pages)).reshape(b,total_pages).astype(np.int32)
                    virtual=pages.copy()
                    if window>=0:
                        for row in range(b):
                            cutoff=max(int(prefixes[row])-window,0)
                            n=cutoff//page
                            virtual[row,:n]=-1
                            poisoned+=n
                    rng_size=(1+b*total_pages)*page
                    d,g=64,8
                    # All input values lie exactly on the E4M3 grid. The CPU model
                    # rounds P to BF16 and performs matrix multiplication in FP32.
                    k=rng.integers(-8,9,(rng_size,d)).astype(np.float32)/8
                    v=rng.integers(-8,9,(rng_size,d)).astype(np.float32)/8
                    q=rng.integers(-8,9,(b*width,g,d)).astype(np.float32)/8
                    kscale=np.array([.75,1.25],np.float32);vscale=np.array([1.5,.5],np.float32)
                    ids=np.arange((width-1)*4).reshape(width-1,4)
                    infos=[v70["proposal_oracle"](ids,rng.normal(size=(width-1,4,4)),999) for _ in range(b)]
                    infos[1][1]=np.arange(width);infos[1][2]=np.arange(width)-1;infos[1][3]=(1<<np.arange(1,width+1))-1
                    depths=np.ascontiguousarray([x[1] for x in infos],dtype=np.int32)
                    bits=np.ascontiguousarray([x[3] for x in infos],dtype=np.int32)
                    cuq=np.arange(0,(b+1)*width,width,dtype=np.int32)
                    splits=7;tm=width*g;tn=128 if window<0 else 64
                    partial=np.full((splits+b-1,1,tm,d),np.nan,np.float32);lse=np.full(partial.shape[:-1],np.nan,np.float32)
                    out=np.full(q.shape,np.nan,np.float32)
                    for part in range(splits):
                        pid[:]=[part,0,0]
                        partition(*[pointer(x) for x in (q,k,v,virtual,lengths,cuq,kscale,depths,bits,partial,lse)],
                            total_pages,page,page*d,d,d,page*d,d,d,1,0,b,1,g,d,.125*math.log2(math.e),window,width,tm,tn,splits,b)
                    for row in range(b):
                        for packed in range(tm):
                            pid[:]=[row,0,packed]
                            merge(*[pointer(x) for x in (partial,lse,lengths,cuq,vscale,out)],
                                1,0,b,1,g,d,window,tm,tn,splits,b,8)
                    ref=np.empty_like(out)
                    for row,start in enumerate(prefixes):
                        for node in range(width):
                            left=max(0,start+depths[row,node]-window) if window>=0 else 0
                            vis=list(range(min(left,start),start))
                            vis.extend(start+j for j in range(width) if int(bits[row,node]) & 1<<j and start+depths[row,j]>=left)
                            vis=np.array(vis)
                            phys=pages[row,vis//page]*page+vis%page
                            idx=row*width+node
                            logits=q[idx]@k[phys].T*(.125*kscale[row])
                            pr=np.exp(logits-logits.max(axis=1,keepdims=True))
                            ref[idx]=(pr@v[phys])/pr.sum(axis=1,keepdims=True)*vscale[row]
                    err=float(np.linalg.norm(out-ref)/np.linalg.norm(ref))
                    assert np.isfinite(err) and err<.004,(page,prefix,window,width,err)
                    worst=max(worst,err);tests+=1;count+=1;checked_rows+=b*width
                    cross_pages+=sum(int(start)//page != (int(start)+width-1)//page for start in prefixes)
        grouped[page]=count
    return dict(cases=tests,per_page_size=grouped,verified_query_rows=checked_rows,cross_page_suffixes=int(cross_pages),
        poisoned_expired_swa_table_entries=poisoned,relative_l2_max=worst,
        coverage="Native full/SWA tile sizes, fragmented physical pages, opposite request scales, mixed chain/tree, dense logical-position oracle.",
        device="CPU source emulator; no GPU measurement")


if __name__=="__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
