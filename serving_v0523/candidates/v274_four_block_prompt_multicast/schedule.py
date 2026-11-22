"""Replay pinned grouped CTA enumeration and compare individual work streams and full TMA boxes."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
header=Path(sys.argv[1])/"include/humming/scheduler.cuh"
source=header.read_text()
for token in ("N_BLOCKS = ProblemShape::N / BlockShape::N / kMultiCastSizeA",
              "if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm)", "m_id = mn_index / N_BLOCKS",
              "n_id = mn_index % N_BLOCKS", "dp_mn_next_index = blockIdx.x / kMultiCastSize",
              "dp_mn_next_index += gridDim.x / kMultiCastSize", "n_block_id = n_block_id * kMultiCastSizeA + cluster_rank"):
    assert token in source

def issued(count):
    return next(n for n in (32,128,224,256) if count<=n)

def schedule(hist,width,grid,n=4096):
    nblocks=n//128
    assert n%128==0 and nblocks%width==0 and grid%width==0
    mtiles=[(e,off,issued(min(count-off,256))) for e,count in enumerate(hist) for off in range(0,count,256)]
    per_group=nblocks//width
    chars=[]
    for cta in range(grid):
        rank=cta%width
        jobs=[]
        for index in range(cta//width,len(mtiles)*per_group,grid//width):
            m,ng=divmod(index,per_group)
            jobs.append((mtiles[m],ng*width+rank))
        chars.append(jobs)
    unique=Counter(job for jobs in chars for job in jobs)
    assert len(unique)==len(mtiles)*nblocks and all(v==1 for v in unique.values())
    return chars

cases=json.loads((here/"routes.json").read_bytes())["cases"]
out=[]
checks=0
for case in cases:
    rows=[]
    for hist in case["histograms"]:
        for grid in (4,8,64,120,124,128,132):
            pair=schedule(hist,2,grid)
            assert pair==schedule(hist,4,grid)
            if grid%8==0:
                assert pair==schedule(hist,8,grid)
            checks+=1
        raw_tiles=sum((n+255)//256 for n in hist)
        tensor_rows=sum(issued(min(256,n-off)) for n in hist for off in range(0,n,256))
        rows.append(dict(local_routes=sum(hist),m_tiles=raw_tiles,issued_tensor_rows=tensor_rows,
                A_full_box_MiB={size:raw_tiles*32*2//size for size in (2,4,8)},
                packed_W2_plus_scales_MiB=raw_tiles*9,
                W2_C_written_MiB=sum(hist)*4096*2/2**20,
                equal_grid_128_max_CTA_jobs=max(map(len,schedule(hist,4,128)))))
    out.append(dict(tokens=case["tokens"],ranks=rows))
for hist in ([0]*32,[1]+[0]*31,[0,513,0,32,0,225,0,63,1025]+[0]*23,[16384]+[0]*31,
             [31,32,33,127,128,129,223,224,225,255,256,257]*2+[0]*8):
    for n in (512,1024,3584,4096):
        for grid in (4,8,64,120,124,128,132):
            assert schedule(hist,2,grid,n)==schedule(hist,4,grid,n)
            checks+=1
print(json.dumps(dict(vendor_scheduler_sha256=sha256(header.read_bytes()).hexdigest(),
    matched_grid_histogram_geometry_cases=checks,primitive="Full K 4096, N 4096, 256x128 per CTA. Packed B/scale reread per M tile. Full source activation boxes before cache or TMA boundary effects.",
    results=out,GPU=False),indent=2))
