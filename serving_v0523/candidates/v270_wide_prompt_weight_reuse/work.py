"""Binomial grouped-route model and persistent-cluster tail sensitivity, independent of production tuning."""
import json
import math
from pathlib import Path

here=Path(__file__).resolve().parent
isa=json.loads((here/"sm90a.json").read_text())["variants"]


def analyze(tokens):
    p=8/256
    mode=int((tokens+1)*p)
    weights=[0.]*(tokens+1)
    weights[mode]=1.
    for count in range(mode,tokens):
        weights[count+1]=weights[count]*(tokens-count)*p/((count+1)*(1-p))
    for count in range(mode,0,-1):
        weights[count-1]=weights[count]*count*(1-p)/((tokens-count+1)*p)
    total=math.fsum(weights)
    rows={}
    for role,widths,channels,multicast in (("ResidualPromptWeightPipeline",(32,96,176),256,1),
                                            ("WidePromptWeightPipeline",(32,128,224,256),128,2)):
        bins={width:0. for width in widths}
        for count,weight in enumerate(weights):
            if weight:
                full,tail=divmod(count,widths[-1])
                bins[widths[-1]]+=weight*full
                if tail: bins[next(w for w in widths if tail<=w)]+=weight
        bins={k:n/total*32 for k,n in bins.items()}
        mtiles=math.fsum(bins.values())
        issued=math.fsum(k*n for k,n in bins.items())
        eight=32*math.fsum(((n+7)//8*8)*mass/total for n,mass in enumerate(weights) if mass)
        ntiles=4096//channels
        sites=math.fsum(isa[role]["loops"][str(k)]["sites"]*n for k,n in bins.items())*ntiles
        rows[role]=dict(expected_rank_routes=tokens,mean_m_tiles=mtiles,mean_ctas=mtiles*ntiles,
            mean_clusters=mtiles*ntiles/multicast,issued_tensor_rows=issued,finest8_tensor_rows=eight,
            excess_rows_vs_finest8=issued-eight,extra_TFLOP_vs_finest8=(issued-eight)*2*4096**2/1e12,
            per_width_m_tiles=bins,W2_packed_request_MiB=9*mtiles,
            A_multicast_full_box_MiB=mtiles*widths[-1]*4096*2*ntiles/multicast/2**20,
            four_K64_stage_consumer_warp_site_proxy=sites*8)
    prior,wide=rows.values()
    # Model equal-cost 256-K work items, static cyclic placement. In the delayed
    # case cluster slots are completed in waves; the earliest remaining slot is reused.
    jobs=round(wide["mean_clusters"])
    def grouped_time(grid,resident):
        ready=[0]*resident
        for c in range(grid):
            idx=min(range(resident),key=ready.__getitem__)
            ready[idx]+=len(range(c,jobs,grid))
        return max(ready)
    sampling={resident:{"fixed_grid_66":grouped_time(66,resident),"sized_grid":grouped_time(resident,resident)}
              for resident in (48,60,62,64,65,66)}
    return dict(tokens=tokens,rows=rows,ideal_equal_job_waves=sampling,
                assumed_equal_cost_logical_cluster_jobs=jobs,
                packed_weight_request_reduction=1-wide["W2_packed_request_MiB"]/prior["W2_packed_request_MiB"],
                consumer_warp_site_proxy_reduction=1-wide["four_K64_stage_consumer_warp_site_proxy"]/prior["four_K64_stage_consumer_warp_site_proxy"],
                issued_tensor_row_change=wide["issued_tensor_rows"]/prior["issued_tensor_rows"]-1)


print(json.dumps(dict(assumption="Uniform independent distinct top-8 among 256 experts, 32 resident per rank. W2 only. No timing.",
                      samples=[analyze(n) for n in (8192,10240,12288,14336,16384)]), indent=2))

