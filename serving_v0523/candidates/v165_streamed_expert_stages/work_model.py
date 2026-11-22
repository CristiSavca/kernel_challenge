"""Model rank tails from fresh uniform distinct top-k routes. No serving inputs or clocks."""
import json
import numpy as np

def model(tokens,trials):
    rng=np.random.default_rng(165_000+tokens)
    traces=[]
    for first in range(0,trials,128):
        n=min(128,trials-first)
        scores=rng.random((n,tokens,256),dtype=np.float32)
        ids=np.argpartition(scores,8,axis=2)[:,:,:8]
        hist=np.zeros((n,256),np.int32)
        np.add.at(hist,(np.arange(n)[:,None,None],ids),1)
        jobs=((hist+7)//8).reshape(n,8,32).sum(axis=2)
        traces.append(jobs)
    jobs=np.concatenate(traces)
    bound=min(tokens*8,32)+max(0,tokens*8-32)//8
    old=min(32,tokens)
    capacity=132*4//32
    groups=min(bound,capacity) if old<capacity else old
    max_jobs=jobs.max(axis=1)
    before=(max_jobs+old-1)//old; after=(max_jobs+groups-1)//groups
    return dict(tokens=tokens,uniform_trials=trials,expected_rank_jobs=float(jobs.mean()),
                expected_busiest_rank_jobs=float(max_jobs.mean()),
                v161_group_cap=old,modelled_v165_group_cap=groups,
                probability_any_rank_grid_stride_before=float(np.mean(before>1)),
                probability_any_rank_grid_stride_after=float(np.mean(after>1)),
                average_longest_cta_full_k_sweeps_before=float(before.mean()),
                average_longest_cta_full_k_sweeps_after=float(after.mean()),
                average_launched_ctas_before=int(32*old),average_launched_ctas_after=int(32*groups),
                average_live_cta_jobs_per_rank=float(jobs.mean()*32),
                average_jobs_above_cap_on_busiest_rank_before=float(np.maximum(max_jobs-old,0).mean()),
                average_jobs_above_cap_on_busiest_rank_after=float(np.maximum(max_jobs-groups,0).mean()))
if __name__=="__main__":
    print(json.dumps(dict(scenarios=[model(8,32768),model(32,4096)],rank_count=8,local_experts=32,hidden=4096,
          sm_count=132,conditional_active_blocks_per_sm=4,source="shape model, H200-limited occupancy estimate",
          measured_gpu=False),indent=2))
