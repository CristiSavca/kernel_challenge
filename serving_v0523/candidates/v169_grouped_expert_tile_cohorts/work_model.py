"""Evaluate the deployed tile planner on uniform independent top-k and skewed routes."""
from pathlib import Path
import importlib.util, json, sys
import numpy as np
root = Path(sys.argv[1])
src=root/"python/sglang/srt/layers/muse_spark/prefill_expert_tiles.py"
sp=importlib.util.spec_from_file_location("plan169",src);p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
state = ({w:{} for w in (136,152,176)},(32,16),132)

def evaluate(tokens, skew, trials=48):
    rng=np.random.default_rng(16900+tokens+int(skew))
    records=[]
    for seed in range(trials):
        ids=np.empty((tokens,8),np.int32)
        for j in range(8):
            draw=rng.integers(0,256,tokens)
            if skew:
                hot=rng.uniform(size=tokens)<0.25
                draw[hot]=rng.integers(0,32,int(hot.sum()))
            duplicate=(ids[:,:j]==draw[:,None]).any(1)
            while duplicate.any():
                n=duplicate.sum()
                repeat=rng.integers(0,256,n)
                if skew:
                    hot=rng.uniform(size=n)<0.25
                    repeat[hot]=rng.integers(0,32,int(hot.sum()))
                draw[duplicate]=repeat
                duplicate=(ids[:,:j]==draw[:,None]).any(1)
            ids[:,j]=draw
        for counts in np.bincount(ids.flatten(),minlength=256).reshape(8,32):
            hist=counts.tolist(); cap=tokens+max(1024,(tokens+7)//8)
            if sum(hist)>cap:
                records.append(None);continue
            plan=p.choose_tiles(state,hist)
            base=176*((counts+175)//176).sum()
            padded=base
            waves=lambda tiles:sum((tiles*n+131)//132 for n in (32,16))
            oldcost=176*waves(base//176);newcost=oldcost;groups=1
            if plan:
                specs,table=plan;groups=len(specs);padded=newcost=read_tiles=0
                for (w,begin,end),off in zip(specs,table[-groups:]):
                    subgroup=np.diff(off);tiles=((subgroup+w-1)//w).sum()
                    read_tiles+=tiles;padded+=tiles*w;newcost+=waves(tiles)*w
                assert read_tiles==base//176
            records.append((sum(hist),base,padded,oldcost,newcost,groups,int(plan is not None)))
    arr=np.array([a for a in records if a is not None]); live,old,new,ow,nw,groups,selected=arr.T
    return dict(tokens=tokens,routing="25% per-draw bias to rank 0" if skew else "uniform",rank_samples=len(records),
                single_window_samples=len(arr),multiwindow_fallbacks=len(records)-len(arr),selected=float(selected.mean()),
                two_cohort_fraction=float((groups==2).mean()),useful_fraction_parent=float(live.sum()/old.sum()),
                modeled_padded_arithmetic_ratio=float(old.sum()/new.sum()),modeled_waves_weighted_by_width_ratio=float(ow.sum()/nw.sum()),
                no_extra_packed_bank_sweeps=True,mean_packed_group_rows_before=float(old.mean()),mean_packed_group_rows_after=float(new.mean()),
                dense_bf16_peak_compute_floor_ms_before=float(old.mean()*2*3*4096**2/989.5e9),
                dense_bf16_peak_compute_floor_ms_after=float(new.mean()*2*3*4096**2/989.5e9))
print(json.dumps(dict(model="Issued row and reserved-wave model; no timestamps. Multiwindow ranks retain parent.",
    trials_per_shape=48,sm_count=132,bytes_per_bank_sweep_per_expert=27*(1<<20),physical_H200=False,
    cases=[evaluate(t,skew) for t in (8192,10240,12288,14336,16384) for skew in (False,True)]),indent=2))
