"""Price V303 vs V298 from observed one-user checkpoint counts and an explicit link model."""
import json
import math
from pathlib import Path
from verify import HERE,digest
parent=HERE.parent/"v298_mapped_page_first_backup"
exposure=json.loads((parent/"exposure.json").read_bytes())
prior=HERE.parent/"v288_paired_cold_and_verify_intervals/check.json"
intervals=json.loads(prior.read_bytes())["matched_cold_summary"]
rungs=HERE.parents[2]/"experiments/iterations/20261118-serving-one-user-phase.json"
periods=json.loads(rungs.read_bytes())["archived_physical"]
form=[]
for swa in (0,2048,16384):
    runs=[16384,swa,16384];units=[4096,11776,2560]
    b=sum(n*u for n,u in zip(runs,units));native=sum(math.ceil(n/64) for n in runs)
    caps=[(8<<20)//u for u in units];splits=sum(math.ceil(n/c) for n,c in zip(runs,caps));mapped=sum(n>0 for n in runs)
    # Pack moves the K/V bytes once from the cache and once to temporary storage;
    # DMA reads the resulting bytes from GPU memory a further time.
    chunk=intervals["1"]["new_mean_s"]
    def bulk_time(link,hbm,call_us):
        return b/(link*1e9)+3*b/(hbm*1e9)+splits*3*call_us/1e6
    logical=dict(rows=runs,bytes=b,pool_row_capacities=caps,bulk_segments=splits,new_KV_D2H_calls=2*splits,
         v298_mapped_kernels=mapped,original_D2H_descriptors=2*sum(runs),original_pack_kernels=native)
    scenarios=[]
    for link,hbm,latency in ((16,2000,5),(32,3000,3),(64,4800,1)):
        bulk=bulk_time(link,hbm,latency)
        scenarios.append(dict(bulk_directional_GBps=link,gpu_memory_GBps=hbm,assumed_serial_us_per_bulk_command=latency,
            estimated_bulk_ms=bulk*1000,break_even_mapped_GBps=b/bulk/1e9,
            max_mapped_GBps_for_chunk_factors={str(k):b/(bulk+chunk*(1-1/k))/1e9 for k in (2,3,4)}))
    hist=[]
    bulk=bulk_time(32,3000,3)
    for run in exposure["waves"]:
        name=run["build"];solo=[w for w in run["waves"] if w["users"]==1]
        for w,q in zip(solo,periods[name]["per_rep"]):
            pts=w["solo_ongoing_checkpoints"];wall=q["last_request_end_s"]
            assert q["rep"]==w["rep"] and pts in (4,5,6)
            cold=q["first_turn_ttft_s"]
            sens=[]
            for bw in (1,4,8,16,32,64):
                # Fractional exposure translates modeled copy duration into end-to-end impact.
                d=pts*(b/(bw*1e9)-bulk)
                assert cold-d>0 and wall-d>0,(name,pts,d)
                sens.append(dict(mapped_GBps=bw,fully_exposed_saved_seconds=d,
                    new_cold_seconds_at_full_exposure=cold-d,full_run_gain_at_full_exposure=wall/(wall-d),
                    full_run_gain_at_quarter_exposure=wall/(wall-d*.25)))
            hist.append(dict(build=name,rep=w["rep"],cold_seconds=cold,full_run_seconds=wall,ongoing_checkpoints=pts,
                max_mapped_GBps_for_2x_full_run_if_only_these_backups_change=b/(bulk+wall/2/pts)/1e9,
                average_GBps_to_move_all_assumed_bytes_during_observed_cold=pts*b/(cold*1e9),sensitivity=sens))
    form.append(dict(**logical,scenario=scenarios,solo=hist))
print(json.dumps(dict(geometry=form,one_user_matched_V44_chunk_s=intervals["1"]["new_mean_s"],
    hashes={p.name:digest(p.read_bytes()) for p in (parent/"exposure.json",prior,rungs)},
    model="Long contiguous host runs, no registration boundary in a segment, one equal transfer at every historical solo unfinished checkpoint. V298's actual mapped bandwidth, destination order, written rows, GPU/host overlap and contention are unknown. Kernel/launch overlap, reuse and CPU scan costs vary. Illustrative rates, not end-to-end results."),indent=2))
