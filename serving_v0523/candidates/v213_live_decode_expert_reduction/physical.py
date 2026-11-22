"""Apply both Humming work schedules to the counted routes of two real, separate H200 kernels."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
src=here.parents[2]/'experiments/iterations/20261104-moe-hardware-profile-recovered.json'
archive=json.loads(src.read_bytes());out=[]
for case in archive['cases']:
    m=case['M'];shape={32:8,256:24}[m]
    counts=case['routing']['local_counts']
    valid=[min(shape,i-k) for i in counts for k in range(0,i,shape)]
    old=shape*len(valid);new=sum((i+7)//8*8 for i in valid);vectors=sum((i+1)//2*2 for i in valid)
    metrics={k:v['value'] for k,v in case['counter_launch']['metrics'].items()}
    cold=case['conditional_same_traffic_4_8TB_s_calculation']
    total=cold['profile_duration_us'];floor=cold['ideal_duration_us']
    out.append(dict(rows=m,local_routes=sum(counts),active_experts=sum(i>0 for i in counts),V211_width=shape,
        native_tiles=len(valid),MMA_rows=[old,new],row_arithmetic_reduction_percent=100*(1-new/old),
        V211_V213_partial_shared_mib=[v*2*(8192+3*4096)*4/2**20 for v in (old,vectors)],
        nominal_Humming_requested_weights_MiB=len(valid)*27,measured_different_kernel=case['counter_launch']['kernel'],
        measured_dram_read_bytes=int(metrics['dram__bytes_read.sum']),measured_dram_write_bytes=int(metrics['dram__bytes_write.sum']),
        measured_time_us=total,measured_percent_dram_peak=metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed'],
        measured_tensor_active_percent=metrics['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'],
        measured_barrier_stall_percent=metrics['smsp__warp_issue_stalled_barrier_per_warp_active.pct'],
        ideal_same_measured_traffic_us=floor,conditional_same_traffic_ratio=total/floor))
print(json.dumps(dict(source=str(src.relative_to(here.parents[2])),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    NCU_policy=archive['policy'],scope='Observed archived rank-local histogram; source-accounted Humming work. The timed profile belongs to another implementation with its own cache-flushed traffic.',cases=out),indent=2))
