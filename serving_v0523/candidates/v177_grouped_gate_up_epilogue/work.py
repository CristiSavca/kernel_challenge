"""Tensor lifetimes and direct W13 activation traffic per rank in an eligible prefill window."""
import json
dh,di,bytes_per_value=4096,4096,2
samples=[]
for routes in (4096,8192,12288,16384,18432):
    parent=bytes_per_value*routes*(max(dh,di)+max(2*di,dh))
    fused=bytes_per_value*routes*(di+dh)
    traffic=2*bytes_per_value*routes*2*di
    samples.append(dict(routes=routes,original_workspace=parent,paired_workspace=fused,
        eliminated_write_plus_read=traffic,nominal_transfer_ms_at_4_8_TBps=traffic/4.8e9,
        saved_traffic_62_layers_GiB=62*traffic/2**30))
print(json.dumps(dict(H=dh,I=di,scored_layers=62,byte_bound="logical global operations, before cache",
    per_nonempty_window_calls=dict(one_cohort=[3,2],two_cohorts=[5,4]),samples=samples),indent=2))
