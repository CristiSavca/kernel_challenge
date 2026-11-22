"""Quantify the changed machine work, native graph coverage and historical timing envelope."""
import json
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
m=read(here/'machine.json')
phase=here.parent/'v262_decode_phase_budget/check.json'
case=read(phase)['cases'][0]
base=here.parent/'v297_wide_fused_decode/work.json'
rungs=read(base)['per_load_V41_scheduler_reports']
data={}
for name,k in [('w13_m8_n256',128),('w2_m8_n128',256)]:
    comp=m['m8'][name]
    info={}
    for tag in ('parent','selected'):
        x=comp[tag]
        info[tag]=dict(K4096_drains_per_WG=4096//k*sum(x['drain_sizes'].values())//4,
                       static_loop_instructions=x['loop_sites'],static_while_matrix_pending=x['checks']['ordinary_instructions_in_flight'])
    data[name]=info
period=case['estimated_period_ms']
print(json.dumps(dict(history_scheduler_reports=[dict(users=int(k),M8=v['M8'],reports=sum(v.values())) for k,v in rungs.items()],
    kernels=data, historical_V64_one_user_period_ms=period, historical_cold_bank_read_at_peak_ms=case['expected_cold_expert_weights_peak_ms_62_layers'],
    sensitivity_at_one_user={str(pct):{'saving_ms':period*(1-1/(1+pct/100)),
        'saving_us_per_target_layer':1000*period*(1-1/(1+pct/100))/62} for pct in (5,10,100)},
    sources={str(f.relative_to(here.parents[2])):hashlib.sha256(f.read_bytes()).hexdigest() for f in (base,phase)},
    scope='Status-line M bins from an earlier eight-H200 run. V64 period uses aggregated committed tokens and median decode rate. No V299 H200 timing or measured stall counter.'),
    indent=2))
