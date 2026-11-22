"""Count real per-expert K-tile requests, including the unpadded output-band wave tail."""
from pathlib import Path
import hashlib,json,math
here=Path(__file__).resolve().parent
profile=here.parents[2]/'experiments/iterations/20261104-moe-hardware-profile-recovered.json'
machine=json.loads((here/'assembly.json').read_bytes())['variants']
def geometry(rows,width,tiles):
    new=width in (16,24)
    out=dict(input_rows=rows,M=width,expert_row_tiles=tiles,weight_and_scale_request_MiB=27*tiles,
        native_MN_jobs_per_projection=32*tiles,selected_MN_jobs_per_projection=(16 if new else 32)*tiles,
        saved_activation_TMA_MiB_per_projection=(16 if new else 0)*tiles*width*4096*2/(1<<20))
    out['two_projection_saved_A_MiB']=2*out['saved_activation_TMA_MiB_per_projection']
    return out
recorded=[]
for case in json.loads(profile.read_bytes())['cases']:
    rows=case['M'];w={32:8,256:24}[rows]
    counts=case['routing']['local_counts'];t=sum((c+w-1)//w for c in counts)
    g=geometry(rows,w,t)
    g['reference_CTA_waves']=math.ceil(g['native_MN_jobs_per_projection']/132)
    g['selected_CTA_waves']=math.ceil(g['selected_MN_jobs_per_projection']/132)
    g['other_standalone_kernel_DRAM_MiB']=case['counter_launch']['metrics']['dram__bytes_read.sum']['value']/(1<<20)
    g['unique_packed_weight_and_scale_MiB']=27*sum(c>0 for c in counts)
    recorded.append(g)
uniform=[]
for u,w in ((1,8),(4,8),(16,16),(32,24)):
    rows=8*u;p=8/256;t=32*sum((c+w-1)//w*math.comb(rows,c)*p**c*(1-p)**(rows-c) for c in range(rows+1))
    uniform.append(geometry(rows,w,t)|dict(users=u))
normalized={}
for ph,kind in [('w13','wide'),('w2','wide3')]:
 for m in (16,24):
    b=machine[f'{ph}_{m}_native'];v=machine[f'{ph}_{m}_{kind}']
    baseline=2*b['full_K_loop_instruction_sites']/b['stages']
    new=v['full_K_loop_instruction_sites']/v['stages']
    normalized[f'{ph}_{m}']=dict(incumbent_instruction_sites_per_K_tile_for_two_output_bands=baseline,
        candidate_instruction_sites_per_equivalent_tile=new,pct_fewer=(1-new/baseline)*100)
print(json.dumps(dict(physical_source=str(profile.relative_to(here.parents[2])), sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
    recorded=recorded,independent_uniform=uniform,per_equal_output_work=normalized,
    denominator='A requests are shared-memory/TMA payload requested before cache; packed weight traffic is unchanged. Compiler site counts are not dynamic time.'),indent=2))
