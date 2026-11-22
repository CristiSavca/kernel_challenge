"""Inspect the actual parent and cooperative output-band code in all 18 SM90a cubins."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

here=Path(__file__).resolve().parent
root=Path(sys.argv[1])
shared=runpy.run_path(str(here.parent/'v199_prompt_expert_overlap/independent_ring.py'))['resident_bytes']
configs=json.loads((root/'configs.json').read_bytes())
result={}
for name,config in configs.items():
    p=root/'images'/name
    source,log,sass=[p.with_suffix(ext).read_text() for ext in ('.cu','.log','.sass')]
    assert hashlib.sha256(source.encode()).hexdigest()==config['source_sha256']
    ops={int(m[1],16):m[2] for l in sass.splitlines() if (m:=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\* 0x',l))}
    m,n,k=config['block'];stages=config['stages']
    bands=config['warp'][1]//16
    sites=4*stages*bands
    assert sass.count(f'HGMMA.64x{m}x16.F32.BF16')==sites
    loops=[]
    for stop,inst in ops.items():
        branch=re.search(r'\bBRA(?:\.U)? (0x[0-9a-f]+)',inst)
        if branch and (start:=int(branch[1],16))<stop:
            body=[op for pc,op in ops.items() if start<=pc<=stop]
            if sum('HGMMA.' in op for op in body)==sites:loops.append(body)
    loop=min(loops,key=len)
    held=0;groups=[]
    for inst in loop:
        held+='HGMMA.' in inst
        if 'WARPGROUP.DEPBAR.LE' in inst:
            groups.append(held);held=0
    assert groups==[bands]*(stages*4) and held==0
    assert shared(p.with_suffix('.cubin'))==config['shared']
    result[name]=dict(**config,cubin_sha256=hashlib.sha256(p.with_suffix('.cubin').read_bytes()).hexdigest(),
        sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),registers=int(re.search(r'Used (\d+) registers',log)[1]),
        stack_store_load=list(map(int,re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log).groups())),
        full_K_loop_instruction_sites=len(loop),stages_per_loop=stages,HGMMA_per_loop=sites,drains_per_loop=len(groups),
        loop_local_access_sites=sum('LDL' in op or 'STL' in op for op in loop),whole_instruction_sites=len(ops))
for phase,kind in [('w13','wide'),('w2','wide3')]:
    for m in (16,24):
        chosen=result[f'{phase}_{m}_{kind}']
        assert chosen['registers']==96 and chosen['stack_store_load']==[0,0,0] and not chosen['loop_local_access_sites']
print(json.dumps(dict(compiler='NVCC 12.9.86, compute_90a/sm_90a, pinned Humming 0.1.12', variants=result),indent=2))
