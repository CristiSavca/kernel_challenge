"""Compare exact Humming SM90a cubins, typed K loops, and scheduler landmarks."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

here = Path(__file__).resolve().parent
root = Path(sys.argv[1])
shared = runpy.run_path(str(here.parent/"v199_prompt_expert_overlap/independent_ring.py"))["resident_bytes"]


def measure(directory, name, width):
    path = root / directory / name
    text = path.with_suffix(".sass").read_text()
    log = path.with_suffix(".log").read_text()
    ops = {int(m[1],16):m[2] for line in text.splitlines()
           if (m:=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\* 0x', line))}
    loops = {}
    for w in range(8,width+1,8):
        candidates = []
        for stop,instr in ops.items():
            b = re.search(r'\bBRA(?:\.U)? (0x[0-9a-f]+)', instr)
            if b and (start:=int(b[1],16)) < stop:
                block = [v for k,v in ops.items() if start <= k <= stop]
                if sum('HGMMA.' in i for i in block)==sum(f'HGMMA.64x{w}x16.F32.BF16' in i for i in block)==32:
                    candidates.append(block)
        body = min(candidates,key=len)
        arrivals = []
        matrices = 0
        for i in body:
            matrices += 'HGMMA.' in i
            if 'SYNCS.ARRIVE' in i:
                arrivals.append(matrices)
        assert arrivals[:4]==[8,16,24,32] and sum('WARPGROUP.DEPBAR.LE' in i for i in body)==8
        assert not any('LDL' in i or 'STL' in i for i in body)
        loops[w] = len(body)
    hashes = {end:hashlib.sha256(path.with_suffix('.'+end).read_bytes()).hexdigest() for end in ('cu','cubin','sass')}
    return dict(sha256=hashes, dynamic_shared=shared(path.with_suffix('.cubin')),
                regs=int(re.search(r'Used (\d+) registers',log)[1]),
                stack_store_load=list(map(int,re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log).groups())),
                kernel_bytes=path.with_suffix('.cubin').stat().st_size, instruction_sites=len(ops), K_loop_sites=loops,
                full_CTA_barrier_sites=sum('BAR.SYNC' in i for i in ops.values()),
                scheduler_votes=sum('VOTE.ANY' in i for i in ops.values()), scheduler_msb=sum('FLO.U32' in i for i in ops.values()))


cases = {}
for phase in ('w13','w2'):
    for width in (8,16,24):
        name = f'{phase}_{width}'
        before,after = [measure(folder,name,width) for folder in ('parent','live')]
        assert before['regs']==after['regs']==96
        assert before['stack_store_load']==after['stack_store_load']==[0,0,0]
        assert before['dynamic_shared']==after['dynamic_shared']
        assert before['full_CTA_barrier_sites']-after['full_CTA_barrier_sites']==2
        assert after['scheduler_msb']==2 and after['scheduler_votes']>=2
        cases[name] = {'V263':before, 'V269':after}
print(json.dumps(dict(compiler='NVCC 12.9.86 SM90a offline', arithmetic='all selected K loops retain 32 BF16 HGMMA, eight drains, four ordered shared-stage releases and no local loads or stores',cases=cases, physical_H200=False),indent=2))
