"""Summarize actual selected SM90a inner loops and predicate acceptance at each width."""
import hashlib
import json
from pathlib import Path
import re
import sys

stage, build=map(Path,sys.argv[1:])
src=stage/'python/sglang/srt/layers/muse_spark/live_decode_reduce.py'
env={};exec(compile(src.read_text(),str(src),'exec'),env)
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result={}
for image in sorted(build.glob('*.sass')):
    match=re.fullmatch(r'(w13|w2)_m(\d+)_n(\d+)_(.+)',image.stem)
    proj,m,n,variant=match.groups();m,n=int(m),int(n)
    wide=m>8; stages=3 if wide and proj=='w2' else 4
    bands=4 if wide else 2
    sass=image.read_text()
    ops={int(a,16):b.strip() for a,b in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?)\s*/\* 0x',sass)}
    loops={}
    for end,op in ops.items():
        dest=re.search(r'\bBRA(?:\.U)? (0x[\da-f]+)',op)
        if not dest or (start:=int(dest[1],16))>=end:continue
        body=[v for pc,v in ops.items() if start<=pc<=end]
        observed=[int(h[1]) for v in body if (h:=re.search(r'HGMMA.64x(\d+)x16.F32.BF16',v))]
        if len(set(observed))!=1 or len(observed)!=4*bands*stages:continue
        width=observed[0]
        if width not in loops or len(body)<loops[width]['sites']:
            loops[width]=dict(sites=len(body),HGMMA=len(observed),waits=sum('WARPGROUP.DEPBAR.LE' in v for v in body),
                               HMUL2=sum('HMUL2' in v for v in body),PRMT=sum('PRMT' in v for v in body),
                               local=sum(bool(re.search(r'\b(?:LDL|STL)\b',v)) for v in body))
    log=image.with_suffix('.log').read_text()
    frame,store,load=map(int,re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log).groups())
    result.setdefault(f'{proj}_m{m}_n{n}',{})[variant]=dict(loops=loops,source=digest(image.with_suffix('.cu')),
        sass=digest(image),cubin=digest(image.with_suffix('.cubin')),frame=frame,spill_stores=store,spill_loads=load,
        native_binary_gate=env['has_live_decode_isa'](sass,m,bands=bands,stages=stages,full_stage=(m==8)),
        serialized='C7512' in log)
print(json.dumps(result,indent=2,sort_keys=True))
