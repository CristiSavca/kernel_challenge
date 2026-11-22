"""Read the actual per-tile steady SM90a paths and verify pending register-source lifetimes."""
import hashlib,json,re,runpy,sys
from pathlib import Path

root=Path(sys.argv[1]); here=Path(__file__).resolve().parent
shared=runpy.run_path(str(here.parent/'v199_prompt_expert_overlap/independent_ring.py'))['resident_bytes']
gate=runpy.run_path(str(root/'sglang/python/sglang/srt/layers/muse_spark/live_decode_reduce.py'))['has_live_decode_isa']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inspect(path,widths):
    text=path.read_text()
    ops={int(m[1],16):m[2] for line in text.splitlines()
         if (m:=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\* 0x',line))}
    loops={}
    for w in widths:
        loops_by_size=[]
        for end,op in ops.items():
            b=re.search(r'\bBRA(?:\.U)? (0x[0-9a-f]+)',op)
            if b and (start:=int(b[1],16))<end:
                block=[v for pc,v in ops.items() if start<=pc<=end]
                if sum('HGMMA.' in x for x in block)==sum(f'HGMMA.64x{w}x16.F32.BF16' in x for x in block)==32:
                    loops_by_size.append((len(block),start,end,block))
        n,begin,end,block=min(loops_by_size)
        held=set();pending,total,protected=0,0,0;drains=[];arrivals=[]
        for inst in block:
            if 'HGMMA.' in inst:
                m=re.search(r'HGMMA\.\S+ R\d+, R(\d+), gdesc\[',inst); assert m,inst
                base=int(m[1]);held.update(range(base,base+4));pending+=1;total+=1
            elif 'WARPGROUP.DEPBAR.LE' in inst:
                assert pending==4;held.clear();drains.append(pending);pending=0
            elif 'SYNCS.ARRIVE' in inst:
                assert not pending;arrivals.append(total)
            elif held and (m:=re.search(r'(?:^| )([A-Z]+[.A-Z0-9_]*) R(\d+)(?:,|\.)',inst)):
                op,base=m[1],int(m[2])
                if op.startswith(('ST','BRA','WARPGROUP','SYNCS','NOP')):continue
                count=4 if op.startswith('LDS.128') else 2 if op.startswith('LDS.64') else 1
                assert held.isdisjoint(range(base,base+count)),inst
                protected+=count
        assert arrivals[:4]==[8,16,24,32] and len(drains)==8 and not held
        assert not any('LDL' in x or 'STL' in x for x in block)
        loops[w]=dict(sites=n,start=hex(begin),end=hex(end),HGMMA=total,groups=drains,
                      stage_release_after=arrivals,physical_register_words_checked=protected)
    cub=path.with_suffix('.cubin');log=path.with_suffix('.log').read_text()
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in log
    opcode=set(re.findall(r"\b(FADD(?:\.[A-Z_]+)*) ",text))
    assert opcode=={"FADD.FTZ"},opcode
    return dict(reduction_fadd=list(opcode),source=sha(path.with_suffix('.cu')),cubin=sha(cub),sass=sha(path),bytes=cub.stat().st_size,
        regs=int(re.search(r'Used (\d+) registers',log)[1]),shared_bytes=shared(cub),stack_spill_load_store=0,
        whole_static_bar_sync=sum('BAR.SYNC' in op for op in ops.values()),loops=loops)
result={}
for phase in ('w13','w2'):
    for width in (8,16,24):
        name=f'{phase}_{width}'
        native=inspect(root/'base'/(name+'.sass'),[width])
        path=root/'live'/(name+'.sass')
        new=inspect(path,range(8,width+1,8));assert new['regs']==native['regs']==96
        assert new['shared_bytes']==native['shared_bytes'] and gate(path.read_text(),width)
        assert not gate(path.read_text().replace('WARPGROUP.DEPBAR.LE','removed',1),width)
        for w in range(8,width+1,8):
            assert not gate(path.read_text().replace(f'HGMMA.64x{w}x16.F32.BF16','removed',1),width)
        result[name]=dict(V211=native,V213=new)
print(json.dumps(dict(nvcc='12.9.86',arch='sm90a',scope='Actual Humming 0.1.12 code, GPU-less compiler',cases=result),indent=2))
