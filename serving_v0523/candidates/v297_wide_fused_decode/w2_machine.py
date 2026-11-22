"""Verify the emitted Hopper loops, async input-register lifetimes and each native live-reduction branch."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
import subprocess

HERE=Path(__file__).resolve().parent
stage,images=map(Path,sys.argv[1:])
startup=json.loads((HERE/'w2_startup.json').read_bytes())
gate=runpy.run_path(str(stage/'python/sglang/srt/layers/muse_spark/live_decode_reduce.py'))['has_live_decode_isa']
resident=runpy.run_path(str(HERE.parent/'v199_prompt_expert_overlap/independent_ring.py'))['resident_bytes']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def writes(op):
    m=re.search(r'(?:^| )([A-Z]+[.A-Z0-9_]*) R(\d+)(?:,|\.)',op)
    if not m or m[1].startswith(('ST','BRA','WARPGROUP','SYNCS','NOP')):return set()
    width=4 if m[1].endswith('.128') else 2 if m[1].endswith('.64') else 1
    return set(range(int(m[2]),int(m[2])+width))

def inspect(name,meta):
    p=images/name
    text=p.with_suffix('.sass').read_text()
    log=p.with_suffix('.log').read_text()
    m=meta['block'][0];bands=meta['warp'][1]//16;stages=meta['stages']
    n=4*bands*stages
    assert digest(p.with_suffix('.cu'))==meta['source_sha256'] and resident(p.with_suffix('.cubin'))==meta['smem']
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in log
    assert gate(text,m,bands=bands,stages=stages) and 'UTMALDG.3D' in text
    loops={}
    ops={int(x[1],16):x[2] for l in text.splitlines() if (x:=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\* 0x',l))}
    for w in range(8,m+1,8):
        mma='HGMMA.64x'+str(w)+'x16.F32.BF16'
        assert not gate(text.replace(mma,'invalid',1),m,bands=bands,stages=stages)
        opts=[]
        for end,op in ops.items():
            back=re.search(r'\bBRA(?:\.U)? (0x[0-9a-f]+)',op)
            if not back or (start:=int(back[1],16))>=end:continue
            body=[v for pc,v in ops.items() if start<=pc<=end]
            if sum(mma in v for v in body)==sum('HGMMA.' in v for v in body)==n:
                opts.append((len(body),start,end,body))
        length,low,high,body=min(opts)
        held=set();since_fence=set();pending=total=protected=0;releases=[];groups=[];fences=0
        for op in body:
            if 'WARPGROUP.ARRIVE' in op:
                since_fence.clear();fences+=1
            elif mma in op:
                src=int(re.search(r'HGMMA\.\S+ R\d+, R(\d+), gdesc\[',op)[1])
                fragment=set(range(src,src+4))
                if bands==4: assert not fragment & since_fence, op
                held |= fragment; pending+=1;total+=1
            elif 'WARPGROUP.DEPBAR.LE' in op:
                assert pending==4 and '0x0' in op
                groups.append(pending);pending=0;held.clear()
            elif 'SYNCS.ARRIVE' in op:
                assert not pending
                releases.append(total)
            else:
                dest=writes(op)
                if held:protected+=len(dest)
                assert not (held & dest),op
                since_fence |= dest
            assert not ('LDL' in op or 'STL' in op)
        assert not held and releases[:stages]==list(range(bands*4,(stages+1)*bands*4,bands*4))
        if bands==4:assert fences>=total
        loops[w]=dict(sites=length,hgmma=total,waits=len(groups),wg_arrivals=fences,
                      math_release_at=releases,protected_register_writes_checked=protected)
    adds=set(re.findall(r'\b(FADD(?:\.[A-Z_]+)*) ',text));assert adds=={'FADD.FTZ'}
    return dict(**meta,cubin_sha256=digest(p.with_suffix('.cubin')),sass_sha256=digest(p.with_suffix('.sass')),
                launch_registers=int(re.search(r'Used (\d+) registers',log)[1]),no_stack_or_spills=True,
                no_recurrent_local_accesses=True,fp32_reduction=sorted(adds),residual_widths=loops)

prefix="python/sglang/"
paths=[prefix+x for x in ("srt/layers/muse_spark/decode_mma_pair.py","srt/layers/muse_spark/live_decode_reduce.py","srt/layers/muse_spark/residual_prompt_mma.py",
                          "kernels/jit/csrc/muse_spark/grouped_decode_mma_pair.cuh","kernels/jit/csrc/muse_spark/live_decode_reduce.cuh")]
tree=subprocess.check_output(["git","-C",stage,"write-tree"],text=True).strip()
for p in paths:
    assert subprocess.check_output(["git","-C",stage,"show",tree+":"+p])==(stage/p).read_bytes()
print(json.dumps(dict(tree=tree, selected_source_sha256={p:digest(stage/p) for p in paths},compiler="CUDA 12.9.86; sm_90a; Humming 0.1.12",cases={n:inspect(n,m) for n,m in startup.items()},
    scope="Real offline compiled candidate and narrow fallback, preserving live K reduction under the production ISA selector. No H200 timing."),indent=2))
