"""Check actual complete-stage and control kernels including asynchronous register lifetimes."""
import hashlib
import json
from pathlib import Path
import re
import sys

here=Path(__file__).resolve().parent
stage, images, controls=map(Path,sys.argv[1:])
parent=here.parent/'v297_wide_fused_decode'
checker=parent/'independent_lifetimes.py'
original=checker.read_text()
def checker_for(size):
    txt=original.replace('assert set(group_sizes) == {4}',f'assert set(group_sizes) == {{{size}}}')
    if size==8:
        txt=txt.replace('if wide:\n                assert src.isdisjoint','if True:\n                assert src.isdisjoint')
    ns=dict(__name__='lifetime_oracle')
    exec(compile(txt,str(checker),'exec'),ns)
    return ns
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def inspect(image, size):
    sass=image.with_suffix('.sass').read_text()
    log=image.with_suffix('.log').read_text()
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in log and 'C7512' not in log
    oracle=checker_for(size); inst=oracle['insns'](sass)
    safety=oracle['verify'](inst,False)
    body_lengths=[]
    for end,_,op,arg in inst:
        if op=='BRA' and (start:=int(arg,16))<end:
            body=[o for pc,_,o,_ in inst if start<=pc<=end]
            if sum(o.startswith('HGMMA.') for o in body)==32:
                body_lengths.append(len(body))
    return dict(source=sha(image),cubin=sha(image.with_suffix('.cubin')),sass=sha(image.with_suffix('.sass')),
                loop_sites=min(body_lengths),no_spill=True,no_serialization=True,
                sass_registers=int(re.search(r'Used (\d+) registers',log)[1]),**safety)
guard_ns={}
helper=stage/'python/sglang/srt/layers/muse_spark/live_decode_reduce.py'
exec(compile(helper.read_text(),str(helper),'exec'),guard_ns)
guard=guard_ns['has_live_decode_isa']
metrics={}
for name in ('w13_m8_n256','w2_m8_n128'):
    cur,prev=(d/(name+'.cu') for d in (images,controls))
    good=inspect(cur,8);baseline=inspect(prev,4)
    current=cur.with_suffix('.sass').read_text();old=prev.with_suffix('.sass').read_text()
    assert guard(current,8,full_stage=True) and not guard(old,8,full_stage=True)
    assert guard(old,8) and not guard(current,8)
    assert not guard(current+'\n /*0020*/ STL [R1], R4;',8,full_stage=True)
    metrics[name]=dict(selected=good,parent=baseline,wrong_wait_or_spill_rejected=True)
other={}
w13=json.loads((parent/'w13_sm90a.json').read_text())
w2=json.loads((parent/'w2_sass.json').read_text())['cases']
for path in sorted(images.glob('*.sass')):
    if path.stem in metrics: continue
    ref=(w13|w2)[path.stem]
    # Strip ELF source sections and module-level assembler metadata only for
    # comparison. Actual disassembled executable instructions must match.
    assert sha(path)==ref['sass_sha256'],path.stem
    other[path.stem]=dict(sass_sha256=sha(path),cubin_sha256=sha(path.with_suffix('.cubin')))
print(json.dumps(dict(m8=metrics,unchanged_machine_schedules=other,guard_sha256=sha(helper)),
                 indent=2,sort_keys=True))
