"""Replay production Humming constructors including failure of the complete-stage image."""
import contextlib
import io
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
prev=here.parent/'v297_wide_fused_decode'
records={}
for phase in ('w13','w2'):
    driver=prev/f'render_{phase}.py'
    for refuse in (False,True):
        trace=[]
        code=driver.read_text()
        point='    assert kernel.both_pairs and kernel.live_reduction' if phase=='w13' else '    assert k.both_pairs and k.live_reduction'
        inst='kernel' if phase=='w13' else 'k'
        add=point+f'\n    trace.append(({inst}.block_shape[0], type({inst}).__name__))\n    if refuse and getattr({inst}, "full_stage", False):\n        raise ValueError("injected full-stage compile rejection")'
        assert code.count(point)==1
        code=code.replace(point,add)
        ns=dict(__file__=str(driver),__name__='offline_'+phase,trace=trace,refuse=refuse)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code,str(driver),'exec'),ns)
        m8=[cls for m,cls in trace if m==8]
        primary='FullStageDecodeGateUpKernel' if phase=='w13' else 'FullStageDecodeDownKernel'
        backup='LiveDecodeGateUpKernel' if phase=='w13' else 'LiveDecodeDownKernel'
        expect=[primary,backup] if refuse else [primary]
        assert m8==expect*(2 if phase=='w2' else 1),(m8,expect)
        assert all('FullStage' not in cls for m,cls in trace if m!=8)
        records[f'{phase}_decline_{refuse}']=dict(m8=m8,calls_at_other_widths=sorted(set(cls for m,cls in trace if m!=8)))
print(json.dumps(records,indent=2))
