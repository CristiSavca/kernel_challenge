"""Compare every emitted specialization's K-warp body and FMA to the native width-specific equivalents."""
import hashlib,json,re,sys
from pathlib import Path

root=Path(sys.argv[1])
def body(text,pattern):
    m=re.search(pattern,text);assert m,pattern
    start=m.end()-1;level=1
    for at in range(start+1,len(text)):
        if text[at]=='{':level+=1
        elif text[at]=='}':level-=1
        if not level:return text[start+1:at].strip()
    raise ValueError(pattern)
out={}
for phase in ('w13','w2'):
    for full in (8,16,24):
        name=f'{phase}_{full}.cu'
        native=(root/'base'/name).read_text()
        shader=(root/'live'/name).read_text()
        native_mma=body(native,r'struct MuseOverlapMma :[^\n]*\{')
        new_mma=body(shader,r'struct MuseOverlapMma :[^\n]*\{')
        run=body(native_mma,r'void run\(uint32_t stage, uint32_t iteration\) \{')
        if full==8:
            assert native_mma==new_mma
        else:
            overload=body(new_mma,r'void run_as\(uint32_t stage, uint32_t iteration\) \{')
            assert overload.replace('SliceOp::fma','MmaOpClass::fma')==run
        widths={}
        for w in range(8,full+1,8):
            native_width=(root/'base'/f'{phase}_{w}.cu').read_text()
            native_op=body(native_width,r'class MmaOpClass \{')
            emitted_op=body(shader,r'class MmaOpClass \{' if w==full else f'struct MuseResidual{w} '+r'\{')
            signature=r'static void fma\([^\n]*\) \{'
            op=body(native_op,signature)
            assert body(emitted_op,signature)==op
            assert f'm64n{w}k16.f32.bf16.bf16' in op
            widths[w]=hashlib.sha256(op.encode()).hexdigest()
        out[name]=dict(K16_body_sha256=hashlib.sha256(run.encode()).hexdigest(),native_fma_body_sha256=widths)
print(json.dumps(out,indent=2))
