"""Verify admission on generated cubins and replay the recorded opcode counts after cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy

here=Path(__file__).resolve().parent
args=argparse.ArgumentParser()
args.add_argument("stage",type=Path)
args.add_argument("w13",type=Path,nargs="?")
args.add_argument("w2",type=Path,nargs="?")
p=args.parse_args()
source=p.stage/"python/sglang/srt/layers/muse_spark/live_decode_reduce.py"
admit=runpy.run_path(str(source))["has_live_decode_isa"]
if p.w13 is None:
    receipt=json.loads((here/"independent_spill_guard.json").read_bytes())
    samples=receipt["samples"]
else:
    assert p.w2 is not None
    samples=[]
    entries=[]
    for m in (16,24):
        entries.extend([(p.w13/"final-built"/f"w13_m{m}_n512.sass",m,4,True),
                        (p.w2/"kept"/f"w2_m{m}_n256.sass",m,3,True)])
    for name,m,st in [("w13_m16_n512_s3r96_single",16,3),("w13_m24_n512_s3r112_single",24,3),
                      ("w13_m24_n512_s4r112_single",24,4)]:
        entries.append((p.w13/"scanned"/(name+".sass"),m,st,False))
    for path,m,st,accepted in entries:
        text=path.read_text()
        assert admit(text,m,bands=4,stages=st) is accepted
        counts={f"HGMMA.64x{w}x16.F32.BF16":text.count(f"HGMMA.64x{w}x16.F32.BF16") for w in range(8,m+1,8)}
        counts["WARPGROUP.DEPBAR.LE"]=text.count("WARPGROUP.DEPBAR.LE")
        counts.update({name:len(re.findall(r"\b"+name+r"\b",text)) for name in ("LDL","STL")})
        samples.append(dict(image=path.stem,assembly_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            rows=m,stages=st,accepted=accepted,opcode_counts=counts))
for case in samples:
    counts=case["opcode_counts"]
    stripped="\n".join((key+"\n")*value for key,value in counts.items() if key not in ("LDL","STL"))
    m,st=case["rows"],case["stages"]
    assert admit(stripped,m,bands=4,stages=st)
    assembly=stripped+" LDL.U.128 R1, [R2];\n"*counts["LDL"]+" STL [R3], R4;\n"*counts["STL"]
    assert admit(assembly,m,bands=4,stages=st) is case["accepted"]
    assert not admit(stripped+" LDL R1, [R2];\n",m,bands=4,stages=st)
    assert not admit(stripped+" STL.128 [R2], R4;\n",m,bands=4,stages=st)
print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),samples=samples,
    meaning="Compiled, spill-heavy alternatives have the required HGMMA and drain counts. The final startup predicate rejects them and selects the original narrow path. The four selected wide images pass. Histograms reproduce the admission result after the binaries are retired."),indent=2))
