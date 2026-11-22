import hashlib,json,re,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
rows=[]
for fname in ("live/w13_8","base/w13_8","ordinary/w13_8","live/w13_16","live/w13_24","live/w2_8","final/prompt_176"):
 p=root/fname
 a=p.with_suffix(".sass").read_text(); log=p.with_suffix(".log").read_text()
 tma="UTMALDG.4D" if "w2" not in fname else "UTMALDG"
 widths={s:a.count(f"HGMMA.64x{s}x16.F32.BF16") for s in [8,16,24,176] if f"HGMMA.64x{s}x16.F32.BF16" in a}
 assert widths and tma in a
 spills=re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",log)
 assert spills
 rows.append(dict(kernel=fname,source_sha256=hashlib.sha256(p.with_suffix(".cu").read_bytes()).hexdigest(),
   assembly_sha256=hashlib.sha256(a.encode()).hexdigest(),registers=int(re.search(r"Used (\d+) registers",log)[1]),
   stack_bytes=int(spills[1]),spill_loads=int(spills[3]),spill_stores=int(spills[2]),instructions_by_M=widths,
   warpgroup_wait=a.count("WARPGROUP.DEPBAR.LE"),TMA2D=a.count("UTMALDG.2D"),TMA4D=a.count("UTMALDG.4D")))
assert all(not r["spill_loads"] and not r["spill_stores"] and r["instructions_by_M"]=={8:32} and r["TMA2D"] and r["TMA4D"] and r["warpgroup_wait"]==k
    for r,k in zip(rows[:3],[8,8,12]))
print(json.dumps(dict(compiler="NVCC release 12.9, SM90a",compiler_target="compute_90a/sm_90a",rows=rows,
    launcher_SHA256=hashlib.sha256((root/"launcher.o").read_bytes()).hexdigest(),compiler_source="Humming 0.1.12 production templates; actual launcher wrappers compiled to object",
    physical_H200=False),indent=2))
