"""Verify native emitted WGMMA widths, build resource logs and archived identity."""
from pathlib import Path
import json, hashlib, re, sys
root = Path(sys.argv[1])
result={}
for n in (4096,8192):
    for m in (136,152,176):
        stem=root/f"{n}_{m}"
        log=stem.with_suffix(".log").read_text(); sass=stem.with_suffix(".sass").read_text()
        ops=re.findall(r'\b(HGMMA\.\d+x\d+x\d+\.F32\.BF16)\b',sass)
        assert len(ops)>0 and set(ops)=={f"HGMMA.64x{m}x16.F32.BF16"}
        spills=re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",log)
        resources=re.search(r"Used (\d+) registers, used (\d+) barriers", log)
        assert spills and resources
        paths={ext:stem.with_suffix("."+ext) for ext in ("cu","cubin","sass","log")}
        result[stem.name]=dict(registers=int(resources[1]),barriers=int(resources[2]),stack_frame=int(spills[1]),
            spill_stores=int(spills[2]),spill_loads=int(spills[3]),hgmma_opcode=ops[0],hgmma_static_sites=len(ops),
            file_sha256={ext:hashlib.sha256(path.read_bytes()).hexdigest() for ext,path in paths.items()},
            cubin_bytes=paths["cubin"].stat().st_size)
print(json.dumps(dict(nvcc="12.9.86",arch="sm_90a",hardware=False,
    humming="0.1.12",wheel_sha256="cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331",
    kernels=result),indent=2))
