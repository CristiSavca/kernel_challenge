"""Audit SM90a back edges and projection pointer bounds using the frozen export."""

from pathlib import Path
import gzip
import hashlib
import json
import re


root = Path(__file__).resolve().parent
patch = (root / "patch.diff").read_text()
path = "python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh"
section = patch.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("diff --git ", 1)[0]
header = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))+"\n"
assert hashlib.sha256(header.encode()).hexdigest()=="901a1c8ebfe37ed4a67586ad8766e27daa7d783a22f51097d35c13078896c43e"
for expression in (
    "constexpr uint32_t ahead = Gate ? 1 : 2", "next = ahead; next < tiles; ++next",
    "i = 0; i < ahead; ++i", "window[i - 1] = window[i]", "window[ahead - 1] = future",
    "const Slice& cur = window[i]", "start + size_t(next) * stored_width, 16 * next",
    "bank.weight + 2 * base + lane * 4", "bank.scale + base + (lane / 4) * 8",
    "input + k + 8", "hidden % (64 * kWarps) == 0 && inner % (64 * kWarps) == 0",
):
    assert expression in header
schedule_cases = pointer_bound_checks = slices = 0
for gate in (False,True):
    ahead = 1 if gate else 2
    for width in range(128,16385,128):
        tiles = width//16
        fetched = list(range(ahead))
        window = list(fetched)
        used = []
        for successor in range(ahead,tiles):
            fetched.append(successor)
            used.append(window.pop(0))
            window.append(successor)
        used.extend(window)
        assert used==fetched==list(range(tiles))
        schedule_cases+=1
        slices+=tiles
        for output in (128,256,4096,8192):
            planes=2 if gate else 1
            stride=planes*output
            for expert in (0,31,63):
                for tile in (0,tiles-1):
                    for col in (0,output-64):
                        for plane in range(planes):
                            base=(expert*tiles+tile)*stride+col+plane*output
                            bank_end=(expert*tiles+tile)*stride+(plane+1)*output
                            for lane in range(32):
                                assert 2*base+lane*4+4<=2*bank_end
                                assert base+lane//4*8+8<=bank_end
                                assert tile*16+(lane%4)*2+8+2<=width
                                pointer_bound_checks+=3

sass=gzip.decompress((root/"sm90a.sass.gz").read_bytes()).decode()
functions=re.split(r"^\s*Function : (\S+)",sass,flags=re.MULTILINE)
stats={}
for name,body in zip(functions[1::2],functions[2::2]):
    if "17projection_kernelIL" not in name:
        continue
    gate="ILb1" in name
    ins={int(a,16):op.strip() for a,op in re.findall(r"/\*([a-f0-9]+)\*/\s+([^\n;]*?)\s*;",body)}
    expected_hmma=8 if gate else 4
    candidates=[]
    for pc,op in ins.items():
        branch=re.search(r"\bBRA(?:\.\w+)*\s+(0x[\da-f]+)",op)
        if not branch:
            continue
        entry=int(branch[1],16)
        if entry>=pc:
            continue
        loop=[val for addr,val in ins.items() if entry<=addr<=pc]
        if sum("HMMA.16816.F32.BF16" in v for v in loop)==expected_hmma:
            candidates.append((entry,pc,loop))
    assert len(candidates)==1
    entry,pc,loop=candidates[0]
    loads=[re.search(r"LDG\.E\.(64|128)\b",v) for v in loop]
    bits=sorted(int(m[1]) for m in loads if m)
    assert bits==([64,64,128,128] if gate else [64,128])
    assert not any(re.search(r"\b[LS]DL",v) for v in loop)
    stats["W13" if gate else "W2"]=dict(start=hex(entry),end=hex(pc),sites=len(loop),
                                        hmma=expected_hmma,packed_load_bits=bits)
print(json.dumps(dict(header_sha256=hashlib.sha256(header.encode()).hexdigest(),
                      sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),schedule_cases=schedule_cases,
                      k_slices=slices,pointer_bound_checks=pointer_bound_checks,steady_loops=stats,
                      physical_gpu=False),indent=2))
