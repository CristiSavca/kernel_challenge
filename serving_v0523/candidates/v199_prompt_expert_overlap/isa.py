"""Check actual SM90a register liveness, producer and consumer backedges, and device resources."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

here = Path(__file__).resolve().parent
scratch = Path(sys.argv[1])
sha = lambda data: hashlib.sha256(data).hexdigest()
fixture = (here.parent / "v183_paired_grouped_mma/isa.py").read_text()
exec(compile(fixture[fixture.index("mac ="):fixture.index("\n\nout={}")], "v183/isa.py", "exec"))
def constant(path, name):
    binary = path.read_bytes()
    shoff = int.from_bytes(binary[40:48],"little")
    stride, count = struct.unpack_from("<HH",binary,58)
    sections = [struct.unpack_from("<IIQQQQIIQQ",binary,shoff+stride*i) for i in range(count)]
    sym = next(s for s in sections if s[1]==2)
    strings = sections[sym[6]][4]
    for i in range(sym[4],sym[4]+sym[5],sym[9]):
        start,info,_,section,offset,size=struct.unpack_from("<IBBHQQ",binary,i)
        if binary[strings+start:binary.index(0,strings+start)].decode()==name:
            addr=sections[section][4]+offset
            return int.from_bytes(binary[addr:addr+size],"little")
    raise ValueError(name)
def loops(path):
    src=path.read_text()
    ops={int(a,16):s for a,s in re.findall(r'^\s*/\*([0-9a-f]+)\*/\s+(.+?)\s*;',src,re.M)}
    mma=[i for i,s in ops.items() if s.startswith('HGMMA')]
    backs=[(int(j[1],16),i) for i,s in ops.items() if (j:=re.search(r'BRA (0x[0-9a-f]+)$',s)) and int(j[1],16)<i]
    consumer=min((p for p in backs if p[0]<=min(mma)<=max(mma)<p[1]),key=lambda p:p[1]-p[0])
    stages=3 if path.stem in ("arith","palette") else 4
    tma_per_stage=2 if path.stem in ("pipeline","stages","16_256_4") else 3
    producer=min(((a,b) for a,b in backs if b<min(mma) and
                  sum("UTMALDG" in op for pos,op in ops.items() if a<=pos<=b)>=stages*tma_per_stage),
                 key=lambda bounds: bounds[1]-bounds[0])
    return tuple(dict(start=hex(a),end=hex(b),sites=(b-a)//16+1,
                tma=sum('UTMALDG' in s for i,s in ops.items() if a<=i<=b)) for a,b in (consumer,producer))

out={}
for name in ["4_256_4","arith","palette","scales_ready","stages","pipeline","16_256_4"]:
    p=scratch/'rendered'/name
    if name not in ("4_256_4","pipeline","16_256_4") and not p.with_suffix(".sass").exists():
        continue  # Optional rejected probes are diagnostic builds, not part of the deployed source.
    c,l=loops(p.with_suffix(".sass"))
    d=dict(consumer=c,loader=l,shared_bytes=constant(p.with_suffix(".cubin"),"SMEM_SIZE"),
            cubin_sha256=sha(p.with_suffix(".cubin").read_bytes()),sass_sha256=sha(p.with_suffix(".sass").read_bytes()))
    if name in ("4_256_4","pipeline"):
        d["register_audit"]=parse(p)
    out[name]=d
native,final=(out[k] for k in ("4_256_4","pipeline"))
assert [i["register_audit"]["group_sizes"] for i in (native,final)]==[[2]*16,[4]*8]
assert [i["loader"]["sites"] for i in (native,final)]==[285,385]
assert [i["consumer"]["sites"] for i in (native,final)]==[1127,1003]
assert [i["shared_bytes"] for i in (native,final)]==[128000,136192]
print(json.dumps(dict(nvcc="12.9.86",arch="sm_90a",humming="0.1.12",physical_gpu=False,
    weighted_static_sites_per_four_K64_stages=[x["consumer"]["sites"]*8+x["loader"]["sites"]*4 for x in (native,final)],
    images=out),indent=2))
