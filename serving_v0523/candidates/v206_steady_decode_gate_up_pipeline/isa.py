"""Compare reassembled V187 and V189 SM90a kernels and their asynchronous register lifetimes."""
import hashlib
import json
from pathlib import Path
import re
import sys

scratch = Path(sys.argv[1])
here = Path(__file__).resolve().parent
stage = scratch / "sglang/python/sglang/kernels/jit/csrc/muse_spark"
sha = lambda data: hashlib.sha256(data).hexdigest()
mac = re.compile(r"HGMMA\.64x(\d+)x16\.F32\.BF16 R(\d+), R(\d+), gdesc\[UR(\d+)\], R(\d+)")
work = {"LOP3", "IMAD", "HMUL2", "PRMT", "SHF"}


def parse(path):
    lines = path.with_suffix(".sass").read_text()
    ins = {int(p,16): code for p,code in re.findall(r"^\s*/\*([0-9a-f]+)\*/\s+(.+?)\s*;",lines,re.M)}
    hgmma = [i for i,op in ins.items() if mac.match(op)]
    assert len(hgmma) == 32
    bounds = [(int(m[1],16),i) for i,op in ins.items() if (m:=re.search(r"\bBRA (0x[0-9a-f]+)$",op)) and
              int(m[1],16) <= min(hgmma) <= max(hgmma) < i]
    lo,hi = min(bounds,key=lambda b:b[1]-b[0])
    inst = {i:s for i,s in ins.items() if lo<=i<=hi}
    assert not any(re.search(r"\b(?:LDL|STL)\b",s) for s in inst.values())
    groups = []; live = set(); since = 0; all_overlap=0; gm_count=0; g_total=0; write_regs=0
    for pc,code in inst.items():
        if m:=mac.match(code):
            n,d,src,desc,c=map(int,m.groups())
            assert d==c
            live.update(range(d,d+n//2));live.update(range(src,src+4))
            gm_count+=1;g_total+=1
            continue
        if "WARPGROUP.DEPBAR.LE" in code:
            assert gm_count and "0x0" in code
            groups.append(dict(mma=gm_count,other_operand_work=since))
            live.clear();gm_count=0;since=0
            continue
        if not live:
            continue
        # All ops encountered in active intervals that take ordinary registers
        # either write operand zero (with the indicated vector width), or use
        # only predicate / uniform or special-register destinations.
        word=re.sub(r"^@!?P\d+\s+", "", code).split()[0]
        op=word.split(".")[0]
        m=re.match(r"^(?:@!?P\d+\s+)?\S+\s+R(\d+)(?:,|\s|$)",code)
        if m:
            if op in work:
                all_overlap+=1
                if gm_count==2: since+=1
            assert op in work|{"IADD3","SEL","LDS","MOV","VIMNMX","VIADD","IMNMX","LEA","FSEL","S2R"},code
            n=4 if ".128" in word else 2 if ".64" in word else 1
            dst=set(range(int(m[1]),int(m[1])+n))
            assert not dst&live,(hex(pc), code, dst&live)
            write_regs+=n
    assert not live and g_total==32
    log=path.with_suffix(".log").read_text()
    regs=int(re.search(r"Used (\d+) registers",log)[1])
    spills=tuple(map(int,re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",log).groups()))
    return dict(cubin_sha256=sha(path.with_suffix(".cubin").read_bytes()), source_sha256=sha(path.with_suffix(".cu").read_bytes()),
        sass_sha256=sha(lines.encode()),registers=regs,stack_spill_store_load_bytes=spills,
        hot_loop=[hex(lo),hex(hi)],instructions=len(inst),mma=g_total,drains=len(groups),recurrent_local=0,
        payload_ops_while_any_mma_outstanding=all_overlap,
        group_sizes=[g["mma"] for g in groups],payload_ops_while_2_mmas_outstanding=sum(g["other_operand_work"] for g in groups),
        checked_non_wgmma_register_writes_during_outstanding=write_regs,
        tensor_2d=lines.count("UTMALDG.2D"),tensor_3d=lines.count("UTMALDG.3D"),tensor_4d=lines.count("UTMALDG.4D"))


out={}
old=json.loads((here.parent/"v197_grouped_prompt_gate_up/sm90a.json").read_bytes())["kernels"]
for w in (8,16,24):
    name=f"decode_{w}"
    before=parse(scratch/"old"/name)
    after=parse(scratch/"full"/name)
    assert old[name]["sass_sha256"]==before["sass_sha256"] and old[name]["cubin_sha256"]==before["cubin_sha256"]
    assert before["group_sizes"]==[4,2,2]*4 and after["group_sizes"]==[4]*8
    assert all(before[k]==after[k] for k in ("registers","tensor_2d","tensor_4d","stack_spill_store_load_bytes"))
    assert after["registers"]==96 and after["stack_spill_store_load_bytes"]==(0,0,0)
    out[name]=dict(parent=before,candidate=after)
prompt=scratch/"full/prompt_176.cubin"
assert sha(prompt.read_bytes())==old["prompt_176"]["cubin_sha256"]
print(json.dumps(dict(arch="sm_90a",nvcc="12.9.86",humming_version="0.1.12",physical_gpu=False,
    prompt_cubin_sha256=sha(prompt.read_bytes()),kernels=out),indent=2))
