"""Compare matched SM90 and SM90a disassembly across packed staging alternatives."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def functions(sass):
    return {s.splitlines()[0]:s for s in sass.split("Function : ")[1:]}


def machine_code(body):
    words=re.findall(r"/\* (0x[\da-f]{16}) \*/",body)
    assert len(words)%2==0
    return b"".join(int(w,16).to_bytes(8,"little") for w in words)


def kernel(body, log, name):
    ins = {int(pc,16):op for pc,op in re.findall(r"^\s*/\*([\da-f]+)\*/\s+(.*?)\s*;\s*/\*",body,re.M)}
    loops=[]
    for pc,op in ins.items():
        target=re.search(r"\bBRA (0x[\da-f]+)",op)
        if target is None or int(target[1],16)>=pc:continue
        start=int(target[1],16)
        block=[s.split()[int(s.startswith("@"))].split(".")[0] for i,s in ins.items() if start<=i<=pc]
        count=Counter(block)
        if count["HMMA"]:loops.append((len(block),start,pc,count))
    n,a,b,counts=min(loops,key=lambda v:v[0])
    match=re.search(r"Compiling entry function '"+name+r"'.*?(\d+) bytes spill stores, (\d+) bytes spill loads.*?"
                    r"Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?",log,re.S)
    assert match and int(match[1])==int(match[2])==0
    return dict(registers=int(match[3]),shared_bytes=int(match[5] or 0),barriers=int(match[4]),
                spill_store=int(match[1]),spill_load=int(match[2]),whole_sites=len(ins),
                steady_backedge=dict(begin=hex(a),end=hex(b),sites=n,
                    histogram={k:counts[k] for k in ["HMMA","HMUL2","LDG","LDGSTS","LDS","BAR","DEPBAR"]}))


def main(folder):
    builds={}
    for tag in ("base","coop","direct","inputs","final","final-low"):
        j={}
        sha={}
        for item in (folder/(tag+".sha")).read_text().splitlines():
            digest,name=item.split()
            sha[name]=digest
        for arch in ("sm_90a","sm_90"):
            sass=(folder/(tag+"."+arch+".sass")).read_text()
            assert hashlib.sha256(sass.encode()).hexdigest()==sha[arch+".sass"]
            sections=functions(sass)
            log=(folder/(tag+".log")).read_text().split("ptxas info    : 0 bytes gmem")
            log=next(s for s in log if f"for '{arch}'" in s)
            kernels={}
            for name,body in sections.items():
                if "projection" not in name or "medium" in name:continue
                kernels[("cooperative" if "channels" in name else "parent")+("_W13" if "ILb1" in name else "_W2")]=kernel(body,log,name)
            if tag=="final-low":
                exported=functions((folder/("final."+arch+".sass")).read_text())
                for name,body in sections.items():
                    assert machine_code(body)==machine_code(exported[name]),name
            j[arch]=kernels
            if tag=="final":
                base=functions((folder/("base."+arch+".sass")).read_text())
                common={}
                for name,src in base.items():
                    before,after=machine_code(src),machine_code(sections[name])
                    assert before==after,name
                    common[name] = dict(instruction_sites=len(before)//16, sha256=hashlib.sha256(before).hexdigest())
                j[arch+"_identical_parent_functions"]=common
        builds[tag]=dict(files=sha,results=j)
    return dict(nvcc="12.9.86",native_host="Ubuntu aarch64 with apache-tvm-ffi headers",
                architecture=["sm_90","sm_90a"],flags=["-O3","--use_fast_math","--expt-extended-lambda",
                    "--expt-relaxed-constexpr","-std=c++20"],hardware=False,builds=builds)


if __name__=="__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
