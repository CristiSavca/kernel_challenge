"""Inspect the steady K backedges of the six matched SM90a ptxas builds."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

TAGS = ("base", "test", "swar", "table", "marlin", "imad")


def analyze(path, tag):
    sass = (path / (tag + ".sass")).read_bytes()
    ptxas = (path / (tag + ".log")).read_text()
    result = {}
    logs = {s.split("'")[0]:s for s in ptxas.split("Compiling entry function '")[1:]}
    for body in sass.decode().split("Function : ")[1:]:
        symbol = body.splitlines()[0]
        if "projection" not in symbol:
            continue
        key = ("medium" if "medium" in symbol else "low") + ("_W13" if "ILb1" in symbol else "_W2")
        regs, barriers = re.search(r"Used (\d+) registers, used (\d+) barriers", logs[symbol]).groups()
        spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", logs[symbol]).groups()
        shared = re.search(r"(\d+) bytes smem", logs[symbol])
        ops = {int(pc, 16): op for pc, op in re.findall(r"^\s*/\*([\da-f]+)\*/\s+(.*?)\s*;\s*/\*",body,re.M)}
        loops=[]
        for pc, op in ops.items():
            branch = re.search(r"\bBRA (0x[\da-f]+)",op)
            if branch is None or int(branch[1],16) >= pc:
                continue
            start = int(branch[1],16)
            block = [o for address,o in ops.items() if start <= address <= pc]
            histogram = Counter(o.split()[1 if o.startswith("@") else 0].split(".")[0] for o in block)
            if histogram["HMMA"] and histogram["LDG"] and len(block)<800:
                loops.append(dict(begin=hex(start),end=hex(pc),sites=len(block),
                    ops={k:histogram[k] for k in ("HMMA","HMUL2","PRMT","LOP3","SHF","IMAD","LDG","LDS")}))
        assert len(loops)==(2 if key=="medium_W13" else 3 if key=="medium_W2" else 1)
        result[key] = dict(regs=int(regs),barriers=int(barriers),shared=int(shared[1]) if shared else 0,
                           spill_store=int(spill[0]),spill_load=int(spill[1]),function_sites=len(ops),loops=loops)
    return dict(sass_sha256=hashlib.sha256(sass).hexdigest(), kernels=result)


if __name__ == "__main__":
    print(json.dumps({k:analyze(Path(sys.argv[1]),k) for k in TAGS},indent=2))
