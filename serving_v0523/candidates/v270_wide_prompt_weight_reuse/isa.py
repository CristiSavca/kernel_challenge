"""Extract the actual dynamic-SMEM ELF symbol and the shortest complete Hopper K-loop from every build."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import sys

here = Path(__file__).resolve().parent
root = Path(sys.argv[1])
resident = runpy.run_path(str(here.parent/"v199_prompt_expert_overlap/independent_ring.py"))["resident_bytes"]


def inspect(path, source):
    asm = path.read_text()
    opcodes = {}
    for line in asm.splitlines():
        item = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\* 0x", line)
        if item:
            opcodes[int(item[1],16)] = item[2]
    widths = Counter(map(int,re.findall(r"HGMMA\.64x(\d+)x16\.F32\.BF16",asm)))
    loops = {}
    for width, number in widths.items():
        ranges = []
        for stop, op in opcodes.items():
            jump = re.search(r"\bBRA(?:\.U)? (0x[0-9a-f]+)",op)
            if jump and int(jump[1],16) < stop:
                first = int(jump[1],16)
                seq = [v for i,v in opcodes.items() if first<=i<=stop]
                if sum("HGMMA." in x for x in seq) == sum(f"HGMMA.64x{width}x" in x for x in seq) == number:
                    ranges.append((len(seq),seq))
        size, seq = min(ranges, key=lambda t:t[0])
        loops[width] = dict(sites=size, hgmma=number, drains=sum("WARPGROUP.DEPBAR.LE" in x for x in seq),
                            local_loads=sum("LDL" in x for x in seq), local_stores=sum("STL" in x for x in seq))
    log = path.with_suffix(".log").read_text()
    cubin = path.with_suffix(".cubin")
    spills = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",log)
    return dict(source_sha256=sha256(source.read_bytes()).hexdigest(),cubin_sha256=sha256(cubin.read_bytes()).hexdigest(),
                registers=int(re.search(r"Used (\d+) registers",log)[1]), shared_bytes=resident(cubin), cubin_bytes=cubin.stat().st_size,
                stack_spill_store_load=list(map(int,spills.groups())),loops=loops,multicast=asm.count("UTMALDG.2D.MULTICAST"))


rows = {}
for name in ("", "check"):
    for path in sorted((root/name/"images").glob("*.sass")):
        rows[path.stem] = inspect(path, root/"images"/(path.stem+".cu"))
candidate = rows["WidePromptWeightPipeline"]
assert candidate["stack_spill_store_load"] == [0,0,0] and candidate["multicast"]
assert set(candidate["loops"]) == {32,128,224,256}
assert all(l["hgmma"]==16 and l["drains"]==8 and not l["local_stores"] and not l["local_loads"] for l in candidate["loops"].values())
assert candidate["shared_bytes"] <= 227*1024
assert rows["m256_n128_w256_a2"]["loops"][256]["local_loads"] > 0
print(json.dumps(dict(toolchain="NVCC 12.9.86, native SM90a with ptxas and cuobjdump", GPU=False, variants=rows),indent=2))

