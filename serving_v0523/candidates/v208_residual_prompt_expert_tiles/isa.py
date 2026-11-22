"""Check the final Humming shader's innermost SM90a K-loop, source hazard and resource declarations."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

root = Path(sys.argv[1])
here = Path(__file__).resolve().parent
shared_size = runpy.run_path(str(here.parent / "v199_prompt_expert_overlap/independent_ring.py"))["resident_bytes"]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect(folder, phase, widths):
    path = root / folder / (phase + ".sass")
    binary = path.with_suffix(".cubin")
    text = path.read_text()
    lines = {int(m[1], 16): m[2] for l in text.splitlines()
             if (m := re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\* 0x", l))}
    loops = {}
    for width in widths:
        candidates = []
        for end, op in lines.items():
            branch = re.search(r"\bBRA(?:\.U)? (0x[0-9a-f]+)", op)
            if not branch or (begin := int(branch[1], 16)) >= end:
                continue
            block = [v for i,v in lines.items() if begin <= i <= end]
            if sum("HGMMA." in op for op in block) == sum(f"HGMMA.64x{width}x16" in op for op in block) == 32:
                candidates.append((len(block),begin,end,block))
        n, start, stop, block = min(candidates)
        legacy = folder == "probe"
        pending, held, release, groups = 0, set(), [], []
        protected, total = 0, 0
        for line in block:
            if "HGMMA." in line:
                match = re.search(r"HGMMA\.[^ ]+ R(\d+), R(\d+), gdesc\[", line)
                assert match, line
                word = int(match[2])
                held.update(range(word,word+4))
                pending += 1
                total += 1
            elif "WARPGROUP.DEPBAR.LE" in line:
                groups.append(pending)
                pending = 0
                held.clear()
            elif "SYNCS.ARRIVE" in line:
                assert pending == 0
                release.append(total)
            elif held and (match := re.search(r"(?:^| )([A-Z]+[.A-Z0-9_]*) R(\d+)(?:,|\.)", line)):
                op, word = match[1], int(match[2])
                if op.startswith(("ST", "BRA", "WARPGROUP", "SYNCS", "NOP")):
                    continue
                count = 4 if op.startswith(("LDS.128","LDL.128")) else 2 if op.startswith(("LDS.64","LDL.64")) else 1
                assert held.isdisjoint(range(word, word+count)), (folder,phase,line,held)
                protected += count
        if not legacy:
            assert groups == ([2]*16 if phase=="w13" else [4]*8), groups
            assert release[:4] == ([6,14,22,30] if phase=="w13" else [8,16,24,32]), release
        loops[width] = dict(sites=n, begin=hex(start), end=hex(stop), matrix_ops=total, drains=len(groups),
            raw_stage_release_after=release, protected_register_destination_words=protected,
            local_loads=sum("LDL" in op for op in block),local_stores=sum("STL" in op for op in block))
    log = path.with_suffix(".log").read_text()
    stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    source_file = root/"images"/({"w13":"GroupedResidualPromptGateUpKernel_176.cu","w2":"ResidualPromptWeightPipeline_176.cu"}[phase])
    if folder == "v2":
        assert source_file.read_bytes() == path.with_suffix(".cu").read_bytes()
        check=runpy.run_path(str(root/"sglang/python/sglang/srt/layers/muse_spark/residual_prompt_mma.py"))["has_residual_isa"]
        expected=16 if phase=="w13" else 8
        assert check(text, expected)
        for width in widths:
            assert not check(text.replace(f"HGMMA.64x{width}x16.F32.BF16","BAD",1),expected)
        assert not check(text.replace("WARPGROUP.DEPBAR.LE","BAD",1),expected)
        assert all(not t["local_loads"] and not t["local_stores"] for t in loops.values())
    return dict(source_sha256=digest(path.with_suffix(".cu")),cubin_sha256=digest(binary),sass_sha256=digest(path),
                text_sites=len(lines),cubin_bytes=binary.stat().st_size,shared_bytes=shared_size(binary),
                registers=int(re.search(r"Used (\d+) registers",log)[1]),stack_store_load_bytes=list(map(int,stack.groups())),
                loops=loops)

if __name__=="__main__":
    args = [("V206","base",[176]),("V208","v2",[32,96,176])]
    if (root/"v3").is_dir():args.append(("rejected_four_width","v3",[32,96,160,176]))
    result={name:{phase:inspect(folder,phase,widths) for phase in ("w13","w2")} for name,folder,widths in args}
    for phase in ("w13","w2"):
        a,b=result["V206"][phase],result["V208"][phase]
        assert a["shared_bytes"]==b["shared_bytes"] and a["registers"]==b["registers"]
        assert a["loops"][176]["sites"]==b["loops"][176]["sites"]
    print(json.dumps(dict(toolchain="NVCC 12.9.86 SM90a, actual Humming 0.1.12 production templates",gpu=False,results=result),indent=2))
