"""Verify actual W2 disassembly, physical register lifetimes and four-stage progress."""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shared_size(cubin):
    blob = cubin.read_bytes()
    start = struct.unpack_from("<Q", blob, 40)[0]
    step, count = struct.unpack_from("<HH", blob, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", blob, start + i*step) for i in range(count)]
    for s in sections:
        if s[1] != 2:
            continue
        strings = sections[s[6]]
        for off in range(s[4], s[4]+s[5], s[9]):
            n, _, _, section, address, _ = struct.unpack_from("<IBBHQQ", blob, off)
            at = strings[4]+n
            if blob[at:blob.index(0, at)] == b"SMEM_SIZE":
                sh = sections[section]
                return struct.unpack_from("<I", blob, sh[4]+address-sh[3])[0]
    raise ValueError("missing Humming shared-memory ELF symbol")


def machine(stage, width, variant):
    path = stage / variant / f"4096_{width}.sass"
    cubin = path.with_suffix(".cubin")
    holding = set()
    groups, releases, dests = [], [], []
    pending, checked, hmma = 0, 0, 0
    for line in path.read_text().splitlines():
        match = re.search(r"/\*([0-9a-f]+)\*/\s*(?:@[!\w]+\s+)?([A-Z][.A-Za-z0-9_]+)\s+(.*?);", line)
        if not match:
            continue
        pc, op, operands = match.groups()
        if op.startswith("HGMMA."):
            assert op == f"HGMMA.64x{width}x16.F32.BF16"
            dst, source, desc, acc, *_ = (item.strip() for item in operands.split(","))
            assert dst == acc and desc.startswith("gdesc[")
            # WGMMA m64nMx16 has a 4x32b register-source fragment per lane.
            word = int(source[1:])
            holding.update(range(word,word+4))
            dests.append(dst)
            hmma += 1
            pending += 1
        elif op.startswith("WARPGROUP.DEPBAR") and hmma:
            assert "0x0" in operands
            groups.append(pending)
            holding.clear()
            pending = 0
        elif op.startswith("SYNCS.ARRIVE") and hmma:
            assert not pending and not holding
            releases.append(hmma)
        elif holding:
            target = re.match(r"R(\d+)(?:,|\.)",operands)
            if target:
                # These are the only scalar/vector destination instruction families in the interval.
                assert op.startswith(("LDS", "IMAD", "PRMT", "LOP3", "SHF","HMUL2","VIADD","SEL")), (pc,op)
                base = int(target[1])
                words = 4 if op.startswith("LDS.128") else 2 if op.startswith("LDS.64") else 1
                assert holding.isdisjoint(range(base,base+words)), (pc,op,base,holding)
                checked += words
    assert hmma==32 and dests==dests[:2]*16 and dests[0] != dests[1]
    new = variant == "paired_twice"
    assert groups == ([4,4] if new else [4,2,2])*4
    assert releases == ([8,16,24,32,32] if new else [6,14,22,30,32])
    log = cubin.with_suffix(".log").read_text()
    assert "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" in log
    return dict(cubin=sha(cubin), sass=sha(path), register_words_checked=checked,
                groups=groups, arrivals_after_mma=releases, registers=int(re.search(r"Used (\d+) registers", log)[1]),
                shared_bytes=shared_size(cubin))


def ring(n):
    # The 128 producer lanes may lag one or all four consumer warpgroups. Ordering symmetric
    # workers lets us enumerate every distinct schedule of publication and stage retirement.
    initial = (0,(0,0,0,0))
    seen = {initial}
    todo = deque([initial])
    terminal = 0
    transitions = 0
    while todo:
        published_after_release, workers = todo.pop()
        ready = min(n,published_after_release+3)
        new = set()
        if published_after_release<n and min(workers)>published_after_release:
            writer = published_after_release+3
            if writer<n and writer>=4:
                assert min(workers)>writer-4
            new.add((published_after_release+1,workers))
        for idx, current in enumerate(workers):
            if current>=ready or current+1<n and current+1>=ready:
                continue
            assert current>=ready-4
            values = list(workers)
            values[idx] += 1
            new.add((published_after_release,tuple(sorted(values))))
        if not new:
            assert workers == (n,)*4 and published_after_release==n
            terminal += 1
        transitions += len(new)
        for state in new-seen:
            seen.add(state)
            todo.append(state)
    assert terminal==1
    return len(seen),transitions


def archived_images():
    evidence = json.loads(Path(__file__).with_name("sm90a.json").read_text())
    assert evidence["arch"] == "sm_90a" and evidence["physical_gpu"] is False
    result = {}
    for width in (8,16,24):
        record=evidence["kernels"][f"4096_{width}"]
        item={}
        for new, tag in ((False,"original"),(True,"candidate")):
            k=record[tag]
            assert k["group_sizes"]==([4,4] if new else [4,2,2])*4
            assert k["mma"]==32 and k["registers"]==96 and k["stack_spill_store_load_bytes"]==[0,0,0]
            item[tag]={key:k[key] for key in ("cubin_sha256","sass_sha256","registers","group_sizes","checked_non_wgmma_register_writes_during_outstanding")}
            # The independent direct disassembly pass observed every empty group and the
            # index of the stage-arrival relative to issued HGMMA. The compact exporter
            # retains hashes and instruction counts but does not retain the raw SASS.
            item[tag]["independently_observed_arrivals_after_mma"]=([8,16,24,32,32] if new else [6,14,22,30,32])
        result[width]=item
    return dict(mode="archived_image_hashes_and_direct_disassembly_observations",
                header_sha256=evidence["header_sha256"],widths=result)


def direct_images(stage):
    header = stage / "sglang/python/sglang/kernels/jit/csrc/muse_spark/grouped_decode_mma_pair.cuh"
    adapter = stage / "sglang/python/sglang/srt/layers/muse_spark/decode_mma_pair.py"
    variants = {}
    for width in (8,16,24):
        parent, selected = (machine(stage,width,tag) for tag in ("paired","paired_twice"))
        assert parent["shared_bytes"] == selected["shared_bytes"]
        variants[width] = dict(parent=parent,selected=selected)
        text = (stage/"paired_twice"/f"4096_{width}.cu").read_text()
        assert "if (warp_iter_id + 1 < Ctx::kWarpIters || slice_iters > 1) s2r_pipe.load_stage_iter" in text
        run = text.index("mma.run(stage_id, warp_iter_id);")
        wait = text.index("consumer.wait_stage((stage_id + 1)",run)
        release = text.index("if (warp_iter_id + 1 == Ctx::kWarpIters) consumer.arrive(stage_id);",wait)
        assert run<wait<release<text.index("mma.transform_b(",release)
    return dict(mode="full_raw_cubin_and_sass",header_sha256=sha(header),adapter_sha256=sha(adapter),widths=variants)


evidence = direct_images(Path(sys.argv[1])) if len(sys.argv)>1 else archived_images()
ns = list(range(1,34))+[64,128]
cases = [(n,*ring(n)) for n in ns]
print(json.dumps(dict(**evidence,slice_counts=ns,uniquely_reachable_states=sum(c[1] for c in cases),
                      state_transitions=sum(c[2] for c in cases),terminal_states=len(cases),physical_gpu=False), indent=2))
