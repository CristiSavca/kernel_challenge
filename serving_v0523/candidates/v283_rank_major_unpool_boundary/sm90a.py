"""Compare compiled Hopper norm/residual implementations in the compile.sh output directory."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pattern = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/\s*\n\s*/\* (0x[0-9a-f]+) \*/")
arithmetic = ("FADD", "FFMA", "FMUL", "F2", "HADD", "HFMA", "HMUL", "MUFU", "SHFL", "REDUX")
def parse(stem):
    sass = stem.with_suffix(".sass").read_text()
    image = {}
    for piece in sass.split("Function : ")[1:]:
        name, body = piece.split("\n",1)
        signature = tuple(re.findall(r"Lb([01])E", name))
        assert "norm_residual_kernel" in name and signature not in image
        ops = []
        words = []
        for _, ins, lo, hi in pattern.findall(body):
            op = ins.split()[0]
            if op.startswith("@"):
                op = ins.split()[1]
            if op == "HFMA2.MMA" and re.search(r", -RZ, RZ, [-.\d]+, [-.\d]+$", ins):
                op = "CONSTANT_MATERIALIZATION"
            ops.append(op)
            words.append(lo + hi)
        assert ops and "EXIT" in ops and "SHFL.BFLY" in ops
        image[signature] = {"ops":ops, "sha256":sha256("".join(words).encode()).hexdigest()}
    assert len(image) == 16
    log = stem.with_suffix(".log").read_text()
    spills = re.findall(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    assert len(spills) == 16 and all(int(n) == 0 for row in spills for n in row)
    regs = list(map(int, re.findall(r"Used (\d+) registers", log)))
    assert len(regs) == 16 and set(re.findall(r" T (__tvm_ffi_run_\w+)", stem.with_suffix(".symbols").read_text())) == {
        "__tvm_ffi_run_std", "__tvm_ffi_run_std_router", "__tvm_ffi_run_scale", "__tvm_ffi_run_scale_router"}
    return image, regs

rows = []
for threads in (256,512):
    parent, _ = parse(root/"old"/f"r1_t{threads}")
    for rank in (1,4,8):
        path = root/"new"/f"r{rank}_t{threads}"
        child, regs = parse(path)
        assert parent.keys() == child.keys()
        exact = 0
        equal_math = 0
        added = []
        math_delta = Counter()
        global_loads = []
        for key, source in parent.items():
            dest = child[key]
            exact += source["sha256"] == dest["sha256"]
            old, new = (Counter(op for op in item["ops"] if op.startswith(arithmetic)) for item in (source,dest))
            equal_math += old == new
            for k in set(old) | set(new):
                if new[k] != old[k]:
                    math_delta[k] += abs(old[k] - new[k])
            added.append(len(dest["ops"]) - len(source["ops"]))
            global_loads.append(sum(x.startswith("LDG") for x in dest["ops"]) - sum(x.startswith("LDG") for x in source["ops"]))
        if rank == 1:
            assert exact == len(parent)
        assert equal_math == len(parent)
        rows.append(dict(threads=threads, column_ranks=rank, kernels=len(child), byte_identical_to_parent=exact,
            identical_arithmetic_opcode_counts=equal_math, absolute_count_differences=dict(math_delta),
            instruction_sites_delta_range=[min(added),max(added)],global_load_sites_delta_range=[min(global_loads),max(global_loads)],
            registers=[min(regs),max(regs)], spill_bytes=0,
            wrapper_sha256=sha256(path.with_suffix(".cu").read_bytes()).hexdigest(),
            object_sha256=sha256(path.with_suffix(".o").read_bytes()).hexdigest()))
print(json.dumps(dict(target="sm_90a", toolchain="CUDA 12.9", width=8192,
    candidate_header_sha256=sha256((root/"new/norm_residual.cuh").read_bytes()).hexdigest(),
    parent_header_sha256=sha256((root/"old/norm_residual.cuh").read_bytes()).hexdigest(),
    translation_units=8,kernel_instances=128,FFI_wrappers_per_TU=4,configurations=rows,
    scope="Static machine code emitted from production JIT wrappers, including all weight and gate variants. Counts are instruction sites, not elapsed cycles."),indent=2))
