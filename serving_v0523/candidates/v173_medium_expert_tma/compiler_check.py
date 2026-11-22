"""Read the two matched CUDA AOT build trees produced by compile_sm90.sh."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def label(name):
    for kind in ("tensor_projection", "streamed_projection", "projection_kernel",
                 "projection_channels", "plan_routes", "plan_rows", "projection"):
        if kind in name:
            break
    if kind.startswith("plan"):
        return kind + ("_padded" if "ILb1E" in name else "")
    gate, special = re.search(kind + r"ILb(\d)E(?:Lb(\d)E)?", name).groups()
    return kind + ("_special" if special == "1" else "") + ("_W13" if gate == "1" else "_W2")


def analyze(name, body, log):
    words = re.findall(r"/\* (0x[\da-f]{16}) \*/", body)
    binary = b"".join(int(word, 16).to_bytes(8, "little") for word in words)
    instructions = {
        int(pc, 16): op for pc, op in
        re.findall(r"^\s*/\*([\da-f]+)\*/\s+(.*?)\s*;\s*/\*", body, re.M)
    }
    assert len(binary) == len(instructions) * 16
    opcodes = {pc: op.split()[int(op.startswith("@"))].split(".")[0]
               for pc, op in instructions.items()}
    loops = []
    for pc, op in instructions.items():
        target = re.search(r"\bBRA (0x[0-9a-f]+)", op)
        if not target or int(target[1], 16) >= pc:
            continue
        start = int(target[1], 16)
        counts = Counter(code for addr, code in opcodes.items() if start <= addr <= pc)
        if counts["HMMA"]:
            loops.append(dict(start=start, end=pc, counts=counts))
    # Retain innermost arithmetic backedges, excluding surrounding grid/job loops.
    inner = [x for x in loops if not any(
        x is not y and x["start"] <= y["start"] and y["end"] <= x["end"] for y in loops)]
    resources = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s+"
        r"ptxas info\s+: Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?",
        log.split("Compiling entry function '" + name + "'", 1)[1].split("Compiling entry function", 1)[0],
    )
    assert resources
    keys = ("stack_bytes", "spill_store_bytes", "spill_load_bytes", "registers", "barriers", "shared_bytes")
    result = dict(zip(keys, [int(i or 0) for i in resources.groups()]))
    result.update(machine_sha256=sha(binary), sites=len(instructions),
                  inner_mma_loops=[dict(sites=sum(x["counts"].values()),
                      offset=hex(x["start"]), end=hex(x["end"]),
                      machine_sha256=sha(binary[x["start"]:x["end"]+16]),
                      interesting_ops={k: x["counts"][k] for k in
                          ("HMMA","HMUL2","LDG","LDS","STS","LDGSTS","BAR","DEPBAR","UTMALDG")
                          if x["counts"][k]}) for x in inner])
    return result


def main(base):
    report = dict(nvcc="12.9.86", hardware=False, build_inputs={}, builds={}, comparisons={})
    for role in ("prior", "new"):
        root = base / role
        report["build_inputs"][role] = {p.name: sha(p.read_bytes()) for p in
            sorted((root / "python/sglang/kernels/jit/csrc/muse_spark").glob("*.cuh"))}
        builds = report["builds"][role] = {}
        for module in ("tma", "medium"):
            for arch in ("sm_90a", "sm_90"):
                prefix = root / "out" / f"{module}.{arch}"
                files = {ext: Path(str(prefix) + "." + ext).read_bytes() for ext in ("o","log","sass")}
                sections = files["sass"].decode().split("Function : ")[1:]
                kernels = {}
                for body in sections:
                    name = body.splitlines()[0]
                    key = label(name)
                    assert key not in kernels
                    kernels[key] = analyze(name, body, files["log"].decode())
                builds[f"{module}.{arch}"] = dict(kernels=kernels,
                    file_sha256={ext: sha(v) for ext,v in files.items()},
                    wrapper_sha256=sha((root / "out" / (module + ".cu")).read_bytes()))
    for arch in ("sm_90", "sm_90a"):
        for module in ("tma","medium"):
            name = module + "." + arch
            old = report["builds"]["prior"][name]["kernels"]
            new = report["builds"]["new"][name]["kernels"]
            report["comparisons"][name] = {key: old[key]["machine_sha256"] == new[key]["machine_sha256"]
                                           for key in old}
            assert all(not k["spill_load_bytes"] and not k["spill_store_bytes"] and
                       not k["stack_bytes"] for k in (*old.values(), *new.values()))
        specialized = report["builds"]["new"]["tma." + arch]["kernels"]
        ordinary = report["builds"]["prior"]["medium." + arch]["kernels"]
        for gate in ("W13", "W2"):
            before = ordinary["projection_" + gate]["inner_mma_loops"]
            after = specialized["projection_special_" + gate]["inner_mma_loops"]
            assert [v["machine_sha256"] for v in before] == [v["machine_sha256"] for v in after]
        assert all(all(v.values()) for v in report["comparisons"].values())
    return report


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
