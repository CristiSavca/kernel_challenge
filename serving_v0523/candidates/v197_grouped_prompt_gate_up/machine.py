"""Check production CUDA output against independent, previously frozen SM90a machine images."""
import hashlib
import json
from pathlib import Path
import re
import sys

here = Path(__file__).resolve().parent
work = Path(sys.argv[1]).resolve()
src = work / "sglang/python/sglang/kernels/jit/csrc/muse_spark"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_bytes())
small = load(here.parent / "v191_decode_gate_up_epilogue/sm90a.json")["kernels"]
large = load(here.parent / "v177_grouped_gate_up_epilogue/sm90a.json")["kernels"]
report = {}
for mode, width in (("decode", 8), ("decode", 16), ("decode", 24), ("prompt", 176)):
    prefix = work / "final" / f"{mode}_{width}"
    actual = digest(prefix.with_suffix(".cubin"))
    old = (large["nvcc_gate_176"] if mode == "prompt" else small[str(width)]["both"])["cubin_sha256"]
    assert actual == old
    isa = prefix.with_suffix(".sass").read_text()
    logs = prefix.with_suffix(".log").read_text()
    sites = {op: isa.count(op) for op in ("HGMMA.64x", "UTMALDG.4D", "UTMALDG.2D", "WARPGROUP.DEPBAR")}
    assert sites["HGMMA.64x"] == 32 and sites["UTMALDG.4D"] > 0
    registers = int(re.search(r"Used (\d+) registers", logs)[1])
    frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", logs)
    report[prefix.name] = dict(cubin_sha256=actual, reused_machine_code_from="V177" if mode=="prompt" else "V191",
        rendered_source_sha256=digest(prefix.with_suffix(".cu")), sass_sha256=digest(prefix.with_suffix(".sass")),
        registers=registers, stack_spill_store_load_bytes=list(map(int, frame.groups())), sites=sites)
exports = sorted((work/"native/exports.txt").read_text().split())
assert exports == ["__tvm_ffi_"+s for s in ("configure","prepare","prepare_prompt","run","run_prompt")]
native = large["nvcc_8192_176"]
print(json.dumps(dict(nvcc="12.9.86",humming="0.1.12",arch="sm_90a",physical_gpu=False,
    header_sha256=digest(src/"decode_gate_up.cuh"),launcher_header_sha256=digest(src/"decode_gate_up_launch.cuh"),
    launcher_object_sha256=digest(work/"native/launcher.o"),llvm_symbols=exports,
    stock_prompt=dict(registers=native["registers"],stack=native["stack"],spill_stores=native["spill_stores"],spill_loads=native["spill_loads"]),
    kernels=report),indent=2))
