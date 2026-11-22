"""Check the exact native W13 generator, each ablation, and all outstanding SM90 register writes."""
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
scratch = Path(sys.argv[1])
old = here.parent / "v189_decode_grouped_mma/isa.py"
# Same machine-level bounds, operand decoding and overlapping-register test as W2.
exec(compile(old.read_text().split("out={}")[0].replace('"FSEL"}','"FSEL","S2R"}'), str(old), "exec"))
stage = scratch / "sglang/python/sglang/kernels/jit/csrc/muse_spark"
gate = (stage / "decode_gate_up.cuh").read_bytes()
pair = (stage / "grouped_decode_mma_pair.cuh").read_bytes()
out = {}
for width in (8, 16, 24):
    name = f"8192_{width}"
    src, expr = (scratch/"original"/(name+".cu")).read_bytes().split(b"\nauto ptr = ")
    fused = src + f"\n#define MUSE_GATE_M {width}\n".encode() + gate
    for mode,payload in (("fusion",fused),("both",fused + b"\n#define MUSE_OVERLAP_CONTEXT MuseGateCtx\n" + pair)):
        assert (scratch/mode/(name+".cu")).read_bytes() == payload + b"\nauto ptr = " + expr
    data = {k:parse(scratch/k/name) for k in ("original","fusion","both")}
    for k, d in data.items():
        assert d["group_sizes"] == ([4,2,2]*4 if k=="both" else [2]*16)
        assert d["registers"]==96 and d["stack_spill_store_load_bytes"]==(0,0,0)
        assert d["tensor_2d"] > 0 and d["tensor_4d"]==(0 if k=="original" else 14)
        assert d["tensor_3d"]==(14 if k=="original" else 0)
    out[width]=data
print(json.dumps(dict(nvcc="12.9.86",humming="0.1.12",arch="sm_90a",physical_gpu=False,
      header_sha256=sha(gate),pair_header_sha256=sha(pair),
      launcher_header_sha256=sha((stage/"decode_gate_up_launch.cuh").read_bytes()),
      launcher_object_sha256=sha((scratch/"launcher.o").read_bytes()),kernels=out),indent=2))
