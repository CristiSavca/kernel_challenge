"""Render all matched V181/V183 CUDA sources for isa.py. Input directory contains humming and sglang."""
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
here = Path(__file__).resolve().parent
gen = here.parent / "v169_grouped_expert_tile_cohorts/vendor_codegen.py"
subprocess.run([sys.executable, str(gen), str(root/"humming"), str(root/"sglang"), str(root/"build")], check=True)
headers = root / "sglang/python/sglang/kernels/jit/csrc/muse_spark"
for directory in ("inspect","parent","final"):
    (root/directory).mkdir(exist_ok=True)
for width in (136,152,176):
    for n in (4096,8192):
        name=f"{n}_{width}.cu"
        source, expression=(root/"build"/name).read_bytes().split(b"auto ptr = ")
        if n == 8192:
            source += f"\n#define MUSE_GATE_M {width}\n".encode() + (headers/"grouped_gate_up.cuh").read_bytes()
        baseline=root/("inspect" if n==4096 else "parent")/name
        if n==4096:
            baseline.write_bytes((root/"build"/name).read_bytes())
        else:
            baseline.write_bytes(source+b"\nauto ptr = "+expression)
        if n==4096:
            source=source.removesuffix(b"\n")  # Strip vendor_codegen’s synthetic symbol separator.
        source+=f"\n#define MUSE_OVERLAP_M {width}\n".encode()+(headers/"grouped_mma_pair.cuh").read_bytes()
        (root/"final"/name).write_bytes(source+b"\nauto ptr = "+expression)
