"""Reconstruct compiled-source identity and compare the native and fused SASS."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

blobs, stage = map(Path, sys.argv[1:])
sha = lambda data: hashlib.sha256(data).hexdigest()
device = stage / "python/sglang/kernels/jit/csrc/muse_spark/grouped_gate_up.cuh"
launcher = device.with_name("grouped_gate_up_launch.cuh")
result = {}
parent = json.loads((Path(__file__).resolve().parent.parent/"v169_grouped_expert_tile_cohorts/sm90a.json").read_bytes())
for width in (136, 152, 176):
    native = (blobs/f"8192_{width}.cu").read_bytes()
    body, expr = native.split(b"auto ptr = ")
    fused = body + f"\n#define MUSE_GATE_M {width}\n".encode() + device.read_bytes() + b"\nauto ptr = " + expr
    assert (blobs/f"gate_{width}.cu").read_bytes() == fused
    for compiler in ("nvcc", "nvrtc"):
        for tag in ("8192","gate"):
            base = f"{tag}_{width}"
            suffix = ".nvrtc" if compiler == "nvrtc" else ""
            cubin, assembly, log = ((blobs/(base+suffix+"."+ext)).read_bytes() for ext in ("cubin","sass","log"))
            code = assembly.decode()
            ops = Counter(re.findall(r"\b(HGMMA\.\d+x\d+x\d+\.F32\.BF16|UTMALDG\.[234]D)\b",code))
            hgmma = f"HGMMA.64x{width}x16.F32.BF16"
            assert ops["UTMALDG.2D"] == 7
            if compiler == "nvcc":
                assert ops[hgmma] == 32
            else:
                assert not ops[hgmma] and "BPT.TRAP" in code
            assert ops["UTMALDG.4D" if tag == "gate" else "UTMALDG.3D"] == 14
            resource = {}
            if compiler == "nvcc":
                mem = re.search(rb"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
                regs = re.search(rb"Used (\d+) registers, used (\d+) barriers", log)
                assert regs and mem
                resource = dict(zip(("registers","barriers","stack","spill_stores","spill_loads"), map(int,(*regs.groups(),*mem.groups()))))
                if tag == "8192":
                    assert parent["kernels"][base]["file_sha256"]["cubin"] == sha(cubin)
            result[f"{compiler}_{base}"] = dict(source_sha256=sha(native if tag=="8192" else fused),
                cubin_sha256=sha(cubin), sass_sha256=sha(assembly), cubin_bytes=len(cubin), sites=ops, trap_sites=code.count("BPT.TRAP"), **resource)
print(json.dumps(dict(nvcc="12.9.86",nvrtc="12.9.86",humming="0.1.12",arch="sm_90a",real_gpu=False,
    native_launcher_object_sha256=sha((blobs.parent/"launcher.o").read_bytes()),
    device_header_sha256=sha(device.read_bytes()),launcher_header_sha256=sha(launcher.read_bytes()),
    kernels=result),indent=2))
