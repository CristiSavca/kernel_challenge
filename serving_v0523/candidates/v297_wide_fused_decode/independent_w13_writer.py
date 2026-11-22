"""Drive the production compact epilogue with independently scattered native RF coordinates."""
import argparse
from array import array
import ast
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import tempfile

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("stage", type=Path)
parser.add_argument("--native-only", action="store_true")
args = parser.parse_args()
kernel = args.stage / "python/sglang/kernels/jit/csrc/muse_spark/decode_gate_up.cuh"
original = kernel.read_text()
writer = original[original.index("template <> class EpilogueGmemWriter"):]
writer, count = re.subn(r'asm\("mul\.rn\.f32[^;]+;[^\n]+\);', "product = __bfloat162float(value) * scale;", writer)
assert count == 1
fixture = next(here.parent.glob("v191_*/independent_epilogue.py"))
headers = runpy.run_path(str(fixture))["CPP"].replace("reduce[MUSE_GATE_M*32]", "reduce[MUSE_GATE_M*TEST_N/8]")
scatter = next(here.parent.glob("v275_*/independent_scatter.py"))
node = next(n for n in ast.parse(scatter.read_bytes()).body if getattr(n, "name", None) == "pack")
functions = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(scatter), "exec"), functions)
main = (here / "independent_w13_writer.cpp").read_text()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

cases = []
with tempfile.TemporaryDirectory(prefix="okc-wide-writer-") as tmp:
    root = Path(tmp)
    source, exe, layout = (root/name for name in ("test.cpp","test","layout.bin"))
    geometry = re.search(r"#ifndef MUSE_GATE_BANDS.*?constexpr uint32_t kMuseGateBands[^;]+;", original, re.DOTALL)
    assert geometry
    source.write_text(headers + "\n" + geometry.group(0) + "\n" + writer + "\n" + main)
    for m, block, pad in [(8,256,1),(16,256,6),(24,256,3),(16,512,7),(24,512,1)]:
        if args.native_only and block != 256:
            continue
        warp = block // 8
        mapping = array("i", functions["pack"](m,block,warp,pad))
        assert mapping.itemsize == 4
        layout.write_bytes(mapping.tobytes())
        defines = [f"-DMUSE_GATE_M={m}", f"-DPAD={pad}", "-DHUMMING_SHAPE_N=8192", f"-DTEST_N={block}",
                   f"-DMUSE_GATE_BANDS={block//128}"]
        subprocess.run(["c++","-std=c++17","-O2","-fno-strict-aliasing",*defines,str(source),"-o",str(exe)],check=True)
        def run():
            return list(map(int,subprocess.check_output([exe,layout],text=True).split()))
        words, guards, errors, gates, ups = run()
        assert errors == 0, (m,block,errors)
        # Move the second half of native RF values onto different up channels.
        bad = array("i",(v if v % block < block//2 else v//block*block+block//2+(v%block+64)%(block//2)
                        for v in mapping))
        layout.write_bytes(bad.tobytes())
        aliased = run()
        assert aliased[2] > 1000
        cases.append(dict(tile_rows=m,block_channels=block,warp_channels=warp,shared_swizzle=pad,
            checked_outputs=words,checked_guard_words=guards,gate_patterns=gates,up_patterns=ups,
            shifted_up_channel_errors=aliased[2]))
print(json.dumps(dict(mode="native" if args.native_only else "combined",
    inputs={p.name:sha(p) for p in (kernel,fixture,scatter,here/"independent_w13_writer.cpp")},
    cases=cases,physical_gpu=False,
    scope="Production compact CUDA writer compiled as host C++. Native WGMMA register labels pass through the pinned swizzled shared-memory mapping. CUDA expf and round-to-nearest BF16 arithmetic are modeled on the host."),indent=2))
