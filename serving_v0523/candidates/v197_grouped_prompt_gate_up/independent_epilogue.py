"""Run the standalone V195-lineage prompt epilogue against the independent native Humming memory oracle."""
import hashlib
import json
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

support = runpy.run_path(str(Path(__file__).resolve().parents[1] / "v177_grouped_gate_up_epilogue/independent_epilogue.py"))


def main(root):
    header = root / "python/sglang/kernels/jit/csrc/muse_spark/decode_gate_up.cuh"
    launcher = header.with_name("decode_gate_up_launch.cuh")
    device = header.read_text()
    writer = device[device.index("template <> class EpilogueGmemWriter"):]
    writer, count = re.subn(r'asm\("mul\.rn\.f32[^;]+;[^\n]+\);',
                            "product = __bfloat162float(value) * scale;", writer)
    assert count == 1
    loader = support["function"](launcher.read_text(), "static bool bank_map(")
    main = support["MAIN"].replace("assert(bank_map(", "assert(bank_map<true>(")
    main = main.replace("start=13, rows=M+start+2", "start=START, rows=M+start+TAIL")
    main = main.replace("ids[i]=int(i%4)", "ids[i]=(i < start ? 1 : (i < start+M ? EXPERT : 3))")
    assert "scales[ids[start]]=factor" in main
    assert "k_tile * (kMuseGateK / 16)" in device and "row += kMuseGateK / 16" in device
    cpp = support["CPP"] + "template<bool Prompt>\n" + loader + "\n" + writer + main
    receipts = []
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch)
        src = path / "oracle.cpp"
        src.write_text(cpp)
        for inner, pad, start, tail, expert in ((128,7,0,0,0),(384,1,183,3,2),(4096,3,13,2,3),(4096,6,177,0,1)):
            cmd = ["c++", "-std=c++17", "-O2", "-fno-strict-aliasing", "-DMUSE_GATE_PROMPT", "-DMUSE_GATE_M=176",
                   f"-DSTART={start}", f"-DTAIL={tail}", f"-DEXPERT={expert}", f"-DHUMMING_SHAPE_N={2*inner}",
                   f"-DPAD={pad}", str(src), "-o", str(path / "oracle")]
            subprocess.run(cmd, check=True)
            words = list(map(int, subprocess.check_output([str(path / "oracle")],text=True).split()))
            receipts.append(dict(intermediate=inner, first_row=start, trailing_rows=tail, expert=expert,
                output_values=words[0], untouched_words=words[1], packed_addresses=words[2],
                distinct_gate_patterns=words[3], distinct_up_patterns=words[4]))
    assert any(r["distinct_gate_patterns"] == r["distinct_up_patterns"] == 65536 for r in receipts)
    return dict(device_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(), cases=receipts,
                scope="C++ source execution with CPU expf, FP32 multiplication and software BF16 RN; descriptor maps and final shared writer independently checked, no hardware CUDA execution.")


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
