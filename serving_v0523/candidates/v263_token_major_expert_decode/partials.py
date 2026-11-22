"""Check every physical source row, including lanes beyond an expert's live suffix."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1])
PREVIOUS = HERE.parent / "v213_live_decode_expert_reduction"

source = (PREVIOUS / "partials.cpp").read_text()
source = source.replace("SharedStorage& smem;", "SharedStorage& smem; struct { unsigned top_k = 0; } params;")
source = source.replace("void exercise(unsigned rows,unsigned seed)", "void exercise(unsigned rows,unsigned seed, bool direct=false)")
source = source.replace("fresh.seek(0,0,0,rows+33,33);",
                        "cb.params.top_k = direct ? (3u << 16) + 6 : 6; fresh.seek(0,0,0,rows+33,33); if(direct) rows=8;")
source = source.replace("exercise<8,2>(i,seed);", "exercise<8,2>(i,seed,true);exercise<8,2>(i,seed);")
with tempfile.TemporaryDirectory() as td:
    test = Path(td)
    paths = {"native":ROOT/"humming/include/humming/epilogue/smem_reducer.cuh",
             "submitted":ROOT/"sglang/python/sglang/kernels/jit/csrc/muse_spark/live_decode_reduce.cuh"}
    for name,p in paths.items():
        body = p.read_text().replace("#include <humming/utils/all.cuh>","")
        body = body.replace("void reduce(uint32_t *regs_ptr)","Task reduce(uint32_t *regs_ptr)")
        body = body.replace("void call(uint32_t* words)","Task call(uint32_t* words)")
        (test/(name+".h")).write_text(body.replace("ctx.sync_math_threads();","co_await std::suspend_always{};"))
    (test/"case.cpp").write_text(source)
    exe = test/"test"
    subprocess.run(["c++","-std=c++20","-O2","-fno-strict-aliasing","-ffp-contract=off","-Wno-deprecated-enum-enum-conversion",
                    str(test/"case.cpp"),"-o",str(exe)],check=True)
    result = json.loads(subprocess.check_output([exe]))
    result["source_sha256"] = {k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    print(json.dumps(result,indent=2))
