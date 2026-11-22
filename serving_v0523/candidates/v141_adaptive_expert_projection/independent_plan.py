"""Execute the production expert planner and launch-loop bodies using CPU warp intrinsics."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
path = Path(sys.argv[1])/"python/sglang/kernels/jit/csrc/muse_spark/expert_medium_route_pair.cuh"
source = path.read_text()
start = source.index("__global__ void plan_rows(")
stop = source.index("\n}",start)+2
plan = source[start:stop].replace("__global__ void plan_rows", "Task plan_rows")
start = source.index("template <bool Gate>\n__global__")
stop = source.index("\n}",start)+2
dispatch = source[start:stop].replace("__global__ __launch_bounds__(32 * kWarps) ", "").replace("__grid_constant__ ", "")
row_write = next(s.strip() for s in source.splitlines() if "const uint32_t row = 8 * g + 2 * (lane % 4)" in s)
column_write = next(s.strip() for s in source.splitlines() if "const uint32_t n = column + part" in s)
assert "outputs[size_t(slots[row]) * p.output_width + n]" in source
assert "p.inputs + size_t(slots[row] / p.top_k) * p.input_width" in source
assert "uint32_t(inner), uint32_t(top.unwrap())" in source
assert "uint32_t(inner), uint32_t(hidden), 1" in source
with tempfile.TemporaryDirectory(prefix="okc-v141-plan-") as tmp:
    d=Path(tmp);cpp=d/"run.cpp";binary=d/"run"
    template=(here/"independent_plan.cpp").read_text()
    cpp.write_text(template.replace("PLANNER_BODY",plan).replace("DISPATCH_BODY",dispatch)
                   .replace("ROW_WRITE_EXPRESSION",row_write).replace("COLUMN_WRITE_EXPRESSION",column_write))
    subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",str(cpp),"-o",str(binary)],check=True)
    result=json.loads(subprocess.check_output([str(binary)]))
print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**result),indent=2))
