"""Instrument the native list-building warp and both projection grids on reused buffers."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def section(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


here = Path(__file__).resolve().parent
cu = Path(sys.argv[1])/"python/sglang/kernels/jit/csrc/muse_spark"
medium = (cu/"expert_medium_route_pair.cuh").read_text()
tma = (cu/"expert_tma_route_pair.cuh").read_text()
base = (cu/"expert_route_pair.cuh").read_text()
template = (here.parent/"v141_adaptive_expert_projection/independent_plan.cpp").read_text()
plan = section(medium, "template <bool PadSparse = false>", "\nstruct Inputs")
plan = plan.replace("__global__ void plan_rows", "Task plan_rows")
project = section(medium, "template <bool Gate, bool SkipSparse = false>", "\n}  // namespace")
project = project.replace("__global__ __launch_bounds__(32 * kWarps) ", "").replace("__grid_constant__ ", "")
tma_gate = section(tma, "    const uint32_t rows = p.plan[size_t(blockIdx.y) * stride];", "\n  }")
assert "if (rows == 0 || rows > kRows) return;" in tma_gate
assert "ExpertList ? p.plan + size_t(j) * stride : p.plan + 1 + j * (kRows + 1)" in tma
finish = section(base, "SGL_DEVICE void finish_channels", "\ntemplate")
output_row = next(line.strip() for line in finish.splitlines() if "const int32_t out =" in line)
output_column = next(line.strip() for line in finish.splitlines() if "const uint32_t n =" in line)
assert "if (out >= 0)" in finish and "outputs[size_t(out) * p.output_width + n]" in finish
sparse = """
uint64_t sparse_coordinates = 0;
void sparse_projection(const Projection& p, unsigned stride){
    NATIVE_GATE
    const int32_t* job = p.plan + size_t(blockIdx.y) * stride;
    const uint32_t column = blockIdx.x*128, lane = threadIdx.x%32, warp = threadIdx.x/32;
    const uint32_t band = warp/4, part = (warp + lane/8)%4;
    for(uint32_t i=0; i<4; ++i) {
        NATIVE_ROW
        if (out >= 0) {
            assert(unsigned(out) < p.count && p.ids[out] == int(blockIdx.y));
            NATIVE_COLUMN
            assert(n < p.output_width && p.written[size_t(out)*p.output_width+n]++ == 0);
            ++sparse_coordinates;
        }
    }
}
""".replace("NATIVE_GATE", tma_gate).replace("NATIVE_ROW", output_row).replace("NATIVE_COLUMN", output_column)
row = next(line.strip() for line in medium.splitlines() if "const uint32_t row = 8 * g + 2 * (lane % 4)" in line)
column = next(line.strip() for line in medium.splitlines() if "const uint32_t n = column + part" in line)
template = template.replace("PLANNER_BODY", "constexpr unsigned kRows=8;\n"+plan)
template = template.replace("DISPATCH_BODY", project+"\n"+sparse)
template = template.replace("ROW_WRITE_EXPRESSION", row).replace("COLUMN_WRITE_EXPRESSION", column)
template = template.replace("=plan_rows(ids.", "=plan_rows<true>(ids.")
mark = "        assert(std::equal(want.begin(),want.end(),plan+ex*stride+1));"
template = template.replace(mark, mark+"""
        for(unsigned k=want.size(); k<std::min<unsigned>(s,kRows); ++k)assert(plan[ex*stride+1+k] == -1);""")
mark = "          for(threadIdx.x=0;threadIdx.x<64;++threadIdx.x)"
template = template.replace(mark, """          if(s>=kRows)for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)sparse_projection(p,stride);
"""+mark)
mark = "            if(gate)projection<true>(p,stride);else projection<false>(p,stride);"
template = template.replace(mark,"""            if(s<kRows){
              if(gate)projection<true>(p,stride);else projection<false>(p,stride);
            }else{
              if(gate)projection<true,true>(p,stride);else projection<false,true>(p,stride);
            }""")
template = template.replace('<<checks', '<<checks+sparse_coordinates')
template = template.replace('<<",\\"one_two_four_fragment_lane_calls', '<<",\\"sparse_coordinates\\":"<<sparse_coordinates<<",\\"one_two_four_fragment_lane_calls')
with tempfile.TemporaryDirectory() as temp:
    cpp = Path(temp)/"plan.cpp"
    binary = Path(temp)/"plan"
    cpp.write_text(template)
    subprocess.run(["clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-O2", cpp, "-o", binary], check=True)
    result = json.loads(subprocess.check_output([binary]))
assert result["sparse_coordinates"] and result["sparse_coordinates"] < result["unique_gate_down_output_coordinates"]
print(json.dumps({**result, "source_sha256": {p: hashlib.sha256((cu/p).read_bytes()).hexdigest()
                 for p in ("expert_medium_route_pair.cuh", "expert_tma_route_pair.cuh", "expert_route_pair.cuh")},
                  "physical_H200": False}, indent=2))
