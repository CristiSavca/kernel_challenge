"""Trace actual packed gate/up source coordinates and descriptor boxes at both tile widths."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
src = Path(sys.argv[1]) / "python/sglang/kernels/jit/csrc/muse_spark"
fixture = runpy.run_path(str(next(here.parent.glob("v191_*/independent_epilogue.py"))))
device, launch = (src/n for n in ("decode_gate_up.cuh", "decode_gate_up_launch.cuh"))
dev, binding = device.read_text(), launch.read_text()
header = fixture["CPP"].replace("SharedStorageType smem;", "unsigned load_thread_id() const {return 0;} SharedStorageType smem;")
header = header.replace("uint32_t top_k; }", "uint32_t top_k; const void *b, *bs; }")
constants = dev[dev.index("#ifdef MUSE_GATE_PROMPT"):dev.index("using MuseGateCtx")]
map_fn = fixture["function"](binding,"static bool bank_map(")
map_sig = re.search(r'template\s*<bool Prompt, uint32_t Bands>\s*static bool bank_map',binding)
assert map_sig
loader = fixture["function"](dev,"struct MusePackedGateLoader") + ";"
assert '"l"(map), "r"(col), "r"(row)' in loader
loader, replaced = re.subn(r'asm volatile\("cp\.async\.bulk\.tensor\.4d.*?: "memory"\);', 
    "last_map = map; last_col = col; last_row = row;",loader,flags=re.DOTALL)
assert replaced == 1
prelude = """
static const void* last_map;
static uint32_t last_col, last_row;
uint32_t cast_smem_ptr_to_uint(void*) { return 0; }
"""
call = """
template<bool Weight> bool make_map(Tensor bank, unsigned inner, CUtensorMap* out) {
  return bank_map<false, MUSE_GATE_BANDS>(bank, inner, Weight, out);
}
"""
harness = here / "independent_w13_maps.cpp"
data=header+constants+"\ntemplate<bool Prompt, uint32_t Bands>\n"+map_fn+prelude+"template<bool Weight>\n"+loader+call+harness.read_text()
cases=[]
with tempfile.TemporaryDirectory(prefix="okc-wide-descriptors-") as name:
    folder=Path(name)
    cpp,exe=folder/"check.cpp",folder/"check"
    cpp.write_text(data)
    for bands in (2,4):
        subprocess.run(["c++","-O2","-std=c++17","-DMUSE_GATE_M=24","-DHUMMING_SHAPE_K=4096","-DPAD=1",f"-DTEST_N={bands*128}",
                        f"-DMUSE_GATE_BANDS={bands}",str(cpp),"-o",str(exe)],check=True)
        correct,wrong_plane,holes=map(int,subprocess.check_output([exe],text=True).split())
        cases.append(dict(block_width=bands*128,checked_packed_element_addresses=correct,
            wrong_addresses_if_gate_plane_duplicated=wrong_plane,missing_addresses_if_half_width_box=holes))
print(json.dumps(dict(cases=cases,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
      for p in (device,launch,harness)},physical_gpu=False,
      scope="Source encoder, source seek and post-increment, and source 4-D descriptor coordinates executed on CPU. The CUDA TMA instruction is replaced by observation of its map and integer coordinates. No device transport or timing."),indent=2))
