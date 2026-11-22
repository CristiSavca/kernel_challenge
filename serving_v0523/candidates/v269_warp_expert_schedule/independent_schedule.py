"""Build the native production schedule class on host with SIMT shuffle/ballot replay."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
header, vendor = map(Path,sys.argv[1:])
if header.is_dir():
    header = header/"python/sglang/kernels/jit/csrc/muse_spark/live_decode_reduce.cuh"
source = header.read_text()
code = source[source.index("template<class Ctx> struct MuseWarpSchedule"):]
vendor_source = vendor/"include/humming/scheduler.cuh"
vendor_loop = vendor/"include/humming/kernel/humming_ws.cuh"
native = vendor_source.read_text()
assert "m_id = mn_index / N_BLOCKS;" in native and "n_id = mn_index % N_BLOCKS;" in native
assert "m_offset = ctx.smem.expert_offset[expert_id] + offset_in_expert;" in native
loop = vendor_loop.read_text()
assert loop.count("while (scheduler.get_next_block())") == 2
assert loop.index("auto scheduler = Scheduler(ctx);") < loop.index("if (ctx.is_load_thread())")
with tempfile.TemporaryDirectory(prefix="okc-warp-schedule-") as temp:
    tmp = Path(temp)
    (tmp/"schedule.inc").write_text(code)
    subprocess.run(["clang++","-std=c++20","-O2","-DSHARED_PREFIX=" + str(int("__clz(" in code)),"-I",temp,str(here/"independent_simt.cc"),
                    "-o",str(tmp/"verify")],check=True)
    result = json.loads(subprocess.check_output([tmp/"verify"]))
result.update(hashes={p.name:sha256(p.read_bytes()).hexdigest()
                      for p in (header,vendor_source,vendor_loop,here/"independent_simt.cc")},
              scope="Host-executed native scheduler, complete grid per case, SIMT dataflow replay, independent expert/row/column enumerator. No CUDA execution or timing.")
print(json.dumps(result,indent=2))
