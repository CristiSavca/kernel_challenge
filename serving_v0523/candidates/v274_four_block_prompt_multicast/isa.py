"""Read the real cubins' ELF resources and disassemble every complete residual K loop."""
from pathlib import Path
from hashlib import sha256
import json
import sys

here=Path(__file__).resolve().parent
fixture=here.parent/"v270_wide_prompt_weight_reuse/isa.py"
ns=dict(__file__=str(fixture))
exec(compile(fixture.read_text().split("rows = {}",1)[0],str(fixture),"exec"),ns)
images=Path(sys.argv[1])
records={}
for sass in sorted(images.glob("*.sass")):
    source=sass.with_suffix(".cu")
    row=ns["inspect"](sass,source)
    name=sass.stem
    if name.startswith(("Wide","Diagnostic")):
        assert row["shared_bytes"]==156672 and row["stack_spill_store_load"]==[0,0,0]
        assert set(row["loops"])=={32,128,224,256} and row["multicast"]>0
        assert all(r["drains"]==8 and r["hgmma"]==16 and r["local_loads"]==r["local_stores"]==0 for r in row["loops"].values())
        s=source.read_text()
        waits=s[s.index("void wait_math_epilogue()"):s.index("void wait_reduce_epilogue()")]
        assert waits.index("mbarrier_wait")<waits.index("ctx.sync_load_threads()")<waits.index("tma_fence_async_shared")<waits.index("mbarrier_arrive<true>")<waits.index("phases[kNumStages] ^=")
        assert "kNumMathThreads * factor + (cluster_rank == 0 && thread_id == kNumStages ? kMultiCastSize - 1 : 0)" in s
        asm=sass.read_text()
        row["async_fences"]=asm.count("FENCE.VIEW.ASYNC.S")
        row["dsm_arrival_instructions"]=asm.count("SYNCS.ARRIVE.TRANS64.RED.A1T0")
    records[name]=row
print(json.dumps(dict(toolchain="CUDA 12.9.86 nvcc -O3 sm_90a; real ptxas, cuobjdump, ELF resources",GPU=False,variants=records),indent=2))
