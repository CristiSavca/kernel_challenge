import json, hashlib, subprocess, sys,tempfile
from pathlib import Path
r=Path(sys.argv[1]);d=Path(__file__).resolve().parent
sources={'native':r/'humming/include/humming/epilogue/smem_reducer.cuh',
         'submitted':r/'sglang/python/sglang/kernels/jit/csrc/muse_spark/live_decode_reduce.cuh'}
with tempfile.TemporaryDirectory() as td:
    t=Path(td)
    for name,p in sources.items():
        s=p.read_text().replace('#include <humming/utils/all.cuh>', '').replace('void reduce(uint32_t *regs_ptr)', 'Task reduce(uint32_t *regs_ptr)').replace('void call(uint32_t* words)','Task call(uint32_t* words)').replace('ctx.sync_math_threads();','co_await std::suspend_always{};')
        (t/(name+'.h')).write_text(s)
    exe=t/'oracle'
    subprocess.run(['c++','-O2','-std=c++20','-fno-strict-aliasing','-ffp-contract=off','-Wno-deprecated-enum-enum-conversion','-I'+td,str(d/'partials.cpp'),'-o',str(exe)],check=True)
    result=json.loads(subprocess.check_output([str(exe)]))
result['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()}
print(json.dumps(result,indent=2))
