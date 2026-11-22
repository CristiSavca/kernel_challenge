"""Run the V213 cooperative native/candidate K-reduction differential on the four-band layout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent
vendor,stage=map(Path,sys.argv[1:])
orig=here.parent/'v213_live_decode_expert_reduction'
host=(orig/'partials.cpp').read_text()
for a,b in [('N=512/G','N=1024/G'),('K=64,N=32','K=64,N=64'),('float[2][1][R/2]','float[4][1][R/2]'),('parts=R,owners','parts=2*R,owners')]:
    assert host.count(a)==1
    host=host.replace(a,b)
sources={'native':vendor/'include/humming/epilogue/smem_reducer.cuh','submitted':stage/'python/sglang/kernels/jit/csrc/muse_spark/live_decode_reduce.cuh'}
with tempfile.TemporaryDirectory(prefix='okc-v297-wide-partials-') as td:
    folder=Path(td)
    for name,path in sources.items():
        code=path.read_text().replace('#include <humming/utils/all.cuh>','').replace('void reduce(uint32_t *regs_ptr)','Task reduce(uint32_t *regs_ptr)').replace('void call(uint32_t* words)','Task call(uint32_t* words)').replace('ctx.sync_math_threads();','co_await std::suspend_always{};')
        (folder/(name+'.h')).write_text(code)
    (folder/'main.cpp').write_text(host)
    subprocess.run(['c++','-O2','-std=c++20','-fno-strict-aliasing','-ffp-contract=off','-Wno-deprecated-enum-enum-conversion','-Wno-deprecated-anon-enum-enum-conversion','-I'+td,folder/'main.cpp','-o',folder/'oracle'],check=True)
    result=json.loads(subprocess.check_output([folder/'oracle']))
result['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()}
result['scope']='Host cooperative execution of production reducers on four output bands, two/four K groups and all live row counts in native accumulation order.'
print(json.dumps(result,indent=2))
