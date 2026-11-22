"""Execute the submitted full-stage method with delayed register reads and poison an alias."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
source = stage / 'python/sglang/kernels/jit/csrc/muse_spark/grouped_decode_mma_pair.cuh'
original = here.parent / 'v183_paired_grouped_mma/mma_oracle.cpp'
native = original.read_text()
body = source.read_text().split('struct MuseFullStageMma',1)[1].split('using MuseSelectedMma',1)[0]
body = 'struct MuseFullStageMma' + body
template = native.replace('struct MmaOpClass {','struct MmaOpClass {\n  using BRegisters = uint32_t[4];')
template = template.replace('#include HEADER', '''struct MuseOverlapCtx { struct BlockShape { static constexpr unsigned M = 8; }; };
struct MuseOverlapMma : WGMMA<MuseOverlapCtx,MainloopArithmetic<MuseOverlapCtx>> {
  static constexpr unsigned bands=2;
};
#include HEADER''')
template = template.replace('if(i>=2) assert','if(i==3) assert').replace('compute<MuseOverlapMma>(s,seed)','compute<MuseFullStageMma>(s,seed)').replace('max_unretired==4','max_unretired==8')
def test(directory, impl):
    path=directory/'body.hpp'
    path.write_text(impl)
    cpp=directory/'check.cpp'
    cpp.write_text(template)
    exe=directory/'check'
    subprocess.run(['c++','-std=c++20','-O2','-DMUSE_OVERLAP_M=8','-DHEADER="'+str(path)+'"',str(cpp),'-o',str(exe)],check=True)
    return subprocess.run([str(exe)],cwd=directory,capture_output=True)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    good=test(root,body)
    assert good.returncode==0,good.stderr
    repeated=body.replace('held[iteration][j]', 'held[iteration % 2][j]')
    bad=test(root,repeated)
    assert bad.returncode and b'memcmp' in bad.stderr,bad
    print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_oracle_sha256=hashlib.sha256(original.read_bytes()).hexdigest(), counts=json.loads(good.stdout),
        alias_mutation_rejected=True, original_cuda_body=True),indent=2))
