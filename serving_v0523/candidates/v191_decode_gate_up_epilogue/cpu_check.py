"""Exercise all two K-warp offsets and the submitted small-row run() against immediate native retirements."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
header=stage/"python/sglang/kernels/jit/csrc/muse_spark/grouped_decode_mma_pair.cuh"
src=(here.parent/"v183_paired_grouped_mma/mma_oracle.cpp").read_text()
src=src.replace("char x[8192]", "char x[32768]")
src=src.replace("std::array<std::array<float,8>,16>", "std::array<std::array<float,8>,64>")
src=src.replace("unsigned kslice=(p.desc%sizeof(Storage::StageStorage))/(2*sizeof(int4));", """
    unsigned offset=p.desc%sizeof(Storage::StageStorage);
    unsigned warp=offset/(MUSE_OVERLAP_M*128);
    unsigned kslice=(offset%(MUSE_OVERLAP_M*128))/(2*sizeof(int4));
    assert(warp<2 && kslice<4);""")
src=src.replace("act[4*stage+kslice]", "act[16*warp+4*stage+kslice]")
src=src.replace("uint64_t desc=stage*sizeof(SharedStorage::StageStorage)+iter*2*sizeof(int4);",
                "uint64_t desc=stage*sizeof(SharedStorage::StageStorage)+iter*2*sizeof(int4)+smem_offset;")
src=src.replace("Op op{}; std::mt19937 rng(seed);",
    """Op op{}; unsigned warp=seed%2; op.smem_offset=warp*MUSE_OVERLAP_M*128; std::mt19937 rng(seed);""")
src=src.replace("act[(s%4)*4+i]", "act[16*warp+(s%4)*4+i]")
src=src.replace("seed<11", "seed<32")
src=src.replace('#include HEADER', 'using MuseGateCtx=KernelContext<MmaOpClass, Shape<0,8192,4096>, Shape<MUSE_OVERLAP_M,256,128>>;\n#define MUSE_OVERLAP_CONTEXT MuseGateCtx\n#include HEADER')
assert "#define MUSE_OVERLAP_CONTEXT" in src
results={}
with tempfile.TemporaryDirectory() as temp:
    work=Path(temp);driver=work/"driver.cpp";driver.write_text(src)
    for w in (8,16,24):
        exe=work/f"v189_{w}"
        subprocess.run(["c++","-std=c++20","-O2",f'-DHEADER="{header}"',f"-DMUSE_OVERLAP_M={w}",str(driver),"-o",str(exe)],check=True)
        results[w]=json.loads(subprocess.check_output([exe]))
print(json.dumps(dict(header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),source_fixture_sha256=hashlib.sha256(src.encode()).hexdigest(),
    k_warp_offsets_tested_per_width=2, widths=results,physical_H200=False),indent=2))
