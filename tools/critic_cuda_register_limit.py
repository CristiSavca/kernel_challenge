#!/usr/bin/env python3
"""CPU call to NVIDIA's shipped occupancy header. No CUDA runtime/GPU used."""
import hashlib,io,json,subprocess,tarfile,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL="https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.0.96-archive.tar.xz"
SHA="25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1"
b=urllib.request.urlopen(URL,timeout=20).read();assert hashlib.sha256(b).hexdigest()==SHA
with tarfile.open(fileobj=io.BytesIO(b),mode="r:xz") as tar:
 name=next(n for n in tar.getnames() if n.endswith("/include/cuda_occupancy.h"));header=tar.extractfile(name).read()
cpp=r'''
#include <cstdio>
#include "cuda_occupancy.h"
int main(){
 cudaOccDeviceProp p{};p.computeMajor=9;p.computeMinor=0;p.regsPerBlock=65536;p.regsPerMultiprocessor=65536;p.warpSize=32;
 for(int threads: {160,544,576})for(int regs: {48,56,64,96,97,104,112,168,200}){
  cudaOccFuncAttributes a{};a.numRegs=regs;cudaOccResult r{};cudaOccPartitionedGCConfig gc=PARTITIONED_GC_OFF;int limit=-1;
  auto err=cudaOccMaxBlocksPerSMRegsLimit(&limit,&gc,&r,&p,&a,threads);
  printf("%d %d %d %d %d\n",threads,regs,int(err),limit,r.allocatedRegistersPerBlock);
 }
}
'''
cpp="#include <initializer_list>\n"+cpp
with tempfile.TemporaryDirectory(prefix="okc-cuda-regs-") as tmp:
 p=Path(tmp);(p/"cuda_occupancy.h").write_bytes(header);exe=str(p/"probe")
 subprocess.run(["clang++","-x","c++","-std=c++17","-O2","-I",tmp,"-o",exe,"-"],input=cpp,text=True,capture_output=True,check=True)
 output=subprocess.check_output([exe],text=True)
rows=[dict(zip(("threads","regs_per_thread","cuda_occ_error","register_admissible_ctas","allocated_registers_per_cta"),map(int,line.split())))for line in output.splitlines()]
assert all(x["cuda_occ_error"]==0 for x in rows)
assert all(x["register_admissible_ctas"]==0 for x in rows if x["threads"]==576 and x["regs_per_thread"]>=97)
result=dict(source_url=URL,archive_sha256=SHA,archive_bytes=len(b),header_sha256=hashlib.sha256(header).hexdigest(),
 method="Compile and call NVIDIA CUDA13.0.96 cudaOccMaxBlocksPerSMRegsLimit unchanged under native clang; SM90 properties. Not an actual device query.",
 conclusion="At576threads,97regs rounds to104 effective; hardware per-CTA verification rounds18warps to20, so66560 assumed registers exceeds65536.96 is maximum register-admissible reported count;112 aggregate arithmetic is insufficient.",
 rows=rows,limitations="Register limit only, excluding shared memory, threads, barriers, compiler spills/serialization and actual GPU/device state.",gpu_used=False)
out=ROOT/"experiments/iterations/20261104-cuda-register-limit-critic.json";out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
