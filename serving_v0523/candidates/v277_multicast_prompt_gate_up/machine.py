"""Compare emitted machine code and check the exact public launcher exports."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys

scratch=Path(sys.argv[1])
hashes={line.split()[1]:line.split()[0] for line in (scratch/"compiled.txt").read_text().splitlines() if len(line.split())==2}
exports=(scratch/"wrapper.symbols").read_text()
for entry in ("configure","prepare","prepare_prompt","prompt_cluster_grid","run","run_prompt","run_prompt_multicast"):
    assert " T __tvm_ffi_"+entry+"\n" in exports
for call in ("cuLaunchKernel","cuLaunchKernelEx","cuOccupancyMaxActiveClusters"):
    assert " U "+call+"\n" in exports
records=[]
for name,mc in (("GroupedResidualPromptGateUpKernel",False),("GroupedMulticastPromptGateUpKernel",True)):
    stem=scratch/"images"/name
    code=stem.with_suffix(".cu").read_bytes()
    sass=stem.with_suffix(".sass").read_text()
    log=stem.with_suffix(".log").read_text()
    assert hashlib.sha256(code).hexdigest()==hashes["images/"+name+".cu"]
    kernels=re.findall(r"HGMMA\.64x(\d+)x16.F32.BF16",sass)
    counts=Counter(kernels)
    assert counts=={"32":32,"96":32,"176":32} and sass.count("WARPGROUP.DEPBAR.LE")==48
    assert sass.count("UTMALDG.4D")==14 and sass.count("UTMALDG.2D")==7 and sass.count("UTMALDG.2D.MULTICAST")==(7 if mc else 0)
    regs=int(re.search(r"Used (\d+) registers",log)[1])
    stack,stores,loads=map(int,re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",log).groups())
    records.append(dict(kernel=name,source_sha256=hashes["images/"+name+".cu"],cubin_sha256=hashes["images/"+name+".cubin"],
                        sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),registers=regs,stack_bytes=stack,spill_store_bytes=stores,
                        spill_load_bytes=loads,HGMMA=counts,warpgroup_waits=48,TMA_4D=14,TMA_2D=7,TMA_2D_multicast=7 if mc else 0))
print(json.dumps(dict(target="sm_90a",nvcc="12.9",images=records,wrapper_object_sha256=hashes["wrapper.o"],
                      wrapper_source_sha256=hashes["wrapper.cu"],TVM_exports=7,actual_GPU=False),indent=2))
