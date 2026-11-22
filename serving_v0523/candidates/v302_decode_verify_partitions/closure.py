"""Check complete generated expert machine code and capture the team's frozen source."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root, donor, combined, rendered = map(Path,sys.argv[1:])
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
cases={}
for path in sorted(donor.glob('*.cubin')):
    stem=path.stem
    cu=donor/(stem+'.cu')
    other=combined/(stem+'.cu')
    assemb=path.with_suffix('.sass')
    assert path.read_bytes()==(combined/path.name).read_bytes(),stem
    assert assemb.read_bytes()==(combined/assemb.name).read_bytes(),stem
    assert other.read_bytes()==(rendered/other.name).read_bytes(),stem
    cases[stem]=dict(donor_source=sha(cu),combined_source=sha(other),cubin=sha(path),sass=sha(assemb))
assert len(cases)==12
paths=['python/sglang/kernels/jit/csrc/muse_spark/grouped_decode_mma_pair.cuh',
       'python/sglang/srt/layers/muse_spark/decode_mma_pair.py',
       'python/sglang/srt/layers/muse_spark/residual_prompt_mma.py',
       'python/sglang/kernels/ops/attention/paged_verify_attention.py']
print(json.dumps(dict(tree=subprocess.check_output(['git','-C',root,'write-tree'],text=True).strip(),
    source_sha256={p:sha(root/p) for p in paths},all_cubins_and_sass_identical=True,cases=cases,
    compiler='CUDA 12.9.86, compute_90a/sm_90a, Humming 0.1.12'),indent=2))
