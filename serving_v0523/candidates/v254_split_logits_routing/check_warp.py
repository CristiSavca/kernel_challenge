"""Compile the pinned and modified CUDA warp bodies portably; replace device primitives only."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
path = 'python/sglang/kernels/jit/csrc/muse_spark/routing.cuh'
orig = subprocess.check_output(['git','-C',stage,'show','HEAD:'+path]).decode()
new = (stage/path).read_text()

def cpu(source, name):
    body=source[source.index('namespace muse_spark {'):source.index('}  // namespace muse_spark')]
    body=body.replace('namespace muse_spark {', f'namespace {name} {{')
    body=body.replace('asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));','r = std::exp2(x);')
    body=body.replace('asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));','r = a / b;')
    body,n=re.subn(r'asm\("add.rn.f32 %0, %0, %1;" : "\+f"\(g.([xyzw])\) : "f"\(r.\1\)\);',r'g.\1 = g.\1 + r.\1;',body)
    assert n==(4 if name=='candidate' else 0) and 'asm(' not in body
    return body+'}\n'

with tempfile.TemporaryDirectory(prefix='okc-v254-warp-') as sub:
    sub=Path(sub)
    (sub/'original_warp.hpp').write_text(cpu(orig,'original'))
    (sub/'candidate_warp.hpp').write_text(cpu(new,'candidate'))
    subprocess.run(['c++','-w','-std=c++20','-O2','-ffp-contract=off','-pthread','-I',sub,here/'router_warp.cpp','-o',sub/'warp'],check=True)
    out=json.loads(subprocess.check_output([sub/'warp']))
out.update(header_sha256=hashlib.sha256(new.encode()).hexdigest(),modes=['sigmoid','softmax'],
           scope='CPU differential of the actual two warp bodies; approximate exp2 and div.full have equal CPU stand-ins.')
print(json.dumps(out,indent=2))
