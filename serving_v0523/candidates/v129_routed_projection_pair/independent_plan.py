"""Run the production V129 route planner under a random cooperative warp scheduler."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
source = stage/'python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh'
text = source.read_text()
start = text.index('__global__ void plan_routes(')
end = text.index('\n}\n',start) + 2
body = text[start:end]
for old,new in (
        ('__global__ void plan_routes(', 'Task plan_routes('),
        ('  extern __shared__ int32_t matches[];\n', ''),
        ('  __shared__ uint32_t next_job;\n', '')):
    assert body.count(old) == 1
    body = body.replace(old, new)
with tempfile.TemporaryDirectory(prefix='okc-v129-plan-') as tmp:
    path = Path(tmp)
    unit = path/'main.cpp'
    unit.write_text((here/'independent_plan.cpp').read_text().replace('PRODUCTION_PLAN',body))
    subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',str(unit),'-o',str(path/'check')],check=True)
    result=json.loads(subprocess.check_output([str(path/'check')],text=True))
print(json.dumps(dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      cpu_planner_sha256=hashlib.sha256(text[start:end].encode()).hexdigest(),
                      **result),indent=2))
