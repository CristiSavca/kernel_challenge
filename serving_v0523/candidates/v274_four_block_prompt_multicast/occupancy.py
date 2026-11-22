"""Check launch ABI round-down, cluster dimensions and cleanup using pinned CUDA launch structures."""
from pathlib import Path
import contextlib
import io
import json
import sys

fixture=Path(sys.argv[2]).read_text()
all_cases=[]
for sms,size in ((132,2),(132,4),(130,4),(128,4)):
    src=fixture.replace('num_threads=384)','num_threads=384,multi_cast_size_a=SIZE)').replace('==(132,1,1)','==(SMS-SMS%SIZE,1,1)').replace('==(2,1,1)','==(SIZE,1,1)').replace('(kernel,132)','(kernel,SMS)').replace('2*min(66,active)','SIZE*min(SMS//SIZE,active)')
    ns=dict(SIZE=size,SMS=sms)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(compile(src,'v270_test_replay','exec'),ns)
    suite=json.loads(buf.getvalue())
    all_cases.append(dict(sms=sms,fanout=size,config_grid=sms-sms%size,capacity_and_injected_error_checks=len(suite['cases'])))
print(json.dumps(dict(cuda_bindings=suite['version'],source_sha256=suite['model_src_sha256'],cases=all_cases,GPU=False),indent=2))
