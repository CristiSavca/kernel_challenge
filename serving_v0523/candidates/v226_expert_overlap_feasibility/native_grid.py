"""Verify the published Humming persistent grid and W13 launch directly from source, in memory."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

root=Path(__file__).resolve().parent.parent
url='https://pypi.org/pypi/humming-kernels/0.1.12/json'
with urlopen(url,timeout=15) as f: meta=json.load(f)
wheel=next(x for x in meta['urls'] if x['filename'].endswith('x86_64.whl'))
with urlopen(wheel['url'],timeout=15) as f: data=f.read(1<<20)
assert len(data)==wheel['size'] and sha256(data).hexdigest()==wheel['digests']['sha256']
with ZipFile(BytesIO(data)) as z:
    findings={}
    for path,items in {
        'humming/csrc/launcher/launcher.cpp':['num_ctas = kernel_data.num_ctas_per_sm * get_num_sms(num_sms, dev)','config.gridDimX = num_ctas'],
        'humming/include/humming/scheduler.cuh':['kNumCtaGroups = gridDim.x / kMultiCastSize','dp_mn_next_index += gridDim.x / kMultiCastSize'],
        'humming/kernel/humming.py':['num_sms = config.pop("num_sms", 0)']}.items():
        source=z.read(path).decode(); lines=source.splitlines()
        findings[path]={'sha256':sha256(source.encode()).hexdigest(),'rows':[
            next(dict(line=i,text=s.strip()) for i,s in enumerate(lines,1) if match in s) for match in items]}
patch=next(root.glob('v224_*/patch.diff')).read_text()
assert 'cuLaunchKernel(reinterpret_cast<CUfunction>(function), sms, 1, 1, Prompt ? 384 : 640' in patch
assert 'config.pop("num_sms", 0) or any(config.get(k) != v' in patch
print(json.dumps(dict(pypi=url,wheel=wheel['filename'],wheel_sha256=sha256(data).hexdigest(),
    native=findings,v224_sha256=sha256(patch.encode()).hexdigest(),full_sms_custom_W13=True,
    custom_rejects_selected_num_sms=True),indent=2))
