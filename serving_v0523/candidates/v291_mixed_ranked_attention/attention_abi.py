"""Exercise the public tail workspace policy and both FA3 output conventions on the frozen tree."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
tree=json.loads((here/"components.json").read_bytes())["integrated240"]
paths=["python/sglang/"+p for p in (
    "kernels/ops/attention/paged_verify_attention.py","srt/layers/attention/flashattention_backend.py",
    "srt/layers/attention/direct_prompt_kv.py","kernels/ops/attention/flash_attention.py",
    "kernels/ops/attention/flash_attention_v3.py","kernels/aot/python/sgl_kernel/flash_attn.py")]
source_digests={}
with tempfile.TemporaryDirectory() as folder:
    for path in paths:
        raw=subprocess.check_output(["git","-C",stage,"show",tree+":"+path])
        dest=Path(folder)/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
        source_digests[path]=hashlib.sha256(raw).hexdigest()
    test=here.parent/"v240_mixed_paged_verify_tail/limits.py"
    report=json.loads(subprocess.check_output([sys.executable,"-B",test,folder]))
assert report["native_abi_cases"]==72 and report["distinct_storage_copybacks"]==72
assert report["provider_legacy_retries"]==36 and sum(report["memory_geometry"].values())==168
print(json.dumps(dict(tree=tree,scope="Exact V291 source on host Torch; native CUDA launches are replaced with explicit spies.",
    sources=source_digests,**report),indent=2))
