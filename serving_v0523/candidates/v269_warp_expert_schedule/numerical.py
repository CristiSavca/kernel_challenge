"""Re-run the native BF16/FP32 reducer  with the submitted source."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
prior = here.parent / 'v263_token_major_expert_decode'
header = Path('sglang/python/sglang/kernels/jit/csrc/muse_spark/live_decode_reduce.cuh')
body = (root / header).read_bytes()
split = b'// The first warp builds one shared prefix of expert tile starts per CTA.'
assert body.count(split) == 1
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / 'humming').symlink_to(root / 'humming', target_is_directory=True)
    h = tmp / header
    h.parent.mkdir(parents=True)
    h.write_bytes(body.split(split)[0])
    reduction = json.loads(subprocess.check_output([sys.executable,'-B',str(prior/'partials.py'),tmp]))

print(json.dumps(dict(checked_header_sha256=sha256(body).hexdigest(),native_FP32_reduction=reduction,
    scope="The V263 fixture compiles the unmodified reducer definition from the submitted header. The independent SIMT fixture executes the adjacent scheduler definition."),indent=2))
