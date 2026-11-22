"""Exercise the unchanged Triton packer on V239's single- and multi-prompt split grids."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import importlib.util

import numpy as np
import triton

source, helper = map(Path, sys.argv[1:3])
spec = importlib.util.spec_from_file_location("packed_attention_fixture", helper)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
rng = np.random.default_rng(239)
cases = []
with tempfile.TemporaryDirectory() as work:
    fn = original.load_jit_module(source, Path(work))._pack_prompt_rows
    for dtype in (np.int32, np.int64):
        for width, q, p in ((64,[4096],[0]), (128,[16384],[49152]), (128,[10617,5767],[32804,917]),
                            (256,[4097],[892]), (1024,[4096],[768])):
            size = original.replay(fn, rng, width, dtype, q, p, -1)
            cases.append(dict(row_width=width, query=q, prefix=p, index=str(dtype), exact_bytes=size))
print(json.dumps(dict(kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),triton=triton.__version__,
    cases=cases,checked_bytes=sum(c['exact_bytes'] for c in cases),source='verbatim Triton JIT kernel executed in CPU interpreter',
    canary_and_guard=True,physical_H200=False), indent=2))

