"""Compare production half projection and native linear at full checkpoint dimensions."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def main(stage):
    path = stage / "python/sglang/srt/layers/communication/sp_half_qkv.py"
    namespace = {"torch": torch}
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "_project")
    exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    project = namespace["_project"]
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(260)
    width, channels = 8192, 2304
    weight = torch.empty((channels, width), dtype=torch.bfloat16).normal_(std=width ** -0.5, generator=rng)
    cases = []
    with torch.inference_mode():
        for rows in (16, 126, 256, 1152, 16384):
            x = torch.randn((rows, width), dtype=weight.dtype, generator=rng)
            reference = F.linear(x, weight)
            dest = torch.empty_like(reference)
            half = rows // 2
            for offset in (0, half):
                project(x[offset:offset + half], weight, dest[offset:offset + half], False)
            # Compare in blocks to bound host memory at the full prompt size.
            max_error = 0.0
            differing = 0
            for offset in range(0, rows, 256):
                actual, native = dest[offset:offset + 256], reference[offset:offset + 256]
                differing += int(torch.count_nonzero(actual.view(torch.int16) != native.view(torch.int16)))
                max_error = max(max_error, (actual.float() - native.float()).abs().max().item())
            cases.append(dict(rows=rows, hidden=width, channels=channels, total=dest.numel(),
                              bit_differences=differing, max_abs_error=max_error))
    return dict(cases=cases, total=sum(c["total"] for c in cases), differing=sum(c["bit_differences"] for c in cases),
                device="CPU", torch=str(torch.__version__), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope="Full native K and output width, including the full 16K prompt. CUDA backend and model outputs require H200.")


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
