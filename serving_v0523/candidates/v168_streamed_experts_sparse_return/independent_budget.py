"""Check V168's token window budget across sparse ownership, padding and nonuniform ranks."""
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def check(stage):
    path = stage / "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
    data = path.read_bytes()
    fn = next(n for n in ast.parse(data).body if getattr(n, "name", "") == "_compact_prefill_windows")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), str(path), "exec"), ns)
    plan = ns[fn.name]
    rng = np.random.default_rng(168009)
    totals = Counter()
    torch.set_num_threads(1)
    for ep in (2, 3, 4, 6, 8):
        for rows in (1, 127, 128, 129, 8191, 8192, 8193, 16383, 16384, 16385):
            for topk in (1, 2, 8):
                for density in (0., 1/ep, .75, 1.):
                    # Storage deliberately has a noncontiguous last dimension.
                    bank = rng.integers(0, 32, size=(rows,topk*2), dtype=np.int32)
                    original = bank[:,::2]
                    original[rng.random(original.shape) >= density] = -1
                    owned = (original >= 0).any(axis=1)
                    compact = original[owned]
                    ids = torch.from_numpy(compact)
                    live = int((original >= 0).sum())
                    expected = (rows*topk + ep - 1)//ep
                    cap = expected + max(128*topk, (expected+7)//8)
                    known = list(plan(ids, 32, 32*ep, live, rows))
                    dynamic = list(plan(ids, 32, 32*ep, None, rows))
                    assert known == dynamic
                    # Rebuild the original global (row, slot) coordinates from
                    # every emitted compact window, independent of its boundaries.
                    owners = np.flatnonzero(owned)
                    want = np.flatnonzero(original.ravel() >= 0)
                    parts = []
                    cursor = 0
                    for begin, end, count in known:
                        assert cursor == begin < end <= len(owners)
                        row, col = np.nonzero(compact[begin:end] >= 0)
                        parts.append(owners[begin+row]*topk + col)
                        assert count == len(col) <= cap
                        cursor = end
                    assert cursor == len(owners)
                    assert np.array_equal(np.concatenate(parts) if parts else np.empty(0), want)
                    if live <= cap: assert len(known) == int(owned.any())
                    totals["cases"] += 1
                    totals["one_window"] += len(known)==1
                    totals["multi_window"] += len(known)>1
                    totals["empty_owners"] += not len(owners)
                    totals["compared_original_route_coordinates"] += live
    return {"source_sha256": hashlib.sha256(data).hexdigest(), "counts": totals, "device": "CPU"}


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1])), indent=2))
