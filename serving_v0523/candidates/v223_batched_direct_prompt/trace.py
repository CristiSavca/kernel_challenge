"""Count V223 long multi-request shape and memory bounds in the archived prefill trace."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

here = Path(__file__).resolve().parent
root = here.parents[2]
archive = root / "experiments/modal_node/20261113T204343Z-bench"
log = next(archive.glob("*v41__server.log"))
load = json.loads(next(archive.glob("*load.json")).read_bytes())
order = [r["concurrency"] for r in load["rungs"] for _ in range(r["reps"])]
rx = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),")
by_load = defaultdict(Counter)
width = 2 * 64
limit = 32 * 1024 * 1024
iteration = -1
for line in log.read_text().splitlines():
    if "Cache flushed successfully!" in line:
        iteration += 1
    if iteration < 0:
        continue
    m = rx.search(line)
    if m is None:
        continue
    assert "TP0 EP0" in line and "cuda graph: True" in line
    count, rows, hits = map(int, m.groups())
    if rows < 4096:
        continue
    rec = by_load[order[iteration]]
    rec["long_forwards"] += 1
    rec["long_rows"] += rows
    if count < 2:
        continue
    rec["long_multi_forwards"] += 1
    rec["long_multi_rows"] += rows
    common = 2 * rows + 256 * count
    if 2 * width * (common + 2047 * count) <= limit:
        rec["swa_forwards_under_worst_case_scratch"] += 1
        rec["swa_rows_under_worst_case_scratch"] += rows
    if 2 * width * (common + hits) > limit:
        rec["full_forwards_over_limit_from_cache_hits_alone"] += 1
assert iteration + 1 == len(order)
print(json.dumps(dict(source=str(log.relative_to(root)), sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                      staged_row_bytes=width, budget=limit, by_load=dict(sorted(by_load.items())),
                      qualification="Accounts for current FP8 K/V, packed K/V, 256 guard rows per sequence. Recorded prefix cache hits bound retained keys from below. Dispatch and latency require hardware."),indent=2))
