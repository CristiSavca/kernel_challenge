"""Count batches meeting the prompt length and cardinality predicates in a public V41 trace."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[3]
trace = runpy.run_path(str(root / "experiments/iterations/20261118-serving-mixed-graph-capacity.py"))
log = next(trace["ARCHIVE"].glob("*v41__server.log"))
client = next(trace["ARCHIVE"].glob("*load.json"))
order = [r["concurrency"] for r in json.loads(client.read_bytes())["rungs"] for _ in range(r["reps"])]
tallies = defaultdict(Counter)
for concurrency, (sequences, tokens, _) in trace["read_trace"](log.read_text(), order):
    counts = tallies[concurrency]
    counts["all_calls"] += 1
    counts["all_rows"] += tokens
    if tokens >= 4096:
        counts["long_calls"] += 1
        counts["long_rows"] += tokens
        if sequences == 1:
            counts["long_single_calls"] += 1
            counts["long_single_rows"] += tokens
for counts in tallies.values():
    counts["single_share_of_long_rows"] = counts["long_single_rows"] / counts["long_rows"]
print(json.dumps(dict(source=str(log.relative_to(root)),file_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    workload="Eight-H200 V41 scheduler log; real #new-seq and #new-token since each cache flush",
    interpretation="Shape opportunities only. The log does not establish V180's pool type, mapping checks, live phase duration or actual H200 performance.",
    by_concurrency=tallies),indent=2))
