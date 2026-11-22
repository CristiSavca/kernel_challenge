"""Measure actual large EXTEND row counts and bound eliminated query materialization."""
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
qwidth, layers, chunk = 16 * 64, 62, 16384
cutoff = 4096
iteration = -1
result = defaultdict(Counter)
for line in log.read_text().splitlines():
    if "Cache flushed successfully!" in line:
        iteration += 1
    match = rx.search(line)
    if iteration < 0 or match is None:
        continue
    requests, new, hits = map(int, match.groups())
    if new < cutoff:
        continue
    assert "TP0 EP0" in line and "cuda graph: True" in line
    record = result[order[iteration]]
    record["forwards"] += 1
    record["new_rows"] += new
    record["multi_request_forwards"] += int(requests >= 2)
    record["read_write_query_bytes_per_rank"] += 4 * new * qwidth * layers
    record["additional_encoding_lookup_bytes_per_rank"] += new * qwidth * layers
assert iteration+1 == len(order)
ref = json.loads((here.parent / "v170_long_prefill_attention_feasibility/check.json").read_bytes())
first_turn = ref["shape"]["first_turn"]
assert first_turn == 74160
print(json.dumps(dict(source=str(log.relative_to(root)), full_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    threshold=cutoff, q_elements_per_rank_per_row=qwidth, layers=layers,
    full_chunk_rows=chunk, saved_named_q_write_and_read_bytes_per_full_chunk=4*chunk*qwidth*layers,
    added_table_load_bytes_per_full_chunk=chunk*qwidth*layers, table_resident_bytes=65536,
    full_chunk_parent_q_cast_launches_removed=layers,
    per_load=dict(sorted(result.items())), public_first_turn_attested_rows=first_turn,
    first_turn_block_decomposition=[chunk]*4+[first_turn-4*chunk],
    first_turn_saved_q_intermediate_bytes=first_turn*qwidth*layers*4,
    qualification="Archive logger measures live rows, calls and graph use. Counts assume the ordinary native Triton Muse and eligible FA3 forward. Named buffer accesses and table loads are logical; they do not establish DRAM transactions or latency.",
    score_or_hardware_speedup_measured=False), indent=2))
