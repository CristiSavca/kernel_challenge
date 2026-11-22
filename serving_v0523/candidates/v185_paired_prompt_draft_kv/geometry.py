"""Count actual-shape opportunities in the V41 trace and size V185's named tensors."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
SIZE, HIDDEN, CHANNELS, ELEMENT_BYTES = 8, 8192, 1280, 2


def geometry(tokens):
    rows = (tokens + SIZE - 1) // SIZE
    full = (SIZE - 1) * rows * HIDDEN * ELEMENT_BYTES

    def buffers(width):
        peers = SIZE // width - 1
        cohort = width * rows
        wire = dict(input=(width - 1) * rows * HIDDEN, weights=peers * HIDDEN * CHANNELS,
                    projected=peers * cohort * CHANNELS)
        stage = dict(gathered=cohort * HIDDEN, remote_weights=wire["weights"],
                     projected=SIZE * rows * CHANNELS, outgoing=wire["projected"])
        return [{k: v * ELEMENT_BYTES for k, v in rec.items()} for rec in (wire, stage)]

    width = min((2, 4), key=lambda value: sum(buffers(value)[0].values()))
    wire, stage = buffers(width)
    sent = sum(wire.values())
    return dict(tokens=tokens, aligned_rows=rows * SIZE, width=width,
                passes_geometry=tokens >= 128 * SIZE and 8 * sent <= 7 * full,
                parent_wire_bytes=full, wire_bytes=wire, total_wire_bytes=sent,
                parent_gathered_bytes=SIZE * rows * HIDDEN * ELEMENT_BYTES,
                staged_bytes=stage, total_staged_bytes=sum(stage.values()),
                live_projected_bytes=tokens * CHANNELS * ELEMENT_BYTES,
                full_k_gemm_flops=2 * SIZE * rows * HIDDEN * CHANNELS)


trace = runpy.run_path(str(ROOT / "experiments/iterations/20261118-serving-mixed-graph-capacity.py"))
log = next(trace["ARCHIVE"].glob("*v41__server.log"))
client = next(trace["ARCHIVE"].glob("*load.json"))
order = [r["concurrency"] for r in json.loads(client.read_bytes())["rungs"] for _ in range(r["reps"])]
counts = defaultdict(Counter)
for load, (_, tokens, _) in trace["read_trace"](log.read_text(), order):
    count = counts[load]
    count["calls"] += 1
    count["rows"] += tokens
    case = geometry(tokens)
    if case["passes_geometry"]:
        count["eligible_by_size"] += 1
        count["eligible_live_rows"] += tokens
        count["width_" + str(case["width"])] += 1
        count["saved_payload_bytes_per_rank"] += case["parent_wire_bytes"] - case["total_wire_bytes"]

print(json.dumps(dict(tp=SIZE, hidden=HIDDEN, local_draft_kv_channels=CHANNELS, dtype="BF16",
    examples=[geometry(n) for n in (753, 1506, 4096, 8192, 8624, 16384)],
    archive=str(log.relative_to(ROOT)), archive_sha256=sha256(log.read_bytes()).hexdigest(),
    archived_shape_opportunities=counts,
    scope="Wire counts are per-rank logical sends. Staging excludes existing weights, owner input shard, allocator rounding, cuBLAS workspace and later cache writes. Archive omits actual V185 capture and module eligibility. Throughput and process peak are unmeasured."), indent=2))
