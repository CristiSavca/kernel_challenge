"""Bound the logical FP8 traffic for single-prompt joins in the archived cold wave."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy


here = Path(__file__).resolve().parent
wave = runpy.run_path(str(here.parents[2] / "experiments/iterations/20261119-serving-cold-wave.py"))
server, client = [next(wave["ARCHIVE"].glob(glob)) for glob in ("*v41__server.log", "*v41__load.json")]
trace = json.loads(client.read_bytes())
epochs = server.read_text().split("Cache flushed successfully!")[1:]
order = [(r["concurrency"], i) for r in trace["rungs"] for i in range(r["reps"])]
assert len(order) == len(epochs) == 7 and trace["gpus"] == 8
budget = 32 * 1024**2
width, heads, dim, groups, sms, layers = 8, 2, 64, 8, 132, 16
row_bytes = 2 * heads * dim
tile_m = 1 << (width * groups - 1).bit_length()
partitions = (2 * sms + heads - 1) // heads
totals = {}
for (users, rep), epoch in zip(order, epochs):
    clients = [r for r in trace["requests"] if (r["load"], r["rep"], r["turn"]) == (users, rep, 0)]
    assert len(clients) == users
    visible_upper = max(r["prompt"] for r in clients)
    counts = totals.setdefault(users, Counter())
    counts["repetitions"] += 1
    for event in wave["cold_wave"](epoch, clients):
        q, tail, prompts = event["new"], event["running"], event["seqs"]
        if not tail or not 16384 < q + width * tail <= 16640 or prompts != 1:
            continue
        # Anonymous request identities: charge the longest possible full prompt.
        parent = (visible_upper + 256 + q + width * tail) * row_bytes
        storage = (visible_upper + max(256, width * tail)) * row_bytes
        scratch = (partitions + tail - 1) * heads * tile_m * (dim + 1) * 4
        assert q == 16384 and parent <= budget and storage + scratch <= budget
        counts["eligible_single_prompt_joins"] += 1
        counts["full_layer_calls"] += layers
        counts["logical_copy_bytes_avoided_per_rank"] += 2 * q * row_bytes * layers
        counts["max_parent_admission_upper_bytes"] = max(counts["max_parent_admission_upper_bytes"], parent)
        counts["max_backing_plus_paged_workspace_upper_bytes"] = max(
            counts["max_backing_plus_paged_workspace_upper_bytes"], storage + scratch)
assert [totals[u]["eligible_single_prompt_joins"] for u in (1, 4, 16, 32)] == [0, 21, 58, 115]
print(json.dumps(dict(patch_sha256=sha256((here/"patch.diff").read_bytes()).hexdigest(),
    server_sha256=wave["digest"](server), client_sha256=wave["digest"](client), rungs=totals,
    full_layers=layers, bytes_per_new_kv_row=row_bytes, budget_bytes=budget,
    assumptions="V41 initial waves replayed with V241 mixed admission, all 16 full layers eligible, native FP8 layout, eight GPUs, full causal width-eight verification. These counts sum repetitions. They represent a removed logical read and write of each prompt row; device memory traffic, end-to-end memory peaks, number of actual joins and timing need hardware measurement.",
    physical_V246=False),indent=2))
