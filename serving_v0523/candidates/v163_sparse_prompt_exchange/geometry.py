"""Apply the grouped-prompt size threshold to archived eight-H200 prompt shapes."""
import ast
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
READER = ROOT / "experiments/iterations/20261118-serving-mixed-graph-capacity.py"


def simulate_exchange(tokens, trials):
    patch = Path(__file__).with_name("patch.diff").read_text()
    marker = "+++ b/python/sglang/srt/layers/muse_spark/sparse_prompt_exchange.py\n"
    hunk = patch.split(marker, 1)[1].split("\ndiff --git ", 1)[0]
    module = ast.parse("\n".join(line[1:] for line in hunk.splitlines() if line.startswith("+")))
    selector = next(n for n in module.body if getattr(n, "name", "") == "_group_width")
    namespace = {}
    exec(compile(ast.Module(body=[selector], type_ignores=[]), "exported_selector", "exec"), namespace)
    group_width = namespace["_group_width"]
    ranks = 8
    shard = tokens // ranks
    rng = np.random.default_rng(163)
    hist = Counter()
    payloads = []
    for _ in range(trials):
        routes = rng.multivariate_hypergeometric([32] * ranks, 8, size=tokens) > 0
        counts = routes.reshape(ranks, shard, ranks).sum(1)
        width = group_width(counts.tolist(), tokens)
        hist[width] += 1
        if width:
            remote = counts.sum() - counts.trace()
            payloads.append(float(remote * 8192 / ranks / 2**20))
    # Opposite extremes also exercise the fallback when its largest peer sets the cost.
    all_owners = np.full((ranks, ranks), shard)
    one_owner = np.zeros_like(all_owners)
    one_owner[:, 0] = shard
    assert group_width(all_owners.tolist(), tokens) == group_width(one_owner.tolist(), tokens) == 0
    return dict(trials=trials, production_destination_width_histogram=sorted(hist.items()),
                mean_remote_MiB_per_rank_on_sparse_rounds=sum(payloads) / len(payloads),
                all_owners_and_one_hot_owner="dense fallback")


def analyze():
    spec = importlib.util.spec_from_file_location("archived_prompts", READER)
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    log = next(trace.ARCHIVE.glob("*v41__server.log")).read_text()
    load = json.loads(next(trace.ARCHIVE.glob("*load.json")).read_text())
    order = [r["concurrency"] for r in load["rungs"] for _ in range(r["reps"])]
    grid = trace.capture_grid(log, r"Capture target prefill CUDA graph begin\. backend=breakable, num_tokens=")
    pad, _ = trace.pinned_selector()
    rows = defaultdict(Counter)
    for concurrency, (_, tokens, _) in trace.read_trace(log, order):
        totals = rows[concurrency]
        totals["all_prompt_calls"] += 1
        totals["live_prompt_rows"] += tokens
        captured = pad(tokens, grid)
        if tokens >= 8192:
            totals["eligible_at_live_shape"] += 1
            totals["eligible_live_rows"] += tokens
        if captured >= 8192:
            totals["eligible_at_captured_shape"] += 1
            totals["eligible_captured_rows"] += captured

    experts, local_experts, topk, ranks = 256, 32, 8, 8
    hit = 1 - math.comb(experts - local_experts, topk) / math.comb(experts, topk)
    layer_rows, hidden_size = 16384, 4096
    gathered = layer_rows * (ranks - 1) / ranks * hidden_size * 2
    saved = gathered * (1 - hit)
    return dict(
        archive=trace.ARCHIVE.relative_to(ROOT).as_posix(),
        archive_substrate="V41, eight H200, SP disabled. Only recorded prompt sizes are reused.",
        source_threshold="V64 grouped route threshold of 256 routes per global expert",
        rows={key:dict(value) for key,value in sorted(rows.items())},
        uniform_model=dict(
            assumptions="Eight distinct expert IDs sampled uniformly per token. All ranks pass the size and scratch gates.",
            expected_unique_owners=hit * ranks,
            full_chunk_baseline_remote_B_per_rank_per_layer=int(gathered),
            full_chunk_expected_sparse_remote_B_per_rank_per_layer=gathered * hit,
            full_chunk_expected_saved_B_per_rank_per_layer=saved,
            full_chunk_metadata_mask_payload_per_gathering_rank_B=(ranks-1)*(layer_rows+1),
            equivalent_time_us_at_assumed_effective_GB_s={bw:saved/(bw*1e9)*1e6 for bw in (150,300,450)},
        ),
        seeded_uniform_split_decisions={n:simulate_exchange(n,64) for n in (8192,16384)},
        hardware_timing=False,
    )


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
