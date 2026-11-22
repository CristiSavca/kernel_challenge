"""Map V211's long-prompt work to the team's archived V41 scheduling chronology."""
from collections import Counter
from contextlib import redirect_stdout
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
archive = ROOT / "experiments/modal_node/20261113T204343Z-bench"
server, = archive.glob("*v41__server.log")
client, = archive.glob("*load.json")
helper = runpy.run_path(str(ROOT / "experiments/iterations/20261118-serving-mixed-graph-capacity.py"))
with redirect_stdout(io.StringIO()):
    model = runpy.run_path(str(next(HERE.parent.glob("v208_*/work.py"))))
estimate = lru_cache(None)(model["estimate"])
loads = [r["concurrency"] for r in json.loads(client.read_bytes())["rungs"] for _ in range(r["reps"])]
histograms = {str(u): Counter() for u in loads}
for load, (_, rows, _) in helper["read_trace"](server.read_text(), loads):
    if rows >= 8192:
        histograms[str(load)][rows] += 1
results = {}
reference = json.loads(next(HERE.parent.glob("v197_*/independent_reach.json")).read_bytes())["archive"]
for load, hist in histograms.items():
    expected = [x for x in reference if x["users"] == int(load)]
    rows = sum(q*n for q,n in hist.items())
    assert rows == sum(x["eligible_logical_rows"] for x in expected)
    assert hist.total() == sum(x["logical_grouped_passes"] for x in expected)
    before = sum(estimate(q)["prior_MMA_rows"] * n for q,n in hist.items())
    after = sum(estimate(q)["new_MMA_rows"] * n for q,n in hist.items())
    results[load] = dict(repetitions=len(expected), eligible_batches=hist.total(), eligible_input_rows=rows,
        full_16384_row_batches=hist[16384], tensor_rows_prior_per_layer=before, tensor_rows_new_per_layer=after,
        percent_fewer_tensor_rows=100*(1-after/before),
        rank_average_BF16_scale_read_write_bytes_per_layer=rows*4096*4)
print(json.dumps(dict(source=str(server.relative_to(ROOT)), source_sha256=sha256(server.read_bytes()).hexdigest(),
    assumptions="V41 schedule; logical live input lengths, on eligible compact grouped W13/W2 with all fast paths selected. "
                "Top eight uniform distinct experts of 256 per token, 32 local experts. All repetitions at each load combined. "
                "Physical routing, capture padding, fallback selection and service time can differ.",
    by_concurrency=results, physical_V211_measurement=False), indent=2))
