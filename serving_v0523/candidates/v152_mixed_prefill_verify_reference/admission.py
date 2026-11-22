"""Fixed-admission placement bound from the archived self-run V41 public server log."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
archive = ROOT / "experiments/modal_node/20261113T204343Z-bench"
server = next(archive.glob("*v41__server.log"))
load = json.loads(next(archive.glob("*v41__load.json")).read_text())
info = json.loads(next(archive.glob("*v41__server_info.json")).read_text())
spec = info["internal_states"][0]
capacity = min(spec["max_prefill_tokens"],spec["chunked_prefill_size"])
width = spec["speculative_num_draft_tokens"]
timeline = [r["concurrency"] for r in load["rungs"] for _ in range(r["reps"])]
rep = -1
stats = defaultdict(Counter)
for line in server.read_text().splitlines():
    if "Cache flushed successfully!" in line:
        rep += 1
    row = re.search(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+)",line)
    if rep < 0 or row is None:
        continue
    new_reqs,tokens,running = map(int,row.groups())
    st = stats[timeline[rep]]
    st["prefill_steps"] += 1
    st["prompt_rows"] += tokens
    st["current_decode_requests"] += running
    if running:
        st["mixed_opportunities"] += 1
        st["prompt_rows_during_mixed"] += tokens
        available = (capacity-tokens)//width
        assert available >= 0
        st["tail_requests_fitting_unchanged_chunks"] += min(available,running)
        st["steps_fitting_all_without_rechunk"] += available >= running
        st["steps_fitting_some_without_rechunk"] += available > 0
        st["prompt_rows_to_move_for_all_tails"] += max(0,tokens+width*running-capacity)
        st["tail_requests_in_graph_tagged_steps"] += running if "cuda graph: True" in line else 0
assert rep+1 == len(timeline)
for b,st in stats.items():
    outputs=sum(r["completion"] for r in load["requests"] if r["load"]==b)
    st["observed_output_tokens"]=outputs
    st["all_tail_input_rows"]=width*st["current_decode_requests"]
    st["four_commit_percent_of_output_unchanged_chunks"]=round(400*st["tail_requests_fitting_unchanged_chunks"]/outputs,2)
    st["four_commit_percent_of_output_rechunked"]=round(400*st["current_decode_requests"]/outputs,2)
print(json.dumps(dict(parent="V41 historical isolated H200 prefill admissions with mixed disabled",
      cap=capacity,width=width,by_concurrency=stats,
      proviso="Counterfactual fixed arrivals, prefill admissions and running membership, no optimizer or timing model. Some free row slots could still be ineligible for memory, finishing requests or graph constraints."),indent=2))
