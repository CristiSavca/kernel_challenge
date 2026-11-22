"""Relate the native V202 and V203 head tensor shapes to archived mixed opportunities."""
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
budget = json.loads(next(HERE.parent.glob("v202_*/phase_budget.json")).read_bytes())
tp, vocab, width = (budget[k] for k in ("tp","vocab","width"))
assert vocab%tp == 0
loads = {}
for key, load in budget["loads"].items():
    events = load["paper_schedule"]
    chain_rows = width*events["tail_appearances"]
    total_rows = chain_rows+events["prompt_requests"]
    assert total_rows == events["head_rows"]
    d = dict(mixed_calls=events["calls"], verified_vocabulary_rows=chain_rows,
             full_vocab_prompt_rows=events["prompt_requests"],
             full_FP32_tail_output_bytes_avoided=4*chain_rows*vocab,
             gathered_tail_pair_bytes=4*2*tp*chain_rows,
             total_local_FP32_lm_scores=total_rows*(vocab//tp))
    loads[key]=d
    if not total_rows:continue
    n=events["rows_per_call"]["max"]
    # At these observed maxima the head row count and concurrency admit
    # exactly one decomposition into whole prompts and complete verify chains.
    opts=[(n-width*t,t) for t in range(1,int(key))
          if n-width*t>=1 and n-width*t+t<=int(key)]
    assert len(opts)==1
    p,t=opts[0]
    r=t*width
    local=n*(vocab//tp)*4
    parent_gather=n*vocab*4
    prompt=p*vocab*4
    # Count just the native local logits, collective destinations and capped
    # output. Native argmax adds another set of small index tensors.
    d["largest_observed_head"] = dict(prompts=p,running_requests=t,all_output_rows=n,
        local_matmul_bytes_both=local,V202_rank_major_gather_bytes=parent_gather,
        V202_full_softcap_result_bytes=parent_gather,V203_prompt_rank_major_gather_bytes=prompt,
        V203_prompt_softcap_result_bytes=prompt,V203_tail_pair_gather_bytes=r*tp*8,
        V203_local_tail_pair_bytes=r*8)
print(json.dumps(dict(loads=loads,measured_gpu_time=False,
    scope="Materialized tensor sizes per rank. Allocator peak, graph buffers, collective transport and throughput require hardware."),indent=2))
