"""How much isolated projection time is available to pay for a TP8 output gather?"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
proj = ROOT / "experiments/runs/b5538a0dd25d46be.json"
serve = ROOT / "serving_v0523/runs/abd217baf0ed49fd.json"
local = json.loads(proj.read_bytes())["response"]["result"]["records"]
times = {(x["case"]["m"],x["case"]["layer"]): x["baseline_raw_ns"]/1000
         for x in local if x["event"] == "benchmark"}
tp = 8
layers = 62
answer = []
for a in json.loads(serve.read_bytes())["result"]["rungs"]:
    users = a["concurrency"]
    m = 8 * users
    # Native SM90 can already shard an unpool of at most eight rows on eligible multicast storage.
    projections = [(4096,8192)] if m <= 8 else [(4096,8192),(8192,4096)]
    saved_matrix = sum(n*k*2*(tp-1)//tp for n,k in projections)
    gather_bytes = sum(m*n*2*(tp-1)//tp for n,k in projections)
    pool,router,unpool = [times[m,k] for k in ("pooler","router","unpooler")]
    baseline = pool + (unpool if len(projections)==2 else 0)
    # Treat these separate task times as serial with serving. Overlap and a changed input
    # distribution can make even this very favorable sensitivity overstate attainable gain.
    period_ms = 1000 * a["acceptance"] / a["tok_s_per_user_p50"]
    drift = 0.0163
    break_even_us = baseline - (period_ms * 1000 / layers) * (1-1/(1+drift))
    answer.append(dict(users=users,rows=m,nominal_matrix_reads_avoided_bytes_per_rank_layer=saved_matrix,
                   nominal_matrix_reads_avoided_GiB_per_rank_pass=layers*saved_matrix/(2**30),
                   additional_logical_incoming_ring_bytes_per_rank_pass=layers*gather_bytes,
                   pool_us=pool,router_us=router,unpool_us=unpool,
                   v64_approximate_verify_period_ms=period_ms,
                   isolated_changed_projection_budget_us_per_layer=baseline,
                   replacement_budget_us_per_layer_for_1_63pct_verify_gain=break_even_us,
                   illustrative_verify_ratio_if_changed_projection_time_were_zero=period_ms/(period_ms-baseline*layers/1000)))
print(json.dumps(dict(hardware_projector_job="b5538a0dd25d46be",hardware_serving_job="abd217baf0ed49fd",
    hardware_projection_receipt_sha256=hashlib.sha256(proj.read_bytes()).hexdigest(),
    hardware_serving_receipt_sha256=hashlib.sha256(serve.read_bytes()).hexdigest(),
    sparse_layers=layers,tp=tp,rungs=answer,
    scope="Nominal BF16 bytes; assumes eligible SM90 multicast at eight rows. Timings are an isolated, single-rank H200 kernel-board baseline and a different serving receipt. Includes no measured projection/gather overlap or V228 wall time."),indent=2))
