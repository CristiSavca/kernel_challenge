"""Use measured V68 slot counts with the five-layer BF16 draft geometry."""
import hashlib
import json
from pathlib import Path
import re

here = Path(__file__).resolve().parent
service = here.parents[1]
log = service / "runs/04b01f958aa5487c.log"
args_log = next((service.parent / "experiments/modal_node/20261113T204343Z-bench").glob("*__v41__server_info.json"))
args = json.loads(args_log.read_bytes())["internal_states"][0]
text = log.read_text()
full, sliding = map(int, re.search(r"full_layer_tokens=(\d+), swa_layer_tokens=(\d+)",text).groups())
draft = int(re.search(r"draft KV cache allocated. dtype: torch.bfloat16, #tokens: (\d+)",text)[1])
host_full = int(re.search(r"Allocating draft hierarchical KV host pool: (\d+) tokens",text)[1])
host_swa = int(re.search(r"Allocating swa hierarchical KV host pool: (\d+) tokens",text)[1])
batch_max = int(re.search(r"Capture draft verify CUDA graph end. bs=\[[0-9, ]+, (\d+)\]",text)[1])
context = int(re.search(r"max_running_requests=\d+, context_len=(\d+)",text)[1])
assert draft == full and args["context_length"] == context and args["page_size"] == 1
assert args["speculative_draft_attention_backend"] == "fa4" and args["enable_hierarchical_cache"]
row = 5*2*1*128*2
assert abs(draft * row / (1<<30) - 8.16)<0.02  # VMM and display rounding
assert round(host_full * row / 1e9,2)==58.18
graph_metadata = 3*batch_max*context*4 + batch_max*args["speculative_num_draft_tokens"]*8
out = dict(patch_sha256=json.loads((here/"validation.json").read_bytes())["patch_sha256"],
    original_H200_job="04b01f958aa5487c",pool_log=str(log.relative_to(service.parent)),
    pool_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    comparison_configuration=str(args_log.relative_to(service.parent)),
    comparison_configuration_sha256=hashlib.sha256(args_log.read_bytes()).hexdigest(),
    archived_args={k:args[k] for k in ("context_length","page_size","attention_backend",
        "speculative_draft_attention_backend","speculative_num_draft_tokens","enable_hierarchical_cache",
        "hicache_mem_layout","hicache_io_backend","hicache_storage_backend","enable_unified_memory")},
    draft_slots=draft,full_slots=full,swa_slots=sliding,full_host_slots=host_full,swa_host_slots=host_swa,
    bytes_per_draft_row=row,fixed_slot_device_payload_bytes_saved=(draft-sliding)*row,
    approximate_fixed_slot_host_payload_bytes_saved=(host_full-host_swa)*row,
    comparison_draft_batch_max=batch_max,native_fa_extra_graph_bytes_at_these_dimensions=graph_metadata,
    caveat="Host bytes count logical sidecar payload only; allocator rounding, pinned/shared host backing, resident memory, graph allocation and post-capture slot counts depend on deployment. No prospective V249 hardware measurement.",
    scope="The three extra graph page tables are one each for decode, verify and draft-extend in one native FA backend; one int64 write-location buffer has B times speculative width entries.")
(here/"capacity.json").write_text(json.dumps(out,indent=2)+"\n")
