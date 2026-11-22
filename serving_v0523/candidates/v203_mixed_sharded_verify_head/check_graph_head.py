"""Pass the explicit sharded-vocabulary request through the model and eager graph tail."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch


HERE = Path(__file__).resolve().parent
suite = next(HERE.parent.glob("v155_*/worker_cpu.py"))
spec = importlib.util.spec_from_file_location("head_graph_source",suite)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
ref.env["project_dflash_prefill_shard"] = lambda *args: None
counts = dict(local=0,full=0,normal_prefill=0,retained_tap_words=0)
with torch.inference_mode():
    for backend in ("eager","FULL","BREAKABLE"):
        for size in (4,8):
            ref.PAR.tp_group = NS(world_size=size)
            for mode in ("shard","opt_out","ordinary","equal","oversized","undersized","tiny_cap","infinite_cap"):
                worker, calls, graph = ref.test_worker(backend,128)
                runner = worker.model_runner
                proc = runner.model.logits_processor.__self__
                group_rank = size-2
                shard = (ref.V+size-1)//size
                cap = {"tiny_cap":1e-55, "infinite_cap":float("inf")}.get(mode,25.)
                width = {"equal":ref.V, "oversized":ref.V+2, "undersized":shard-1}.get(mode,shard)
                proc.do_tensor_parallel_all_gather = True
                proc.defer_verify_gather = True
                proc.final_logit_softcapping = cap
                # The hardware precondition is supplied on this CPU fixture. All
                # subsequent mixed dispatch and graph/worker shape checks are real.
                proc._can_gather_softcap_fused = lambda *_: True
                memory = {}
                def project(h, w, embedding_bias, *, out):
                    full = h.float() @ w.T
                    memory["reference"] = full * proc.logit_scale
                    local = torch.zeros(len(h),width)
                    begin = group_rank * width
                    if begin < ref.V:
                        live = min(width, ref.V-begin)
                        local[:,:live].copy_(full[:,begin:begin+live])
                    memory["local"] = local.clone() * proc.logit_scale
                    return local
                def gather_softcap(local, meta, use_buffer):
                    memory["gather"] = True
                    return torch.tanh(memory["reference"]/cap)*cap
                proc._compute_lm_head = project
                proc._gather_softcap_fused = gather_softcap
                lengths = [11,3,8,8]
                batch = ref.schedule(lengths)
                count = None if mode=="ordinary" else 2
                result = worker.forward_batch_generation(batch,is_verify=True,capture_hidden_mode=ref.Capture.FULL,
                    return_kv_loc_plan=True,mixed_verify_prompt_count=count,
                    mixed_verify_sharded_logits=mode!="opt_out")
                out = result.logits_output
                limited = mode=="shard"
                expect = memory["local"] if limited else torch.tanh(memory["reference"]/cap)*cap
                torch.testing.assert_close(out.next_token_logits, expect, equal_nan=True, rtol=0,atol=0)
                assert ("gather" not in memory) == limited
                assert calls[-1].mixed_verify_sharded_logits == (count is not None and mode!="opt_out")
                assert out.next_token_logits.shape[0] == (4 if count is None else 18)
                taps = torch.cat(ref.body_tokens(batch.input_ids)[1],dim=1)
                assert torch.equal(out.hidden_states,taps)
                counts["local" if limited else "full"]+=1
                counts["normal_prefill"]+=int(count is None)
                counts["retained_tap_words"]+=taps.numel()
print(json.dumps(dict(**counts,backends=["eager","Full","Breakable"],physical_h200=False),indent=2))
