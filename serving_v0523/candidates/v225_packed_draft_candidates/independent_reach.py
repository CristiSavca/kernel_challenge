"""Follow selector replay and eager paths and check against the two archived judge logs."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import torch
from independent_transport import DIR, MODEL, PARALLEL, bind, source, util


WORKER = "speculative/dflash_worker_v2.py"
COMMON = "speculative/draft_worker_common.py"
GRAPH = "model_executor/runner/decode_cuda_graph_runner.py"
VOCAB = "layers/vocab_parallel_embedding.py"


def assert_method(path, cls, method, *snippets):
    raw = source(path)
    node = ast.parse(raw)
    for part in (cls, method):
        if part:
            node = next(n for n in node.body if getattr(n, "name", None) == part)
    result = []
    for snippet in snippets:
        assert snippet in ast.get_source_segment(raw, node), (path, cls, method, snippet)
        total = raw.splitlines()
        number = next(i for i in range(node.lineno, node.end_lineno+1) if snippet in total[i-1])
        result.append({"file": util.SRT + path, "line": number, "statement": snippet})
    return result


def hook_chain():
    # Hook, sampler, model entry, eager fallback and their static buffer writes execute.
    # The terminal selector copies the first ID; candidate_topk is checked independently.
    seen = []
    shard_starts = torch.arange(8, dtype=torch.int64) * 25256
    def candidate_topk(hidden, head, k, *, candidate_shard_starts=None):
        assert head is shared_head
        seen.append((candidate_shard_starts, hidden.clone()))
        ids = hidden[:, :1].long() + torch.arange(k)[None, :]
        return ids, ids.float(), None
    selector = NS(top_k=16, build_lattice=lambda **p: p["unary_logits"],
                  sample_path=lambda **p: (p["candidate_ids"][:,:,0], p["scores"]))
    draft_type = type("Draft", (), {})
    setattr(draft_type, "compute_candidates", bind(MODEL, "compute_candidates", {"candidate_topk": candidate_topk},
        cls="DFlash2DraftModel"))
    setattr(draft_type, "_transform_unary_logits", bind(MODEL, "_transform_unary_logits", {"torch": torch},
        cls="DFlash2DraftModel"))
    draft = draft_type()
    shared_head = object()
    draft.lm_head = shared_head
    draft.candidate_selector = selector
    draft.draft_config = NS(output_multiplier=1.0, final_logit_softcapping=None)
    env = {"torch": torch, "_selector_lattice": bind(WORKER, "_selector_lattice", {})}
    sampler_type = type("Sampler", (), {})
    for name in ("__init__", "__call__"):
        setattr(sampler_type, name, bind(WORKER, name, env, cls="_SelectorDraftSampler"))
    sampler = sampler_type(draft_model=draft, block_size=8, max_bs=32, device="cpu",
                           sampling_enabled=False, shard_starts=shard_starts)
    captures = bind(COMMON, "make_draft_sampler_capture_hook", {"LogitsProcessorOutput": NS})(sampler)
    eager = bind(WORKER, "_propose_selector_block", env, cls="DFlashWorkerV2")
    worker = NS(draft_model=draft, block_size=8, selector=selector, _selector_sampling_enabled=False)
    cases = []
    for bs in (1,4,16,32):
        hidden = torch.arange(bs*8*4).reshape(bs*8,4)
        ids = torch.arange(bs*8) + 10000
        output = NS(hidden_states=hidden)
        anchors = ids[::8]
        captures(None, output, NS(input_ids=ids), ids.numel())
        captured_starts, captured_hidden = seen.pop()
        assert captured_starts is shard_starts
        assert torch.equal(captured_hidden, hidden.reshape(bs,8,4)[:,1:].reshape(-1,4))
        returned = eager(worker, draft_logits_output=output, bs=bs, lm_head=shared_head,
                         anchor_token_ids=anchors, sampling_info=None)
        eager_starts, eager_hidden = seen.pop()
        assert eager_starts is None and torch.equal(eager_hidden, captured_hidden)
        assert torch.equal(returned, sampler.out[:bs*7].reshape(bs,7))
        assert torch.equal(sampler.candidate_out[:bs], sampler.q_out[:bs].long())
        cases.append({"batch": bs, "selector_rows": captured_hidden.shape[0], "candidate_words": bs*7*16})
    return cases


def run():
    references = []
    for args in [
        (WORKER,"DFlashWorkerV2","init_cuda_graphs","self._draft_sampler = self._maybe_build_draft_sampler()",
         "make_draft_sampler_capture_hook(self._draft_sampler)","self._draft_worker.init_cuda_graphs("),
        (WORKER,"DFlashWorkerV2","_maybe_build_draft_sampler","_selector_shard_starts(self.draft_model, lm_head)",
         "self.draft_model.lm_head = lm_head","shard_starts=shard_starts"),
        (WORKER,None,"_selector_shard_starts","group = get_parallel().tp_group","torch.distributed.get_backend(group.cpu_group)",
         "DFlash2DraftModel.compute_candidates","weight.dtype == torch.bfloat16","0 < k <= size <= 65536",
         "group.all_gather_object(","torch.tensor([spec[1]"),
        (GRAPH,"DecodeCudaGraphRunner","capture_one_shape","for capture_hook in self.model_runner.capture_tail_hooks:",
         "capture_hook(self, out, forward_batch, num_tokens)"),
        (WORKER,"DFlashWorkerV2","propose_decode_block","self._draft_sampler.stage_sampling_params(",
         "draft_out = self.draft_model_runner.forward(forward_batch)","elif self._draft_sampler is not None and draft_out.can_run_graph:",
         "self._draft_sampler.candidate_out[:bs]","self._draft_sampler.q_out[:bs]","draft_next = self._propose_selector_block("),
        (PARALLEL,"GroupCoordinator","__init__",'backend="gloo"'),
        (PARALLEL,"GroupCoordinator","all_gather_object","group=self.cpu_group"),
        (PARALLEL,None,"get_tp_group","return get_parallel().tp_group"),
        ("distributed/communication_op.py",None,"tensor_model_parallel_all_gather","return get_tp_group().all_gather(input_, dim)"),
        (PARALLEL,"GroupCoordinator","all_gather","self.all_gather_into_tensor(output_tensor, input_)","output_tensor.movedim(0, dim)"),
        (PARALLEL,"GroupCoordinator","_all_gather_into_tensor","pynccl_comm.all_gather("),
        ("distributed/device_communicators/pynccl_wrapper.py","ncclDataTypeEnum","from_torch","return cls.ncclInt32"),
        (MODEL,None,"_radix_topk","_flashinfer_top_k(scores, k, sorted=True, deterministic=True)"),
        (MODEL,None,"candidate_topk","candidate_shard_starts is not None and not with_partition","tensor_model_parallel_all_gather(vals.float()"),
        ("models/muse_spark.py","MuseSparkTextForCausalLM","__init__","self.lm_head = ParallelLMHead(","quant_config=None"),
    ]:
        references.extend(assert_method(*args))
    logs = []
    for version, job in (("V64","ac89fddc4ee147ab"),("V68","04b01f958aa5487c")):
        path = DIR.parents[1] / "runs" / (job+".log")
        data = path.read_bytes()
        lines = data.decode().splitlines()
        line = next(i for i,s in enumerate(lines,1) if "DFLASH selector decode folded into the draft cuda graph (sampling_enabled=True)" in s)
        logs.append({"candidate": version, "path": "../../runs/"+path.name, "sha256": sha256(data).hexdigest(),
                     "line": line, "folded_selector": True})
    requirements = subprocess.check_output(["git","-C",util.BASE,"show",util.PIN+":python/pyproject.toml"]).decode()
    assert 'flashinfer_python[cu13]==0.7.2' in requirements
    return {"passed":True,"patch_sha256":sha256((DIR/"patch.diff").read_bytes()).hexdigest(),
            "paired_cpu_hook_and_eager_runs":hook_chain(),"source_trace":references,"judge_observations":logs,
            "flashinfer_radix_source":"https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/topk.py#L962-L997",
            "flashinfer_contract":"Pinned deterministic radix path allocates top-k values with input.dtype and returns int64 local indices.",
            "limitation":"Archived jobs establish capture for their respective parents. New CUDA capture, NCCL latency and H200 speed remain unmeasured."}


if __name__ == "__main__":
    result = run()
    (DIR/"independent_reach.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"passed": True, "source_checks":len(result["source_trace"]),
                      "hook_cases":len(result["paired_cpu_hook_and_eager_runs"])}))
