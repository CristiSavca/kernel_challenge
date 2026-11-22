"""Exercise the unchanged V64 capture hook and greedy sampler entry on CPU."""

import argparse
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree"))
import cpu_check as fx


def run(checkout):
    src = checkout / "python/sglang/srt"
    worker = src / "speculative/dflash_worker_v2.py"
    calls = []
    head = object()
    spec = SimpleNamespace(speculative_dflash_fused_candidate_head=False)

    def candidate_topk(hidden, lm_head, k, *, fuse_candidate_head):
        calls.append((hidden.clone(), k, fuse_candidate_head))
        assert lm_head is head
        ids = torch.arange(k).unsqueeze(0).expand(len(hidden), -1)
        values = (k - ids).float()
        return ids, values, None

    def sample_path(*, candidate_ids, scores, uniforms, temperatures, greedy_mask):
        assert greedy_mask.all() and candidate_ids.shape == scores.shape
        tokens = candidate_ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        return tokens, scores.softmax(-1)

    _, (compute,) = fx.program(src / "models/dflash.py", ("compute_candidates",), "DFlash2DraftModel",
                               {"get_spec": lambda: spec, "candidate_topk": candidate_topk})
    selector = SimpleNamespace(top_k=16, build_lattice=lambda **kw: kw["unary_logits"], sample_path=sample_path)
    model = SimpleNamespace(lm_head=head, candidate_selector=selector, _transform_unary_logits=lambda x: x)
    model.compute_candidates = MethodType(compute, model)
    _, (lattice, Sampler) = fx.program(worker, ("_selector_lattice", "_SelectorDraftSampler"),
                                       dependencies={"torch": torch})

    class Output:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    _, (capture_hook,) = fx.program(src / "speculative/draft_worker_common.py", ("make_draft_sampler_capture_hook",),
                                    dependencies={"LogitsProcessorOutput": Output})
    sampler = Sampler(draft_model=model, block_size=8, max_bs=32, device="cpu", sampling_enabled=True)
    hook = capture_hook(sampler)
    for enabled in (False, True):
        spec.speculative_dflash_fused_candidate_head = enabled
        for batch in (1, 4, 16, 32):
            hidden = torch.arange(batch * 8 * 3).reshape(batch * 8, 3).float()
            forward = SimpleNamespace(input_ids=torch.arange(batch * 8))
            sampler.stage_sampling_params(bs=batch, sampling_info=None)
            result = hook(None, Output(hidden), forward, batch * 8)
            seen, k, flag = calls[-1]
            assert torch.equal(seen, hidden.view(batch, 8, 3)[:, 1:].reshape(-1, 3))
            assert k == 16 and flag is enabled
            assert sampler.out[: batch * 7].eq(0).all()
    assert len(calls) == 8
    return len(calls)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkout", type=Path)
    print(f"{run(p.parse_args().checkout)} native greedy capture-hook dispatch cases")
