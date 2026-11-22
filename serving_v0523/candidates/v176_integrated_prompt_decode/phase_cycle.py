"""Run successive SP prompts and verifications through the same native model/logits objects."""
from copy import copy
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch

import torch


def verify_cycle(stage, owner, model_forward, prompt_batch, prompt_mode, features, hidden, group):
    """Verify with fresh lengths before and after a failure inside the decoder."""
    here = Path(__file__).resolve().parent
    schedule = runpy.run_path(str(here.parent/"v132_verify_scheduler_metadata/verify_dispatch.py"))
    backend, _, _, trace = schedule["build"](stage)
    backend.page_size = 1
    md, geometry, layer, q, cache, _ = schedule["tensors"](2, 8, 1, 131072, 16, 2)
    cache = tuple(tensor.unsqueeze(0) for tensor in cache)
    batch = copy(prompt_batch)
    outer = {"caller": object()}
    batch._fa3_verify_schedules = outer
    batch.spec_info = geometry.spec_info
    batch.draft_extend_select_index = None
    batch.forward_mode = SimpleNamespace(**(vars(prompt_mode) | {"is_target_verify": lambda: True}))
    live_hidden, live_features = hidden[:16], features[:16]
    ids = torch.arange(16)
    expected_logits = torch.mm(live_hidden.float(), owner.lm_head.weight.T.float())
    original_body = owner.model
    counters = {"verification_calls": 0, "exceptions": 0, "layer_calls": 0, "preparations": 0}
    group.gathers.clear()
    try:
        with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=7)), patch.object(
            torch.cuda, "current_device", return_value=0
        ):
            for fail in (False, True, False):
                counters["verification_calls"] += 1
                md.cache_seqlens_int32.add_(3)
                before = len(trace.prepare)

                def body(input_ids, positions, fb, **kwargs):
                    assert fb is batch and fb._fa3_verify_schedules == {}
                    store = fb._fa3_verify_schedules
                    for i in range(62):
                        window = (-1, -1) if i % 4 == 0 else (2047, 0)
                        plan = backend._verify_schedule(store, q, *cache, md.page_table,
                            md.cache_seqlens_int32, md.cu_seqlens_q, 8, layer, window, batch)
                        counters["layer_calls"] += 1
                        assert plan.live[0] == tuple(md.cache_seqlens_int32.tolist())
                        if fail and i == 3:
                            raise ArithmeticError("test exception inside decoder")
                    assert len(store) == 2
                    return live_hidden, live_features

                owner.model = body
                if fail:
                    try:
                        model_forward(owner, ids, None, batch)
                    except ArithmeticError as exc:
                        counters["exceptions"] += 1
                        assert str(exc) == "test exception inside decoder"
                    else:
                        raise AssertionError("decoder exception was swallowed")
                else:
                    out = model_forward(owner, ids, None, batch)
                    assert out.dflash_prefill_context_done is False
                    assert torch.equal(out.next_token_logits, expected_logits)
                    assert torch.equal(out.hidden_states, live_features)
                assert len(trace.prepare) == before + 2
                counters["preparations"] += len(trace.prepare) - before
                assert batch._fa3_verify_schedules is outer and tuple(outer) == ("caller",)
                assert group.gathers == []
    finally:
        owner.model = original_body
    assert counters == {"verification_calls": 3, "exceptions": 1, "layer_calls": 128, "preparations": 6}


# Retain all real V171 projection/logit setup and Gloo edge cases in the composed tree.
# Insert the phase change after an actual ordinary prompt at each of three threshold geometries.
here = Path(__file__).resolve().parent
fixture = here.parent/"v171_prompt_endpoint_context/check_prompt.py"
source = fixture.read_text()
edits = {
    "    model = nn.Module()": "    cycle_count = 0\n    model = nn.Module()\n    model._share_verify_schedule = True",
    "        assert calls == 108": "        assert calls == 108 and cycle_count == 3",
    "                            calls += 1": """                            if scenario == "ordinary" and ids is not None and context is handle:
                                verify_cycle(stage, model, forward, batch, mode, dense, hidden, group)
                                cycle_count += 1
                            calls += 1""",
    '"prompt_forwards": 324,': '''"prompt_forwards": 324, "paired_phase_cycles": 9,
                      "additional_verification_calls": 27, "injected_body_exceptions": 9,
                      "scheduled_layer_calls": 1152, "fresh_full_and_sliding_preparations": 54,
                      "reference_scope": "pinned model, schedule selection and native prompt/logits methods with Gloo; FA3 builder on CPU",''',
}
for old, new in edits.items():
    assert source.count(old) == 1
    source = source.replace(old, new)
exec(compile(source, str(fixture), "exec"), globals())
