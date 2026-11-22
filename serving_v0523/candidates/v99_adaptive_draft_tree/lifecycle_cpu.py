"""Run V99's real DFLASH state/relay methods on PyTorch CPU, with minimal runtime dependencies."""
from __future__ import annotations

import ast
import itertools
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


class SpecInput:
    dsa_topk_indices = None
    future_dsa_topk_indices_available = False

    def __init__(self, spec_input_type):
        self.spec_input_type = spec_input_type


def load_methods(root):
    scope = dict(
        torch=torch,
        dataclass=dataclass,
        SpecInput=SpecInput,
        SpecInputType=SimpleNamespace(DFLASH_DRAFT=1),
        ConfidenceRelay=SimpleNamespace,
        _DEBUG_ASSERT=False,
        _offered_relay=None,
        _is_cuda=False,
        __name__=__name__,
    )
    for file, names in [
        ('speculative/dflash_info_v2.py', {'DFlashDraftInputV2'}),
        ('speculative/draft_worker_common.py', {'make_draft_input_v2'}),
        ('managers/overlap_utils.py', {'RelayPayload', 'FutureMap'}),
    ]:
        path = root / 'python/sglang/srt' / file
        parsed = ast.parse(path.read_text(), filename=str(path))
        nodes = [n for n in parsed.body if getattr(n, 'name', '') in names]
        unit = ast.parse('from __future__ import annotations')
        unit.body.extend(nodes)
        exec(compile(unit, str(path), 'exec'), scope)
    return SimpleNamespace(**{
        name: scope[name]
        for name in ('DFlashDraftInputV2', 'make_draft_input_v2', 'RelayPayload', 'FutureMap')
    })


def run(root):
    src = load_methods(root)
    rng = random.Random(29991)
    counts = Counter()

    def idx(rows):
        return torch.tensor(rows, dtype=torch.int64)

    def draft(rows, values=None):
        ids = idx(rows)
        item = src.make_draft_input_v2(bonus_tokens=ids, new_seq_lens=ids + 120)
        if values is not None:
            item.tree_policy_advantage = torch.tensor(values, dtype=torch.int32)
        return item

    def as_list(values):
        return None if values is None else values.tolist()

    for a, b in itertools.product((0, 1, 3, 8), repeat=2):
        for left_enabled, right_enabled in itertools.product((True, False), repeat=2):
            left_values = [17 - i for i in range(a)] if left_enabled else None
            right_values = [-23 + i for i in range(b)] if right_enabled else None
            left = draft(range(a), left_values)
            right = draft(range(50, 50 + b), right_values)
            expected = None
            if left_enabled or right_enabled:
                expected = (left_values or [0] * a) + (right_values or [0] * b)
            left.merge_batch(right)
            assert as_list(left.tree_policy_advantage) == expected
            counts['direct_merges'] += 1
            kept = rng.sample(range(a + b), rng.randint(0, a + b))
            left.filter_batch(idx(kept), kept)
            assert as_list(left.tree_policy_advantage) == (
                None if expected is None else [expected[i] for i in kept]
            )
            counts['direct_filters'] += 1

    algo = SimpleNamespace(is_ngram=lambda: False, is_some=lambda: True)
    pool = SimpleNamespace(req_to_token=torch.empty((66, 1), dtype=torch.int64))

    def relay():
        m = src.FutureMap('cpu', algo, pool)
        # The omitted stock routine only initializes Eagle top-k/hidden tensors.
        m._maybe_init_forward_bufs = lambda _payload: None
        m.publish_ready = SimpleNamespace(record=lambda: None)
        return m

    def resolved(m, slots):
        s = src.DFlashDraftInputV2.create_idle_input(torch.device('cpu'))
        s.future_indices = idx(slots)
        m._resolve_spec_extras(SimpleNamespace(spec_info=s))
        return s

    owner, peer = relay(), relay()
    expected = [0] * 66
    bonuses = [-1] * 66
    owner.stash(idx([]), src.RelayPayload(bonus_tokens=idx([])))
    assert owner.tree_policy_advantage_buf is None
    assert resolved(owner, []).tree_policy_advantage is None
    counts['idle'] += 1
    for step in range(160):
        slots = rng.sample(range(1, 66), rng.randint(1, 32))
        old = None if owner.tree_policy_advantage_buf is None else owner.tree_policy_advantage_buf.clone()
        token_ids = [60000 + step * 70 + s for s in slots]
        vals = None if step % 3 == 0 else [rng.randint(-8, 8) for _ in slots]
        # New seq lengths may be available early. Request history only becomes
        # visible after the matching bonus payload is stashed on the forward stream.
        owner.publish(idx(slots), idx(token_ids))
        assert old is None or torch.equal(owner.tree_policy_advantage_buf, old)
        counts['early_publishes'] += 1
        packet = src.RelayPayload.from_draft_input(draft(token_ids, vals))
        owner.stash(idx(slots), packet)
        peer.stash(idx(slots), src.RelayPayload.from_draft_input(draft(token_ids)))
        assert peer.tree_policy_advantage_buf is None
        counts['nonowner_or_disabled'] += 1
        for i, s in enumerate(slots):
            expected[s] = vals[i] if vals is not None else 0
            bonuses[s] = token_ids[i]
        poison = packet.tree_policy_advantage
        if poison is not None:
            poison.fill_(1007)
        eligible = [s for s in range(1, 66) if bonuses[s] >= 0]
        shuffled = rng.sample(eligible, rng.randint(1, len(eligible)))
        part = rng.randint(0, len(shuffled))
        handles = []
        for section in (shuffled[:part], shuffled[part:]):
            handle = draft(section, [99] * len(section))
            handle.future_indices = idx(section)
            reverse = list(range(len(section) - 1, -1, -1))
            handle.filter_batch(idx(reverse), reverse)
            assert handle.tree_policy_advantage is None
            handles.append(handle)
        handles[0].merge_batch(handles[1])
        merged_slots = shuffled[:part][::-1] + shuffled[part:][::-1]
        owner._resolve_spec_extras(SimpleNamespace(spec_info=handles[0]))
        assert handles[0].bonus_tokens.tolist() == [bonuses[s] for s in merged_slots]
        expected_rows = None if step == 0 else [expected[s] for s in merged_slots]
        assert as_list(handles[0].tree_policy_advantage) == expected_rows
        if step:
            assert owner.tree_policy_advantage_buf[0] == 0
        counts['poison_reorder_resolve'] += 1

    # A disaggregated/retracted request admitted under a reused physical slot
    # starts neutral. Its previous occupant's nearby slots retain their values.
    frozen = owner.tree_policy_advantage_buf.clone()
    owner.stash(idx([11]), src.RelayPayload(bonus_tokens=idx([201])))
    frozen[11] = 0
    assert torch.equal(owner.tree_policy_advantage_buf, frozen)
    assert resolved(owner, [11]).tree_policy_advantage.tolist() == [0]
    counts['explicit_reused_slot_resets'] += 1
    return {'torch': torch.__version__, 'device': 'cpu', 'cases': dict(counts)}


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
