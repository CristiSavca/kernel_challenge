"""Exercise request reordering, retired slots and whole-batch replacement with W + 3 votes."""
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v99_adaptive_draft_tree"))
from lifecycle_cpu import load_methods


def check(source):
    methods = load_methods(source)
    rng = torch.Generator().manual_seed(14257)

    def index(items):
        return torch.tensor(items, dtype=torch.int64)

    def batch(tokens, votes=None):
        value = methods.make_draft_input_v2(bonus_tokens=index(tokens), new_seq_lens=index(tokens)+1)
        value.tree_policy_advantage = votes
        return value

    checks = {"merge_filter": 0, "shuffled_relay_resolve": 0, "reused_slot_reset": 0}
    slots = [3, 1, 4, 2]
    for width in range(2, 9):
        pool = SimpleNamespace(req_to_token=torch.zeros((6, 1), dtype=torch.int64))
        algorithm = SimpleNamespace(is_ngram=lambda: False, is_some=lambda: True)
        relay = methods.FutureMap("cpu", algorithm, pool)
        relay._maybe_init_forward_bufs = lambda _: None
        relay.publish_ready = SimpleNamespace(record=lambda: None)
        votes = torch.randint(-width, width+1, (4, width+3), dtype=torch.int32, generator=rng)
        for has_left, has_right in itertools.product((False, True), repeat=2):
            left = batch([11, 16], votes[:2] if has_left else None)
            right = batch([31, 29], votes[2:] if has_right else None)
            left.merge_batch(right)
            perm = index([3, 1, 2])
            left.filter_batch(perm, perm.tolist())
            if has_left or has_right:
                expected = votes.clone()
                if not has_left:
                    expected[:2].zero_()
                if not has_right:
                    expected[2:].zero_()
                assert torch.equal(left.tree_policy_advantage, expected[perm])
            else:
                assert left.tree_policy_advantage is None
            checks["merge_filter"] += 1
        expected = votes.clone()
        payload = methods.RelayPayload.from_draft_input(batch(slots, votes))
        relay.publish(index(slots), payload.bonus_tokens)
        relay.stash(index(slots), payload)
        votes.fill_(77)
        def recover(order):
            req = batch(slots)
            req.future_indices = index([slots[i] for i in order])
            relay._resolve_spec_extras(SimpleNamespace(spec_info=req))
            return req.tree_policy_advantage
        for order in itertools.permutations(range(len(slots))):
            assert torch.equal(recover(order), expected[index(order)])
            checks["shuffled_relay_resolve"] += 1
        relay.stash(index([slots[1]]), methods.RelayPayload(bonus_tokens=index([99])))
        expected[1].zero_()
        assert torch.equal(recover(range(4)), expected)
        checks["reused_slot_reset"] += 1
    return checks


if __name__ == "__main__":
    print(json.dumps({"torch": torch.__version__, **check(Path(sys.argv[1]))}, indent=2))
