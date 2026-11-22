"""Independently exercise V101 rank-local relay and adjacent-policy identification."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch


def check_relay(root: Path):
    inherited = Path(__file__).resolve().parents[1] / 'v99_adaptive_draft_tree/lifecycle_cpu.py'
    spec = importlib.util.spec_from_file_location('v99_lifecycle', inherited)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    src = module.load_methods(root)
    counts = Counter()
    rng = random.Random(497311)
    algorithm = SimpleNamespace(is_ngram=lambda: False, is_some=lambda: True)
    pool = SimpleNamespace(req_to_token=torch.zeros((37, 2), dtype=torch.int64))

    def ids(values):
        return torch.tensor(values, dtype=torch.int64)

    def draft(keys, margins=None):
        k = ids(keys)
        state = src.make_draft_input_v2(bonus_tokens=k, new_seq_lens=k + 1)
        state.tree_policy_advantage = margins
        return state

    def map_():
        result = src.FutureMap('cpu', algorithm, pool)
        result._maybe_init_forward_bufs = lambda _: None
        result.publish_ready = SimpleNamespace(record=lambda: None)
        return result

    def margins(keys, width, epoch):
        # Different signed values per slot, column and time expose accidental broadcasts.
        return torch.tensor([
            [((k + j * 7 + epoch * 11 * (j + 1)) % 31) - 15 for j in range(width)]
            for k in keys
        ], dtype=torch.int32).reshape(len(keys), width)

    for width in [4, 8, 12, 16]:
        for a, b, la, rb in itertools.product([0, 1, 5], [0, 1, 7], [False, True], [False, True]):
            x, y = list(range(1, a + 1)), list(range(20, 20 + b))
            v, w = margins(x, width, 3), margins(y, width, 8)
            left, right = draft(x, v if la else None), draft(y, w if rb else None)
            want = None if not (la or rb) else torch.cat([v if la else torch.zeros_like(v), w if rb else torch.zeros_like(w)])
            left.merge_batch(right)
            picked = rng.sample(range(a + b), rng.randint(0, a + b))
            left.filter_batch(ids(picked), picked)
            if want is None:
                assert left.tree_policy_advantage is None
            else:
                assert torch.equal(left.tree_policy_advantage, want[ids(picked)])
            counts['direct_merge_then_filter'] += 1

        leader, peer = map_(), map_()
        leader.stash(ids([]), src.RelayPayload(bonus_tokens=ids([])))
        assert leader.tree_policy_advantage_buf is None
        slots = torch.zeros((37, width), dtype=torch.int32)
        bonuses = torch.zeros(37, dtype=torch.int64)
        for epoch in range(120):
            write = rng.sample(range(1, 37), rng.randint(1, 30))
            active = epoch % 3 != 0
            keys = [epoch * 100 + k for k in write]
            v = margins(write, width, epoch)
            before = None if leader.tree_policy_advantage_buf is None else leader.tree_policy_advantage_buf.clone()
            leader.publish(ids(write), ids(keys))
            if before is not None:
                assert torch.equal(leader.tree_policy_advantage_buf, before)
            packet = src.RelayPayload.from_draft_input(draft(keys, v if active else None))
            leader.stash(ids(write), packet)
            peer.stash(ids(write), src.RelayPayload.from_draft_input(draft(keys)))
            assert peer.tree_policy_advantage_buf is None
            slots[ids(write)] = v if active else 0
            bonuses[ids(write)] = ids(keys)
            if active:
                packet.tree_policy_advantage.fill_(0x3456)
            handles = []
            reorders = []
            live_slots = (bonuses != 0).nonzero().reshape(-1).tolist()
            chosen = rng.sample(live_slots, rng.randint(0, len(live_slots)))
            pivot = rng.randint(0, len(chosen))
            for keys_ in (chosen[:pivot], chosen[pivot:]):
                handle = draft(keys_, margins(keys_, width, 1811))
                handle.future_indices = ids(keys_)
                reorder = rng.sample(range(len(keys_)), len(keys_))
                handle.filter_batch(ids(reorder), reorder)
                assert handle.tree_policy_advantage is None
                reorders.extend(keys_[i] for i in reorder)
                handles.append(handle)
            handles[0].merge_batch(handles[1])
            leader._resolve_spec_extras(SimpleNamespace(spec_info=handles[0]))
            assert torch.equal(handles[0].bonus_tokens.to(torch.int64), bonuses[ids(reorders)])
            if epoch == 0 or not reorders:
                assert handles[0].tree_policy_advantage is None
            else:
                assert torch.equal(handles[0].tree_policy_advantage, slots[ids(reorders)])
            if epoch:
                assert torch.equal(leader.tree_policy_advantage_buf, slots)
            counts['early_publish_late_stash_resolve'] += 1
            counts['nonowner_neutral'] += 1
    return counts


def check_identification():
    # A token tuple denotes a complete path. The scored selector chooses unique
    # vocabulary IDs at a given position, so sibling IDs are unique on this path.
    rng = random.Random(900325)
    counts = Counter()
    for w in [4, 5, 8, 12, 16]:
        for _ in range(35):
            ranked = [()]
            while len(ranked) < w:
                child = rng.choice(ranked) + (rng.randrange(2),)
                if child not in ranked:
                    ranked.append(child)
            native = tuple(rng.randrange(2) for _ in range(w - 1))
            policies = []
            for s in range(w):
                forced = {native[:i] for i in range(s + 1)}
                extra = [path for path in ranked if path not in forced][: w - s - 1]
                shapes = forced | set(extra)
                assert len(shapes) == w
                assert all(not path or path[:-1] in shapes for path in shapes)
                if policies:
                    old = policies[-1]
                    assert len(shapes - old) == len(old - shapes) <= 1
                    for leaf in shapes ^ old:
                        assert not any(len(p) > len(leaf) and p[:len(leaf)] == leaf for p in shapes | old)
                policies.append(shapes)
                counts['closed_topologies'] += 1
            streams = itertools.product(range(2), repeat=w - 1) if w <= 8 else (
                tuple(rng.randrange(2) for _ in range(w-1)) for _ in range(48)
            )
            for stream in streams:
                gold = [1 + max(d for d in range(w) if stream[:d] in tree) for tree in policies]
                for observed in set(gold):
                    lower = [min(reward, observed + 1) for reward in gold]
                    for s in range(1, w):
                        diff = gold[s] - gold[s - 1]
                        assert abs(diff) <= 1
                        # Actual updater excludes identical incumbent topologies from censoring.
                        censored = lower[s] == lower[s-1] == observed + 1 < w and policies[s] != policies[s-1]
                        if not censored:
                            assert lower[s] - lower[s-1] == diff
                            counts['identified_adjacent_rewards'] += 1
                        else:
                            counts['censored_comparisons'] += 1
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sglang', type=Path)
    args = parser.parse_args()
    files = [
        'python/sglang/srt/speculative/draft_worker_common.py',
        'python/sglang/srt/managers/overlap_utils.py',
        'python/sglang/srt/speculative/dflash_info_v2.py',
    ]
    print(json.dumps({
        'torch': torch.__version__, 'device': 'cpu', 'widths_lifecycle': [4, 8, 12, 16],
        'lifecycle': check_relay(args.sglang), 'independent_theorem': check_identification(),
        'source_sha256': {file: hashlib.sha256((args.sglang/file).read_bytes()).hexdigest() for file in files},
        'scope': 'Production DFlash state and FutureMap CPU methods; independent mathematical oracle for semantic path families and partial rewards. No GPU or throughput estimate.'
    }, indent=2))
