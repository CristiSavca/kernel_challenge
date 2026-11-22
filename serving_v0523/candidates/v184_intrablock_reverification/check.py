"""Exercise bounded Jacobi re-verification using pinned DFLASH acceptance and a fresh causal oracle."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from typing import Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
V68 = ROOT / "candidates/v68_grouped_decode_verify/patch.diff"
W = 8
torch.set_num_threads(1)


def pinned(name):
    path = "python/sglang/" + name
    assert "diff --git a/" + path + " " not in V68.read_text()
    return subprocess.check_output(["git", "-C", ROOT / "sglang", "show", PIN + ":" + path], text=True)


def native(name, code):
    node, = [x for x in ast.parse(code).body if getattr(x, "name", "") == name]
    scope = {"torch": torch, "Tuple": Tuple}
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), scope)
    return scope[name]


worker = pinned("srt/speculative/dflash_worker_v2.py")
accept = native("compute_dflash_correct_drafts_and_bonus", pinned("kernels/ops/speculative/dflash.py"))
commit = native("_commit_accept", worker)


def accepted(inputs, predictions):
    lengths, bonus = accept(candidates=inputs, target_predict=predictions)
    tokens, count = commit(inputs, lengths, bonus)
    return tokens, count.long()


def replay(inputs, model):
    """At committed count C, repair the remaining W−C physical rows on their true prefix."""
    pred1, taps1 = model(inputs)
    prefix_out, lengths = accepted(inputs, pred1)
    # Return counts only determine what is published. Old proposed rows still occupy the suffix.
    assembled, total, cache_input, cache_taps = prefix_out.clone(), lengths.clone(), inputs.clone(), taps1.clone()
    for c in range(1, W):
        subset = torch.nonzero(lengths == c).flatten()
        if subset.numel() == 0:
            continue
        # Commit's final bonus is output c−1. It becomes the INPUT at c on the next target pass.
        # The last speculative output has no physical input slot at W in this reservation.
        repair = pred1[subset, c - 1: W - 1]
        repaired_input = torch.cat((inputs[subset, :c], repair), dim=1)
        preds, taps = model(repaired_input)
        extra, count = accepted(repair, preds[:, c:])
        assembled[subset, c:] = extra
        total[subset] += count
        cache_input[subset, c:] = repair
        cache_taps[subset, c:] = taps[:, c:]
    # Independent autoregressive oracle, including hidden features on each materialized input.
    greedy = inputs[:, :1]
    expected = []
    expected_taps = []
    for col in range(W):
        pred, feature = model(greedy)
        out = pred[:, -1:]
        expected.append(out)
        expected_taps.append(feature[:, -1:])
        greedy = torch.cat((greedy, out), dim=1)
    mask = torch.arange(W)[None] < total[:, None]
    assert torch.equal(assembled[mask], torch.cat(expected, dim=1)[mask])
    assert torch.equal(cache_input[mask], greedy[:, :W][mask])
    assert torch.equal(cache_taps[mask], torch.cat(expected_taps, dim=1)[mask])
    return lengths, total


def bit_target(mode):
    # Position-dependent transitions include identity, complement and both constant outputs.
    def model(x):
        rows = torch.arange(x.size(1))[None]
        key = (x * torch.where(rows % 2 == 0, 1, 3)).cumsum(dim=1) % 13
        pred = (mode[rows, key] >> x) & 1
        # Exact inclusive prefix fingerprint through width eight. It is also the synthetic KV payload.
        tap = (x << rows).cumsum(dim=1)
        return pred, tap
    return model


def sensitivity(mean, bs):
    def expectation(p, width):
        return sum(p ** n for n in range(width))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        if expectation(p, W) < mean:
            lo = p
        else:
            hi = p
    p = (lo + hi) / 2
    probability = [(1 - p) * p ** (c - 1) for c in range(1, W)] + [p ** (W - 1)]
    repeat_batch = 1 - probability[-1] ** bs
    rows = []
    for q in (0, 0.25, 0.5, 0.75, 0.9, 1):
        gain = sum(mass * expectation(q, W - c) for c, mass in enumerate(probability, 1))
        rows.append(dict(conditional_repair_agreement=q, extra_commits_per_request=gain,
                         max_extra_target_fraction_of_native_round_at_break_even=gain / mean / repeat_batch,
                         speed_if_extra_target_costs_75pct=(1 + gain / mean) / (1 + repeat_batch * 0.75)))
    distribution_free = {}
    for cost in (0,0.5,0.75,1):
        # G <= W−E[C] and G <= (W−1)*Pr(at least one repair).
        # Ratio increases with G; the favorable extremum uses only C=1 and C=W.
        gain = W - mean
        distribution_free[str(cost)] = (1 + gain / mean) / (1 + cost * gain / (W - 1))
    return dict(mean_commits=mean, iid_native_match=p, probability_full_native=probability[-1],
                batch_needs_replay_independent_requests=repeat_batch, sensitivity=rows,
                upper_bound_with_constant_cost_per_nonempty_repair=distribution_free)


def main():
    torch.manual_seed(184)
    inputs = torch.tensor(list(itertools.product(range(2), repeat=W)), dtype=torch.int64)
    lengths_seen = set()
    checked = 0
    for _ in range(256):
        lens, total = replay(inputs, bit_target(torch.randint(4, (W, 13))))
        lengths_seen.update(zip(lens.tolist(), total.tolist()))
        checked += len(lens)
    examples = {}
    for m in range(1, W):
        a = torch.tensor([[1] * m + [0] * (W - m)])
        best, worst = [], []
        for mode, store in ((3, best), (2, worst)):
            lens, full = replay(a, bit_target(torch.full((W, 13), mode)))
            assert lens.item() == m
            store.append(full.item())
        assert best == [W] and worst == [m + 1]
        examples[m] = dict(same_input=a.tolist()[0], same_true_output=[1] * W,
                           full_if_position_only=best[0], full_if_input_sensitive=worst[0])
    assert lengths_seen == {(c, t) for c in range(1, W) for t in range(c + 1, W + 1)} | {(W, W)}
    results = {}
    for label, job in (("V64","abd217baf0ed49fd"),("V68","04b01f958aa5487c")):
        report = json.loads((ROOT / "runs" / (job + ".json")).read_text())
        results[label] = {str(r["concurrency"]): sensitivity(r["acceptance"], r["concurrency"])
                          for r in report["result"]["rungs"]}
    # The graph supplies only one shared aux storage and the existing worker consumes it before reuse.
    assert "every decode graph size shares one aux output" in worker
    body = worker[worker.index("        # --- 2)"):] if "        # --- 2)" in worker else worker
    assert body.rfind("on_publish(new_seq_lens)") < body.rfind("self._append_target_hidden_to_draft_kv_by_loc(")
    verify = pinned("srt/speculative/dflash_info.py")
    assert "(bs + 1) * self.draft_token_num" in verify and "kv_indices_extra = self.draft_token_num * bs" in verify
    return dict(base=PIN,v68_sha256=hashlib.sha256(V68.read_bytes()).hexdigest(),
                native_accept_commit_cases=checked, observed_length_pairs=len(lengths_seen),
                synthetic_same_prefix_counterexamples=examples, measured_acceptance_and_modeled_cost=results,
                synthetic=True, physical_rerun_timing=None)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
