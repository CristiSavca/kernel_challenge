"""Native verification oracle, conditional cold-weight model and adversarial head certificate."""
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SERVING = HERE.parents[1]
WIDTH = 8
DIM = 8192
VOCAB = 202048
HEAD_MIB = (VOCAB // 8) * DIM * 2 / 2**20


def native_oracle():
    source = HERE.parent / "v127_native_prefix_verify/prefix_study.py"
    spec = importlib.util.spec_from_file_location("prefix_study", source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    accept = previous.load_function("kernels/ops/speculative/dflash.py", "compute_dflash_correct_drafts_and_bonus")
    commit = previous.load_function("srt/speculative/dflash_worker_v2.py", "_commit_accept")
    patterns = torch.tensor(list(itertools.product((False, True), repeat=WIDTH - 1)))
    target = 3 * torch.arange(len(patterns) * WIDTH).reshape(-1, WIDTH) + 7
    draft = torch.cat((target[:, :1] - 2, torch.where(patterns, target[:, :-1], target[:, :-1] + 1)), dim=1)
    matched, bonus = accept(candidates=draft, target_predict=target)
    native, counts = commit(draft, matched, bonus)
    cases = 0
    for split in range(1, WIDTH):
        for row in range(len(target)):
            seen = []
            wants_more = True
            for begin, end in ((0, split), (split, WIDTH)):
                if wants_more:
                    for j in range(begin, end):
                        seen.append(int(target[row, j]))
                        wants_more = j + 1 < WIDTH and int(draft[row, j + 1]) == seen[-1]
                        if not wants_more:
                            break
            assert seen == native[row, :counts[row]].tolist()
            cases += 1
    return dict(staged_splits_and_native_patterns=cases, width=WIDTH, torch=str(torch.__version__))


def witness_oracle():
    # Give either an exact rejection witness, or no witness, independently for every mismatch.
    cases = 0
    for state in itertools.product(range(3), repeat=WIDTH - 1):
        mismatch = [i for i, x in enumerate(state) if x]
        witnesses = [i for i, x in enumerate(state) if x == 2]
        native_count = min(mismatch, default=WIDTH-1) + 1
        head_limit = min(witnesses, default=WIDTH-1) + 1
        assert head_limit >= native_count
        assert min((i for i in mismatch if i < head_limit), default=head_limit - 1) + 1 == native_count
        cases += 1
    return cases


def bank(m):
    return 32 * (1 - (31 / 32) ** m)


def cold_sparse_layer(m):
    if not m:
        return 0
    return (132 if m <= 8 else 188) + 27 * bank(m)


def forecasts(b, mean):
    lo, hi = 0., 1.
    for _ in range(60):
        q = (lo + hi) / 2
        if sum(q**j for j in range(WIDTH)) < mean:
            lo = q
        else:
            hi = q
    q = (lo + hi) / 2
    rows = []
    for r in range(1, WIDTH):
        p = q ** r
        probs = [math.comb(b, n) * p**n * (1-p)**(b-n) for n in range(b+1)]
        dense_second = 1 - probs[0]
        base = cold_sparse_layer(WIDTH*b)
        two = cold_sparse_layer(r*b) + sum(w*cold_sparse_layer((WIDTH-r)*n) for n,w in enumerate(probs))
        active = bank(r*b) + sum(w*bank((WIDTH-r)*n) for n,w in enumerate(probs))
        rows.append(dict(first_rows=r, continuation_per_request=p,
                         continuation_bounds_from_mean=[max(0,(mean-r)/(WIDTH-r)), min(1,(mean-1)/r)],
                         any_batch_continuation=dense_second, expected_target_rows=b*(r+(WIDTH-r)*p),
                         expert_banks_saved_per_sparse_layer=bank(WIDTH*b)-active,
                         sparse_layer_mib_native=base, sparse_layer_mib_staged=two,
                         cold_sparse_layer_byte_ratio_native_over_staged=base/two,
                         extra_head_sweep_mib_per_rank=HEAD_MIB*dense_second))
    return dict(homogeneous_match_probability=q, split_cases=rows)


def certificate_sensitivity(trials=32):
    rng = np.random.default_rng(131)
    rhos = (.95, .99, .999, .9999, .99999, .999999)
    counts = {r: dict(in_draft_top16=0, certified=0) for r in rhos}
    for _ in range(trials):
        # Exact projected distribution of isotropic Gaussian weight rows, w ~ N(0,I/H).
        u, v = rng.standard_normal((2, VOCAB))
        residual = rng.chisquare(DIM-2, size=VOCAB)
        max_norm = math.sqrt(float(np.max(u*u + v*v + residual)) / DIM)
        kept = np.argpartition(u, -16)[-16:]
        cutoff = np.min(u[kept])
        for rho in rhos:
            scores = rho * u + math.sqrt(1-rho**2) * v
            winner = np.argmax(scores)
            in_set = bool(np.any(kept == winner))
            certified = float(np.max(scores[kept])) > cutoff + max_norm * math.sqrt(2*DIM*(1-rho))
            assert not certified or in_set
            counts[rho]["in_draft_top16"] += int(in_set)
            counts[rho]["certified"] += int(certified)
    return dict(synthetic_trials=trials, dim=DIM, vocabulary=VOCAB, thresholds=counts,
                excludes_numeric_error=True, seed=131)


def main():
    measured = json.loads((SERVING/"runs/abd217baf0ed49fd.json").read_text())
    patch = HERE.parent/"v64_humming_grouped_sequence_parallel/patch.diff"
    sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert sha == measured["result"]["patch"]["sha256"]
    return dict(scored_v64=measured["id"], sha256=sha, native=native_oracle(),
                witness_finite_cases=witness_oracle(), head_mib=HEAD_MIB,
                cold_footprint_forecasts={str(r["concurrency"]): forecasts(r["concurrency"],r["acceptance"])
                                          for r in measured["result"]["rungs"]},
                shared_head_certificate=certificate_sensitivity())


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
