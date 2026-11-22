"""Check native greedy prefix acceptance and quantify the associated work model on CPU."""

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
WIDTH = 8
WIDTHS = range(1, WIDTH + 1)
PATCH = ROOT / "candidates/v64_humming_grouped_sequence_parallel/patch.diff"


def pinned(path):
    assert f"diff --git a/python/sglang/{path} " not in PATCH.read_text()
    return subprocess.check_output(
        ["git", "-C", ROOT / "sglang", "show", f"{PIN}:python/sglang/{path}"], text=True
    )


def load_function(path, name):
    functions = ast.parse(pinned(path)).body
    selected, = [node for node in functions if isinstance(node, ast.FunctionDef) and node.name == name]
    scope = {"torch": torch, "Tuple": Tuple}
    exec(compile(ast.Module([selected], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def check_native_prefix():
    accept = load_function("kernels/ops/speculative/dflash.py", "compute_dflash_correct_drafts_and_bonus")
    commit = load_function("srt/speculative/dflash_worker_v2.py", "_commit_accept")
    patterns = list(itertools.product((0, 1), repeat=WIDTH - 1))
    target = torch.arange(len(patterns) * WIDTH).reshape(-1, WIDTH) * 3 + 5
    current = torch.full((len(patterns), 1), -1)
    suffix = torch.where(torch.tensor(patterns).bool(), target[:, :-1], target[:, :-1] + 1)
    draft = torch.cat((current, suffix), dim=1)
    full_accept, full_bonus = accept(candidates=draft, target_predict=target)
    full_out, full_count = commit(draft, full_accept, full_bonus)
    cases = 0
    for w in WIDTHS:
        short_draft = draft[:, :w].contiguous()
        short_target = target[:, :w].contiguous()
        matches, bonus = accept(candidates=short_draft, target_predict=short_target)
        out, count = commit(short_draft, matches, bonus)
        assert torch.equal(count, full_count.clamp(max=w))
        for row, n in enumerate(count.tolist()):
            assert torch.equal(out[row, :n], full_out[row, :n])
            assert torch.equal(out[row, :n], target[row, :n])
            cases += 1
    return cases


def commits(p, width):
    return sum(p**depth for depth in range(width))


def expected_bank(batch, width):
    return 32 * (1 - (1 - 8 / 256) ** (batch * width))


def work_model(mean):
    left, right = 0.0, 1.0
    for _ in range(64):
        midpoint = (left + right) / 2
        if commits(midpoint, WIDTH) < mean:
            left = midpoint
        else:
            right = midpoint
    p = (left + right) / 2
    fixed = []
    for w in WIDTHS:
        kept = commits(p, w) / mean
        expert_saving = 1 - expected_bank(1, w) / expected_bank(1, WIDTH)
        fixed.append({
            "width": w,
            "same_anchor_commits_bounds": [1 + (mean - 1) * (w - 1) / (WIDTH - 1), min(mean, w)],
            "expected_commits": commits(p, w),
            "expert_time_share_to_break_even": (1 - kept) / expert_saving if w < WIDTH else None,
            "speedup_by_expert_share": {
                f: kept / (1 - f * expert_saving) for f in (0.25, 0.5, 0.75, 1.0)
            },
        })

    # Both regimes independently generate native-prefix attempts. Their mix has
    # the same mean as the homogeneous model. The policy observes regime only.
    low, high = 0.25, 0.96
    high_probability = (mean - commits(low, WIDTH)) / (commits(high, WIDTH) - commits(low, WIDTH))
    adaptive = []
    for share in (0.25, 0.5, 0.75):
        def time(w):
            return 1 - share + share * expected_bank(1, w) / expected_bank(1, WIDTH)

        for classifier in ("perfect", "no_information"):
            pairs = itertools.product(WIDTHS, repeat=2) if classifier == "perfect" else zip(WIDTHS, WIDTHS)
            speeds = []
            for a, b in pairs:
                expected_tokens = (1 - high_probability) * commits(low, a) + high_probability * commits(high, b)
                expected_time = (1 - high_probability) * time(a) + high_probability * time(b)
                speeds.append((expected_tokens / expected_time / mean, a, b))
            ratio, low_width, high_width = max(speeds)
            adaptive.append(dict(expert_time_share=share, classifier=classifier, speedup=ratio,
                                 low_width=low_width, high_width=high_width))
    return dict(homogeneous_match_probability=p, fixed_widths=fixed, high_regime_probability=high_probability,
                low_match_probability=low, high_match_probability=high, two_regime_widths=adaptive)


if __name__ == "__main__":
    receipt = json.loads((ROOT / "runs/abd217baf0ed49fd.json").read_text())
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == receipt["result"]["patch"]["sha256"]
    one_user = next(rung for rung in receipt["result"]["rungs"] if rung["concurrency"] == 1)
    result = dict(assumption="Model the rounded published acceptance as a same-anchor mean per call.",
                  scored_receipt=receipt["id"], v64_sha=receipt["result"]["patch"]["sha256"], torch=str(torch.__version__),
                  native_prefix_cases=check_native_prefix(), one_user=one_user,
                  model=work_model(one_user["acceptance"]))
    print(json.dumps(result, indent=2))
