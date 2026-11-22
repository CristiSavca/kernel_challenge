"""Feed real grouped-callback output into the target used by the mixed scheduler replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType
from unittest.mock import patch

import torch
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()


def load(name, pattern):
    path, = HERE.parent.glob(pattern)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def experts():
    cb = load("v211_experts", "v207_*/check_callback.py")
    tree = subprocess.check_output(["git", "-C", STAGE, "write-tree"]).decode().strip()
    base = cb.MANIFEST["parent_tree"]
    rows = 8192
    x = cb.host["rand"](rows, 128)
    scores = cb.host["rand"](rows, 8).float()
    ids = torch.full((rows, 8), -1, dtype=torch.int32)
    for slot in range(3):
        ids[:80, slot] = (torch.arange(80) + slot) % 8
    ids[80::73, 0] = 3
    cb.layer.humming_prompt_gate_up = cb.host["pstate"]("paired")
    cb.layer.humming_prompt_down = cb.fixture["bind"]()
    cb.fixture["scope"]["_post_expert_norm_weighted_sum_kernel"] = cb.Finalizer()
    outputs = []
    for revision in (base, tree):
        group = cb.definition(revision, cb.hum, "humming_grouped_expert_outputs",
                              dict(cb.host["env"], humming_expert_outputs=cb.indexed))
        scope = dict(cb.fixture["scope"])
        combine = cb.definition(revision, cb.moe, "_grouped_humming_and_combine_impl",
                                scope, "MuseSpark1xFp4Experts")
        cb.fixture["model_hum"].humming_grouped_expert_outputs = group
        cb.layer._grouped_humming_and_combine = MethodType(combine, cb.layer)
        cb.fixture["events"].clear()
        with torch.inference_mode(), patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            outputs.append(cb.fixture["forward"](x, scores, ids))
        assert [event[0] for event in cb.fixture["events"]] == ["paired", "W2 custom"]
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    assert cb.launches == {"ordinary compact": 1, "scaled compact": 1}
    bank = outputs[1][:80, :8].contiguous().clone()
    assert torch.isfinite(bank).all() and (bank.abs().sum(1) > 0).all()
    return bank


def rank_worker(rank, size, folder):
    entry = load("v211_scheduler", "v209_*/entry_scheduler.py")
    bank = torch.load(Path(folder) / "rows.pt", weights_only=True)
    original_load = entry.load
    def with_experts(name, path):
        fixture = original_load(name, path)
        if name == "scheduler":
            def body(ids):
                hidden = bank.index_select(0, ids.long().remainder(len(bank)))
                return hidden, [hidden * scale for scale in (1, 2, 3, 4, 5)]
            assert fixture.ref is fixture.settle.ref
            fixture.ref.body_tokens = body
        return fixture
    entry.load = with_experts
    entry.check_rank(rank, size, folder)


if __name__ == "__main__":
    bank = experts()
    records = {}
    for size in (4, 8):
        with tempfile.TemporaryDirectory() as folder:
            torch.save(bank, Path(folder) / "rows.pt")
            mp.spawn(rank_worker, args=(size, folder), nprocs=size, join=True)
            report = json.loads((Path(folder) / "out.json").read_bytes())
            rounds = [r for r in report if "case" in r]
            records[size] = dict(
                complete_rounds=len(rounds), compared_pairs=len(rounds)//2,
                declined=sum("decline" in r for r in report),
                cache_words_per_rank=sum(r["cache_words"] for r in rounds),
                scenarios=sorted({r["case"] for r in rounds}),
                full_graph_examples=[r for r in rounds if r["backend"] == "FULL" and r["case"] == "long"],
            )
    print(json.dumps(dict(candidate_sha256=hashlib.sha256((HERE / "patch.diff").read_bytes()).hexdigest(),
        expert_output_words_compared=8192*128, connected_bf16_words=bank.numel(),
        connected_sha256=hashlib.sha256(bank.view(torch.int16).numpy().tobytes()).hexdigest(),
        results=records, physical_H200=False,
        scope="Real host expert and scheduler source, shared synthetic expert results drive all target hidden states, "
              "CPU-modeled expert GEMMs and graph body, native host verification, physical slot checks, real Gloo."), indent=2))
