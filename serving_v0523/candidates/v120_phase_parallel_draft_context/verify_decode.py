"""Compare the production draft-KV append boundary on a real CPU/Gloo group."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(rank, world, rendezvous, source):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=world, timeout=timedelta(minutes=2))
    torch.set_num_threads(1)
    # Reuse the parent fixture's extraction of exact pinned SGLang methods.
    parent = Path(__file__).resolve().parents[1] / "v108_replicated_sp_draft_context"
    sys.argv = [sys.argv[0], str(source)]
    fixture = load("prefill_fixture", parent / "cpu_check.py")
    candidate = fixture.load_projection()
    get_method = fixture.method

    class Group(fixture.Group):
        rank_in_group = rank

        def all_gather(self, value, dim):
            assert value.ndim == 2 and dim == -1
            rows, channels = value.shape
            collected = value.new_empty((world * rows, channels))
            self.all_gather_into_tensor(collected, value)
            return collected.view(world, rows, channels).movedim(0, 1).reshape(rows, -1)

    group = Group(world)

    class Draft(torch.nn.Module):
        quantizable = False

    Draft.project_target_hidden = get_method(
        "models/dflash.py", "DFlashDraftModel", "project_target_hidden", {"torch": torch}
    )
    dflash = ModuleType("sglang.srt.models.dflash")
    dflash.DFlashDraftModel = Draft
    sys.modules[dflash.__name__] = dflash
    append = get_method(
        "speculative/dflash_worker_v2.py", "DFlashWorkerV2", "_append_target_hidden_to_draft_kv_by_loc",
        {"torch": torch, "logger": logging.getLogger("decode"), "draft_tp_context": lambda _: nullcontext(),
         "project_dflash_verify": candidate.project_dflash_verify},
    )
    draft = Draft()
    seen = []
    def capture(**payload):
        seen.append({name: value.clone() for name, value in payload.items() if value is not None})

    worker = SimpleNamespace(
        draft_parallel_layout=None, draft_model=draft, model_runner=SimpleNamespace(device=torch.device("cpu")),
        lilicorr=None, _use_fused_kv_materialize=True, _fused_kv_helper=object(), _append_target_hidden_fused=capture,
    )
    with torch.no_grad():
        for features, outputs, shapes in [
            (160, 32, [(1, 1), (1, 8), (3, 7), (4, 8), (16, 8), (32, 8), (7, 12)]),
            (40960, 24, [(3, 8)]),
        ]:
            generator = torch.Generator().manual_seed(6120)
            draft.fc = torch.nn.Linear(features, outputs, bias=False).bfloat16()
            draft.fc.weight.copy_(torch.randint(-4, 5, (outputs, features), generator=generator).bfloat16() / 8)
            draft.hidden_norm = torch.nn.RMSNorm(outputs, eps=1e-5).bfloat16()
            with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)), patch.object(
                torch.cuda, "get_device_capability", return_value=(9, 0)
            ):
                handle = candidate.prepare_dflash_verify_projection(draft, group)
            assert handle is not None
            begin = rank * (outputs // world)
            assert handle.first_channel == begin and handle.width == outputs // world
            for batch_size, width in shapes:
                rows = batch_size * width
                positions = torch.randperm(rows, generator=generator, dtype=torch.int64) + 17
                slots = torch.randperm(rows, generator=generator, dtype=torch.int64) + 200
                lengths = torch.arange(batch_size, dtype=torch.int32) % (width + 1)
                lengths[-1] = width
                inputs = torch.randint(-8, 9, (rows + 3, 2 * features), generator=generator,
                                       dtype=torch.int8)[:, ::2].bfloat16() / 8
                # A view exercises the native worker's cap on padded target outputs.
                padded = torch.stack((inputs, -inputs), dim=-1)[..., 0]
                assert not padded.is_contiguous()
                for context in (None, handle):
                    worker._verify_projection = context
                    group.gathers.clear()
                    append(worker, target_hidden=padded, positions=positions, cache_loc=slots,
                           cache_loc_2d=slots.view(batch_size, width), commit_lens=lengths)
                    assert group.gathers == ([] if context is None else [(rows * world, outputs // world)])
                assert seen[-1].keys() == seen[-2].keys()
                for key in seen[-1]:
                    assert torch.equal(seen[-1][key], seen[-2][key]), (world, rank, rows, key)
                reference = draft.hidden_norm(torch.nn.functional.linear(padded[:rows], draft.fc.weight))
                assert torch.equal(seen[-1]["ctx_hidden"], reference)
                assert torch.equal(seen[-1]["ctx_positions"], positions)
                assert torch.equal(seen[-1]["ctx_cache_loc_2d"], slots.view(batch_size, width))
                assert torch.equal(seen[-1]["commit_lens"], lengths)
                for already_projected in (False, True):
                    group.gathers.clear()
                    hidden = torch.nn.functional.linear(padded[:rows], draft.fc.weight) if already_projected else padded[:rows]
                    append(worker, target_hidden=hidden, positions=positions, cache_loc=slots,
                           extend_lens=lengths, prefill_fc_done=already_projected)
                    assert not group.gathers
                    assert torch.equal(seen[-1]["ctx_hidden"], reference)
            group.gathers.clear()
            before = len(seen)
            append(worker, target_hidden=torch.empty((0, features), dtype=torch.bfloat16),
                   cache_loc=slots[:0], positions=positions[:0], commit_lens=lengths[:0],
                   cache_loc_2d=slots[:0].view(0, width))
            assert not group.gathers and len(seen) == before
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with tempfile.TemporaryDirectory() as temporary:
        for world in (2, 4, 8):
            mp.spawn(run, args=(world, str(Path(temporary) / str(world)), args.source), nprocs=world)
    print(json.dumps({
        "torch": torch.__version__, "collectives": "real Gloo all_gather_into_tensor and startup MIN",
        "tp": [2, 4, 8], "decode_cases_per_tp": 8, "ordinary_and_preprojected_prefill_cases_per_tp": 16,
        "empty_worker_calls_per_tp": 2, "full_K": 40960,
        "maximum_verify_rows": 256, "compare": "bitwise BF16 normalized draft input, lengths, all physical slots and positions",
        "cuda_h200": False,
    }, indent=2))
