"""Actual PyTorch/Gloo comparison of compact DFLASH noise embeddings under TP8."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[2]
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def source(path):
    return subprocess.check_output(["git", "-C", str(ROOT/"sglang"), "show",
                                    f"{BASE}:python/sglang/{path}"]).decode()


class Embedding(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.weight = nn.Parameter(torch.randn(64, 256).bfloat16(), requires_grad=False)
        self.calls = []

    def forward(self, tokens):
        part = self.local(tokens)
        self.calls.append(tuple(tokens.shape))
        dist.all_reduce(part)
        return part

    def local(self, tokens):
        local = tokens - self.rank * 64
        return self.weight[local.clamp(0, 63)].masked_fill(((local < 0) | (local >= 64))[..., None], 0)


def compact(model, block_ids):
    # The model's public hooks can mutate repeated IDs or derive outputs from
    # individual positions. All ranks must pick the same collective shape.
    compatible = not (model._forward_hooks or model._forward_pre_hooks
                      or nn.modules.module._global_forward_hooks
                      or nn.modules.module._global_forward_pre_hooks)
    ready = torch.tensor(int(compatible), dtype=torch.int32)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        return model(block_ids)
    anchors = block_ids[:, 0]
    packed = torch.cat((anchors, block_ids[0, 1:2]))
    features = model(packed)
    mask = features[-1].view(1, 1, -1).expand(len(anchors), block_ids.size(1)-1, -1)
    return torch.cat((features[:-1, None, :], mask), dim=1)


def mixed_dense(model, block_ids):
    if (model._forward_hooks or model._forward_pre_hooks
            or nn.modules.module._global_forward_hooks or nn.modules.module._global_forward_pre_hooks):
        return model(block_ids), False
    b,w = block_ids.shape
    local = model.local(torch.cat((block_ids[:,0], block_ids[0,1:2])))
    part = torch.cat((local[:-1,None,:], local[-1].view(1,1,-1).expand(b,w-1,-1)),dim=1)
    model.calls.append(tuple(part.shape[:2]))
    dist.all_reduce(part)
    return part, True


def run(rank, init, folder, dense_only):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+init,rank=rank,world_size=8)
    torch.manual_seed(1230+rank)
    model = Embedding(rank)
    cases = clean = values = 0
    for bs in [1, 4, 16, 32]:
        for width in [4, 8, 12]:
            for mode in ["ordinary","input","output","global","updated_mask"]:
                mask = 497
                tok = torch.full((bs,width), mask, dtype=torch.long)
                tok[:,0] = (torch.arange(bs)*89+7)%512
                handles = []
                if mode == "input" and rank==1:
                    # Only one rank mutates mask positions before embedding.
                    def hook(mod, args):
                        ids = args[0].clone()
                        ids[...,2::2] = 64+rank
                        return (ids,)
                    handles.append(model.register_forward_pre_hook(hook))
                if mode == "output" and rank==3:
                    handles.append(model.register_forward_hook(lambda m,a,y:y+.125))
                if mode == "global" and rank==0:
                    handles.append(nn.modules.module.register_module_forward_pre_hook(lambda *a:None))
                if mode == "updated_mask" and mask//64==rank:
                    model.weight[mask%64].add_(.25)
                with torch.no_grad():
                    expected = model(tok)
                    if dense_only:
                        actual, looked_up_compactly = mixed_dense(model,tok)
                    else:
                        actual = compact(model,tok)
                assert torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                fast = mode in ("ordinary","updated_mask")
                if dense_only:
                    assert model.calls[-1] == (bs,width)
                    fast = looked_up_compactly
                else:
                    assert model.calls[-1] == ((bs+1,) if fast else (bs,width))
                for handle in handles: handle.remove()
                cases += 1; clean += int(fast); values += expected.numel()
    path=Path(folder)/f"{rank}.json"
    if dense_only:
        path.write_text(json.dumps({"rank":rank,"cases":cases,"compact_local_lookups":clean,
                                    "raw_bf16_values":values,"all_full_size_collectives":True}))
        dist.destroy_process_group()
        return
    flag=torch.tensor(1,dtype=torch.int32)
    durations=[]
    for i in range(130):
        start=time.perf_counter_ns()
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        elapsed=(time.perf_counter_ns()-start)/1000
        if i>=30: durations.append(elapsed)
    path.write_text(json.dumps({
        "rank":rank,"cases":cases,"compact":clean,"raw_bf16_values":values,
        "host_vote_median_us":statistics.median(durations),
        "host_vote_p90_us":sorted(durations)[89]}))
    dist.destroy_process_group()


def report():
    worker=source("srt/speculative/dflash_worker_v2.py")
    assert "noise_embedding = embed_module(block_ids)" in worker
    assert "_resolve_dflash_embedding_module(self.draft_model, target_model)" in worker
    embed=source("srt/layers/vocab_parallel_embedding.py")
    assert "output_parallel = tensor_model_parallel_all_reduce(output_parallel)" in embed
    fa=source("srt/layers/attention/flashattention_backend.py")
    assert "normal_decode_set_metadata(" in fa and "skip_page_table=translated" in fa
    v64=json.loads((ROOT/"runs/abd217baf0ed49fd.json").read_text())["result"]
    geometry=[]
    for rung in v64["rungs"]:
        b=rung["concurrency"]; width=8; hidden=8192
        baseline=b*width*hidden*2
        packed=(b+1)*hidden*2
        geometry.append({"users":b,"original_kib":baseline/1024,"compact_kib":packed/1024,
                         "reduction_payload_fraction_removed":1-packed/baseline,
                         "ideal_ring_send_saved_kib":2*7/8*(baseline-packed)/1024,
                         "illustrative_verify_step_ms":1000*rung["acceptance"]/rung["tok_s_per_user_p50"]})
    return geometry


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--dense-only",action="store_true")
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="v128-gloo-") as d:
        mp.start_processes(run,(str(Path(d)/"init"),d,args.dense_only),8,start_method="spawn")
        ranks=[json.loads((Path(d)/f"{i}.json").read_text()) for i in range(8)]
    print(json.dumps({"pin":BASE,"mode":("mixed local lookup, uniform dense Gloo TP8" if args.dense_only
                                        else "CPU BF16, Gloo TP8"),"ranks":ranks,
                      "geometry":report(),"h200_timing":None},indent=2))
