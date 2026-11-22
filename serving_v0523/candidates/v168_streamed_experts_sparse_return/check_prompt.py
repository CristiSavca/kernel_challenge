"""Check V168's model/BCG/Gloo replay and in-place compact expert windows on CPU."""
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "v167_sparse_expert_return/integration.py"
spec = importlib.util.spec_from_file_location("v167_fixture", SOURCE)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


class ScaledSilu:
    def run(self, gate, out, ids, scales):
        gate, up = (gate * scales[ids.long(), None]).bfloat16().float().chunk(2, dim=1)
        out.copy_(torch.nn.functional.silu(gate) * up)


def fixture(counts):
    f = prior.fixture()
    f.install(f.source(f.HUM), ["_compact_prefill_windows", "compact_prefill_and_combine"], f.env)
    window = f.env["_compact_prefill_windows"]

    def windows(ids, experts, global_experts, known=None, budget=None):
        planned = list(window(ids, experts, global_experts, known, budget))
        if budget is not None:
            assert budget >= len(ids)
            counts["sparse_compact_calls"] += 1
        if len(planned) > 1:
            counts["multiwindow_calls"] += 1
        assert sum(w[2] for w in planned) == int((ids >= 0).sum())
        yield from planned

    class Finalizer:
        def __getitem__(self, grid):
            def launch(values, weight, scores, out, slots, ids, scales, **kw):
                assert kw["HAS_ROW_MAP"] and kw["HAS_EXPERT_SCALE"] == kw["HAS_IDS"]
                if kw["HAS_EXPERT_SCALE"]:
                    active = slots >= 0
                    assert torch.equal(ids >= 0, active)
                    raw = values
                    by_row = values.new_empty((len(values),1))
                    by_row[slots[active].long(),0] = scales[ids[active].long()]
                    values = (raw * by_row).bfloat16()
                    counts["finalizer_scaled"] += 1
                else:
                    counts["finalizer_unscaled"] += 1
                f.finalize(values, scores, norm_weight=weight, eps=kw["eps"], weight_first=kw["WEIGHT_FIRST"],
                           slot_expert_ids=None,slot_row_indices=slots,out=out)
                f.phase.records.append(("finalize", values.shape, slots, out.data_ptr()))
            return launch
    f.env["_post_expert_norm_weighted_sum_kernel"] = Finalizer()
    f.env["_compact_prefill_windows"] = windows
    return f


def worker(rank,size,uri,store):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=size, timeout=timedelta(seconds=90))
    counts = Counter()
    f = fixture(counts)
    with (torch.no_grad(), patch.object(torch.Tensor,"is_cuda",property(lambda _self:True)),
          patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
          patch.object(torch.cuda,"is_current_stream_capturing",side_effect=lambda:f.capture.get() is not None)):
        for mode in ("off","on","one_rank_missing_jit"):
            f.Expert.humming_prefill_silu = ScaledSilu() if mode=="on" or mode=="one_rank_missing_jit" and rank!=size-1 else None
            for n in (67,256,1025,2049):
                prior.exercise(f,rank,size,n,counts)
    Path(store,f"{rank}.json").write_text(json.dumps(counts))
    dist.destroy_process_group()


def cap_sweep():
    counts = Counter()
    f = fixture(counts)
    e = prior.bank(f,0,8,8)
    n,k = 16384,8
    e.num_local_experts,e.global_num_experts = 32,256
    e._expert_map = torch.where(torch.arange(256)<32,torch.arange(256),-1).int()
    e.weights = {name:w.repeat(16,1,1) for name,w in e.weights.items()}
    e.w13_global_scale=e.w13_global_scale.repeat(16)
    e.w2_global_scale=e.w2_global_scale.repeat(16)
    gen=torch.Generator().manual_seed(168)
    ids_rows=torch.arange(n)
    hidden=(torch.randint(-13,14,(n,8),generator=gen).float()/4).bfloat16()
    scores=torch.randn(n,k,generator=gen).softmax(1)
    records=[]
    for m,live in ((0,0),(10844,16384),(6000,18431),(6000,18432),(6000,18433),(12288,98304),(16384,131072)):
        select=ids_rows[(ids_rows*8179)%n < m]
        local=torch.full((n,k),-1,dtype=torch.int32)
        if m:
            pos=torch.arange(live)
            local[select[pos%m],pos//m]=((pos%m+(pos//m)//2)%32).int()
        ci=local[select]
        original=list(f.env["_compact_prefill_windows"](ci,32,256,live,n))
        naive=list(f.env["_compact_prefill_windows"](ci,32,256,live))
        assert sum(t[2] for t in original)==live
        if live: assert (len(original)==1)==(live<=18432)
        ref,_,_=f.reference(e,hidden,local,scores)
        for scaled in (False,True):
            e.humming_prefill_silu=ScaledSilu() if scaled else None
            allocation=torch.full((m+6,8),187,dtype=torch.bfloat16)
            inout=allocation[3:3+m]
            inout.copy_(hidden[select]);f.phase.records.clear()
            f.env["_grouped_humming_and_combine_impl"](
                e,inout,ci,scores[select],inout,local_route_count=live,window_token_count=n)
            assert torch.equal(inout.view(torch.int16),ref[select].view(torch.int16)),(m,live,scaled)
            assert (allocation[:3]==187).all() and (allocation[3+m:]==187).all()
            counts["inplace_bf16_outputs"]+=inout.numel()
            counts["untouched_guards"]+=48
        records.append({"owned_tokens":m,"live_slots":live,"preserved_routes_by_window":[r[2] for r in original],
                        "naive_compact_windows":len(naive)})
    return dict(counts=counts,cases=records)


if __name__ == "__main__":
    torch.set_num_threads(1)
    result={"torch":torch.__version__,"full_geometry":cap_sweep(),"gloo":{}}
    for size in (2,4,8):
        with tempfile.TemporaryDirectory() as t:
            mp.start_processes(worker,(size,"file://"+str(Path(t)/"init"),t),nprocs=size,join=True,start_method="fork")
            result["gloo"][size]=sum((Counter(json.loads((Path(t)/f"{r}.json").read_text())) for r in range(size)),Counter())
    print(json.dumps(result,indent=2))
