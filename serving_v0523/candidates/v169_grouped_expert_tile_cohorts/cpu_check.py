"""Exercise actual grouped planning, native row ABI, role exchange and BF16 finalization on CPU."""
from collections import Counter
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
v168 = module("cpu168", here.parent / "v168_streamed_experts_sparse_return/check_prompt.py")
policy = module("tiles", stage/"python/sglang/srt/layers/muse_spark/prefill_expert_tiles.py")
state = ({w: {"w13": w, "w2": w} for w in (136, 152, 176)}, (32, 16), 132)

def fixture(counts):
    f = v168.fixture(counts)
    def plan(tile_state, hist):
        counts["histogram_windows"] += 1
        result = policy.choose_tiles(tile_state, hist)
        counts["selected_"+str(len(result[0]) if result else 0)] += 1
        return result
    f.env.update(choose_tiles=plan, sort_tile_cohorts=policy.sort_tile_cohorts)
    return f

def numerical():
    out = Counter()
    f = fixture(out)
    e = v168.prior.bank(f,0,8,8)
    e.num_local_experts,e.global_num_experts = 32,256
    e.humming_prefill_silu = v168.ScaledSilu()
    e._expert_map = torch.where(torch.arange(256)<32,torch.arange(256),-1).int()
    e.weights = {name:w.repeat(16,1,1) for name,w in e.weights.items()}
    e.w13_global_scale = torch.linspace(.5,2.,32).bfloat16()
    e.w2_global_scale = torch.linspace(.3,1.6,32).bfloat16()
    e.humming_prefill_tiles = state
    rng = torch.Generator().manual_seed(169)
    for n in (128, 1024, 8192, 12288, 16384):
        for dist_name in ("uniform", "remote", "mixed", "window", "gaps"):
            ids = torch.randint(256,(n,8),generator=rng).int()
            if dist_name=="remote": ids.fill_(75)
            if dist_name=="mixed": ids[:n//7,:2]=torch.arange(n//7)[:,None]%17
            if dist_name=="window": ids.remainder_(7)
            if dist_name=="gaps": ids.masked_fill_(ids%3==1,254)
            x=(torch.randint(-12,13,(n,8),generator=rng).float()/4).bfloat16()
            scores=torch.randn(n,8,generator=rng).softmax(1)
            for first in (False,True):
                e._post_expert_norm_weight_first=first
                ref,_,_=f.reference(e,x,ids,scores)
                local=f.env["local_humming_expert_ids"](e,ids)
                owned=(local>=0).any(1);selected=owned.nonzero().flatten()
                hist=torch.bincount(local[local>=0].long(),minlength=32).tolist()
                known=sum(hist)
                a=torch.full((len(selected)+4,8),-21,dtype=torch.bfloat16)
                dest=a[2:2+len(selected)];dest.copy_(x[selected])
                e.gemms.clear()
                f.env["_grouped_humming_and_combine_impl"](
                    e,dest,local[selected],scores[selected],dest,local_route_count=known,
                    window_token_count=n,expert_counts=hist)
                assert torch.equal(dest.view(torch.int16),ref[selected].view(torch.int16)),(n,dist_name)
                assert (a[:2]==-21).all() and (a[-2:]==-21).all()
                for name in ("w13","w2"):
                    native=[bounds for layer,bounds in e.gemms if layer==name]
                    assert sum(b[-1] for b in native)==known
                    assert all(0==b[0] and b==sorted(b) and len(b)==33 for b in native)
                    assert [sum(b[i+1]-b[i] for b in native) for i in range(32)]==hist
                out["cases"]+=1;out["inplace_bf16"]+=dest.numel();out["expert_assignment_values"]+=known
    return out

def worker(rank,size,uri,temp):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",rank=rank,world_size=size,init_method=uri,timeout=timedelta(seconds=40))
    counts=Counter();f=fixture(counts)
    with (torch.no_grad(), patch.object(torch.Tensor,"is_cuda",property(lambda _: True)),
          patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
          patch.object(torch.cuda,"is_current_stream_capturing",side_effect=lambda:f.capture.get() is not None)):
        for mode in ("all", "missing_one"):
            f.Expert.humming_prefill_tiles=state if mode=="all" or rank!=0 else None
            f.Expert.humming_prefill_silu=v168.ScaledSilu()
            for n in (256,1025,2049):
                v168.prior.exercise(f,rank,size,n,counts)
    Path(temp,f"{rank}.json").write_text(json.dumps(counts))
    dist.destroy_process_group()

if __name__=="__main__":
    torch.set_num_threads(1)
    result={"numerical":numerical(),"gloo":{},"physical_H200":False}
    for size in (2,4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.start_processes(worker,(size,"file://"+str(Path(tmp)/"init"),tmp),nprocs=size,join=True,start_method="fork")
            result["gloo"][size]=sum((Counter(json.loads(Path(tmp,f"{r}.json").read_text())) for r in range(size)),Counter())
    print(json.dumps(result,indent=2))
