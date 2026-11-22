"""Run live-prefix P2P with poisoned graph padding, including empty ranks and cohorts."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

v185 = load("peer_fixture", next(HERE.parent.glob("v185_*/check_transport.py")))
proj = v185.projection


def run(rank, size, temp):
    torch.set_num_threads(1)
    v185.fixture.install("sglang.srt.batch_invariant_ops", {"is_batch_invariant_mode_enabled": lambda: False})
    dist.init_process_group("gloo",init_method="file://"+str(Path(temp)/"init"),rank=rank,world_size=size)
    base = v185.fixture.Comm(rank)
    group = NS(world_size=size, rank_in_group=rank, pynccl_comm=base, cpu_group=dist.group.WORLD)
    count = Counter()
    hidden, channels = 64, 10
    weight = torch.zeros(hidden,channels,dtype=torch.bfloat16)
    weight[:channels] = torch.eye(channels,dtype=weight.dtype)*(rank+1)
    helper = NS(n_layers=5, layer_out_dim=2, flat_kv_weight_t=weight)
    for rep in range(2):
        for n,pitch in [(size*128,128),(size*128-3,128),(size*128+1,256),
                        (size*128+3,512),(size*128-1,1024),(size*128,1024)]:
            x = torch.arange(n*hidden,dtype=torch.int64).reshape(n,hidden).remainder(11).add(rep).bfloat16()
            local = torch.full((pitch,hidden),float('nan'),dtype=x.dtype)
            start = rank*pitch
            k = min(pitch,max(n-start,0))
            if k:
                local[:k] = x[start:start+k]
            count["zero_owners"] += not k
            ever_warm = getattr(helper,"_paired_kv_peer",None) is not None
            sends = []
            orig_send, native_mm = base.send, torch.mm
            def send(t,peer):
                assert torch.isfinite(t).all(), "padding sent"
                sends.append((t.shape,peer))
                orig_send(t,peer)
            gemms = []
            def mm(a,b,*,out):
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), "padding entered GEMM"
                gemms.append(len(a))
                return native_mm(a,b,out=out)
            with patch.object(base,"send",send), patch.object(torch,"mm",mm), proj.dflash_sp_prefill_scope(None,helper):
                got = proj.project_paired_draft_kv(local,n,group)
            assert got is not None and torch.equal(got,x@weight) and got.shape == (n,channels)
            assert getattr(helper,"_paired_kv_peer").remote_weight.untyped_storage().nbytes() == weight.numel()*weight.element_size()
            assert helper._paired_kv_peer.remote_weight[0,0] == ((rank+size//2)%size)+1
            def volume(width):
                peers=size//width-1
                return (width-1)*pitch*hidden+(peers-int(ever_warm))*hidden*channels + peers*width*pitch*channels
            width=min((v for v in (2,4) if v<size),key=volume)
            b=rank//width*width
            cohort=min(width*pitch,max(n-b*pitch,0))
            peers=size//width-1
            assert gemms == ([cohort]*(1+peers) if cohort else [])
            payload = sum(a.numel()*2 for a,_ in sends)
            wanted = 2*((width-1)*k*hidden + (peers-int(ever_warm))*hidden*channels + peers*cohort*channels)
            assert payload == wanted
            count.update(cases=1,bitwise_bf16_outputs=got.numel(),sent_bytes=payload,zero_cohorts=int(not cohort),
                         cold=int(not ever_warm),warm=int(ever_warm))
    all_counts=[None]*size
    dist.all_gather_object(all_counts,dict(count))
    if rank==0:
        total=Counter()
        for item in all_counts: total.update(item)
        Path(temp,"counts").write_text(json.dumps(total))
    dist.destroy_process_group()


if __name__=="__main__":
    worlds={}
    for s in [4,8]:
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(run,args=(s,tmp),nprocs=s,join=True)
            worlds[s]=json.loads(Path(tmp,"counts").read_text())
    path=STAGE/"python/sglang/srt/layers/dflash_context_projection.py"
    print(json.dumps(dict(projection_source_sha256=sha256(path.read_bytes()).hexdigest(),torch=torch.__version__,
        by_world=worlds,physical_H200=False,real_gloo=True),indent=2))
