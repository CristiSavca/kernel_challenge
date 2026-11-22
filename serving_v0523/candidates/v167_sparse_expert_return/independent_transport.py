"""Exercise the full compact input and return exchange with real CPU Gloo all-to-all."""
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


PARENT = Path(__file__).resolve().parent.parent / "v163_sparse_prompt_exchange/independent_transport.py"
spec = importlib.util.spec_from_file_location("v163_data", PARENT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def worker(rank, size, uri, path):
    torch.set_num_threads(1)
    base.install()
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=size, timeout=timedelta(seconds=50))
    group = base.Group(rank,size)
    base.SP.sp_exit_gather = lambda x,n: base.gather(x,group)[:n]
    counts, widths, seen = Counter(), Counter(), []
    cuda_payload = ModuleType("sglang.kernels.ops.moe.muse_spark_sparse_return")
    def reduce_group(indices,received,previous,output,offset,first):
        active = indices >= 0
        local = indices[active] - offset
        assert sorted(local.tolist()) == list(range(local.numel()))
        assert local.numel() <= received.shape[0]
        result = torch.zeros_like(output,dtype=torch.float32) if first else previous.clone().float()
        for src in range(indices.shape[0]):
            rows = active[src]
            result[rows] += received[indices[src,rows].long()-offset].float()
        output.copy_(result)
        counts["sum_calls"] += 1
        counts["sum_empty"] += int(not local.numel())
    cuda_payload.sum_rank_group = reduce_group
    sys.modules[cuda_payload.__name__] = cuda_payload

    def normal_reduce(partial, group):
        all_parts = base.gather(partial.unsqueeze(0),group)
        value = torch.zeros_like(partial,dtype=torch.float32)
        for part in all_parts:
            value += part.float()
        return torch.nn.functional.pad(value.bfloat16(),(0,0,0,(-len(value)) % size)).chunk(size)[rank]
    base.SP.reduce_scatter_token_rows = normal_reduce

    def local_kernel(expert,x,ids,scores,out,*,local_route_count):
        global_local = base.local_humming_expert_ids(expert,expert.global_ids)
        sparse = x is out
        active = (global_local >= 0).any(1) if sparse else torch.ones(len(global_local),dtype=torch.bool)
        assert local_route_count == int((global_local >= 0).sum())
        assert torch.equal(x,expert.global_x[active]) and torch.equal(ids,global_local[active])
        assert torch.equal(scores,expert.scores[active])
        out.copy_(base.oracle_result(expert,expert.global_x[active],expert.global_ids[active],scores))
        counts["compressed_payloads"] += int(sparse)
    base.Expert._grouped_humming_and_combine_impl = local_kernel

    native = dist.all_to_all_single
    def transfer(out,inp,osplits,isplits,*args,**kwargs):
        assert sum(osplits) == len(out) and sum(isplits) == len(inp)
        seen.append((out.untyped_storage().data_ptr(),inp.untyped_storage().data_ptr(),
                     out.untyped_storage().nbytes(),inp.untyped_storage().nbytes(),len(out),len(inp)))
        counts["zero_sends"] += int(not len(inp))
        counts["zero_receives"] += int(not len(out))
        return native(out,inp,osplits,isplits,*args,**kwargs)
    patterns = ("none","dense","all_same","self","self_other","band","cluster","duplicate","random",
                "hot","distributed_empty","cover3","cover5","cover6","cover7")
    scenarios = [(p,m) for p in patterns for m in ("plain","remap")]
    scenarios += [(p,m) for p in ("self","dense") for m in ("hook","pre","compiled","unsupported")]
    with (patch.object(dist,"all_to_all_single",transfer),
          patch.object(torch.Tensor,"is_cuda",property(lambda _:True)),
          patch.object(torch.cuda,"get_device_capability",lambda _:(9,0))):
        for case,(pattern,mode) in enumerate(scenarios):
            rows = 7 + case % 3
            n = rows * size - case % min(size,3)
            ids,scores,x = base.make_data(n,rows,size,pattern,422+case,mode=="remap")
            expert = base.Expert(rank,size,mode=="remap")
            expert.global_x,expert.global_ids,expert.scores = x[:n],ids,scores
            hook = None
            if rank == size-1:
                if mode == "hook": hook = expert.register_forward_hook(lambda m,a,o:o+1)
                if mode == "pre": hook = expert.register_forward_pre_hook(lambda m,a:(a[0],a[1]*2,a[2],a[3]))
                if mode == "compiled": expert._compiled_call_impl = lambda x,s,i,n:base.oracle_result(expert,x,i,s)+3
                if mode == "unsupported": expert.use_humming = False
            local = base.local_humming_expert_ids(expert,ids)
            needs = base.gather(torch.nn.functional.pad((local >= 0).any(1),(0,size*rows-n))[None],group)
            table = needs.view(size,size,rows).sum(2).T.tolist()
            w = base.ENV["_group_width"](table,rows*size) if mode in ("plain","remap") else 0
            expected = normal_reduce(expert(x[:n],scores,ids,expert.post_expert_norm),group)
            seen.clear()
            result = torch.full((rows,base.HIDDEN),float("nan"),dtype=torch.bfloat16)
            base.ENV["_prompt_experts_into"](expert,x[rank*rows:(rank+1)*rows],scores,ids,group,result)
            if hook: hook.remove()
            assert torch.equal(result.view(torch.int16),expected.view(torch.int16)),(size,rank,mode,pattern)
            rounds = size//w if w else 0
            assert len(seen) == rounds*2
            if rounds:
                assert len({s[1] for s in seen[:rounds]}) <= 1
                assert len({s[0] for s in seen[rounds:]}) == 1
                assert all(s[2]+s[3] <= (rows*size+1)*base.HIDDEN*2 for s in seen)
                for f,b in zip(seen[:rounds],seen[rounds:]):
                    assert f[4:6] == tuple(reversed(b[4:6]))
            widths[w] += 1
            counts["rank_cases"] += 1; counts["output_values"] += result.numel()
    Path(path,f"{rank}.json").write_text(json.dumps([counts,widths]))
    dist.destroy_process_group()


def main():
    output = {"module_sha256":base.HASH, "torch":torch.__version__}
    for size in (2,4,8):
        with tempfile.TemporaryDirectory() as t:
            mp.start_processes(worker,(size,"file://"+str(Path(t)/"init"),t),nprocs=size,join=True,start_method="fork")
            ranks=[json.loads((Path(t)/f"{i}.json").read_text()) for i in range(size)]
        output[f"tp{size}"] = {"counts":sum((Counter(r[0]) for r in ranks),Counter()),"group_widths":ranks[0][1]}
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
