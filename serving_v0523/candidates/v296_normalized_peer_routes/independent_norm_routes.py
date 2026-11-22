"""Run the V295 distributed model fixture with V296 norm admission and bank lifetime checks."""
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"v295_published_prompt_routes"))
sys.path.insert(0,str(HERE.parent/"v293_peer_prompt_norm"))
import independent_routes as distributed
import numeric


def normalized(shard, gain, eps):
    inputs = shard.contiguous().view(torch.int16).numpy().view(np.uint16)
    weights = gain.view(torch.int16).numpy().view(np.uint16)
    answer = numeric.reference(inputs,weights,eps)
    return torch.from_numpy(answer.view(np.int16)).view(torch.bfloat16)


def build_fixture(libpath, poison_alias, late_helper):
    # Adapt the frozen *test infrastructure*. Each staged production definition
    # is still parsed and executed directly by distributed.bind.
    code = inspect.getsource(distributed.fixture)
    def replace(old,new):
        nonlocal code
        assert code.count(old)==1,old
        code=code.replace(old,new)

    replace('    parallel = NS()', '''    parallel = NS()
    native_kernel = numeric.load(libpath)
    class HostNorm:
        def run(self, x, gain, idx, out, epsilon):
            assert x.shape == (1,*out.shape) and torch.equal(idx.flatten(),torch.arange(out.shape[0],dtype=torch.int32))
            native_kernel(x.data_ptr(),gain.data_ptr(),idx.data_ptr(),out.data_ptr(),1,*out.shape,1,epsilon)
            state.counts["direct"]+=1

    class Norm:
        hidden_size, eps, training, zero_centered_gamma = WIDTH, 1e-6, False, False
        _forward_pre_hooks = {}
        def __init__(self):
            self.weight = weights(rank)
        def __call__(self,x):
            state.counts["module"]+=1
            return normalized(x,self.weight,self.eps)

    def weights(owner):
        return (torch.arange(WIDTH).remainder(11)/19+.3125+owner/32).bfloat16()

    def functional(x,shape,gain,eps):
        assert shape==(WIDTH,)
        state.counts["functional"]+=1
        return normalized(x,gain,eps)
''')
    replace('    modules.append(module(PACKAGE+"peer_prompt_pack"', '''    humming=next(m for m in modules if m.__name__==PACKAGE+"muse_spark_humming_moe")
    humming._ranked_pool_module=lambda:HostNorm()
    modules.extend((module(PACKAGE+"muse_spark_layers",MuseSpark1xRMSNorm=Norm),
                    module("sglang.srt.batch_invariant_ops",is_batch_invariant_mode_enabled=lambda:False)))
    try_routes=pen["_try_published_routes"]
    def check_clones(workspace,experts,shard,scores,ids,group,out,source_norm=None):
        before=workspace.next_bank
        accepted,returned=try_routes(workspace,experts,shard,scores,ids,group,out,source_norm)
        if not accepted and source_norm is not None:
            raw_bank=workspace.banks[before][0][:shard.shape[0]]
            assert returned.data_ptr()!=raw_bank.data_ptr()
            if state.counts["direct"]:
                assert torch.equal(returned.view(torch.int16),raw_bank.view(torch.int16))
                if poison_alias: returned=raw_bank
                raw_bank.view(torch.int16).fill_(0x2357)
                state.counts["retired_clone"]+=1
        return accepted,returned
    pen["_try_published_routes"]=check_clones
    modules.append(module(PACKAGE+"peer_prompt_pack"''')
    replace('            assert norm is self.post_expert_norm and torch.equal(h.view(torch.int16),state.gold)', '''            assert norm is self.post_expert_norm
            mismatches=int((h.view(torch.int16)!=state.gold).sum())
            assert not mismatches, f"{mismatches} BF16 words differ in normalized {state.case} fallback"''')
    replace('    bind(stage,MODEL,{"_forward_sp_replicated_experts"},men,"MuseSpark1xMoE")\n    model = NS(top_k=K,num_experts=world*2,experts=experts)', '''    methods=["_forward_sp_experts","_forward_sp_replicated_experts"]
    bind(stage,MODEL,set(methods),men,"MuseSpark1xMoE")
    model_type=type("MuseSpark1xMoE",(),{n:men[n] for n in methods})
    men.update(MuseSpark1xMoE=model_type,MuseSpark1xFp4Experts=Experts)
    model=model_type()
    model.top_k,model.num_experts,model.experts=K,world*2,experts
    model.sp_replicate_expert_input=True
    model.pre_expert_norm=Norm()''')
    replace('    with (torch.no_grad(),', '''    with (torch.no_grad(),patch.object(torch.version,"git_version","5c4886908584029761b579af026dcfb627c84070"),
          patch.object(torch.nn.functional,"rms_norm",functional),''')
    replace('state.counts = dict(route_pairs=0,activation_words=0,empty_local=0)',
            'state.counts = dict(route_pairs=0,activation_words=0,empty_local=0,direct=0,module=0,functional=0,retired_clone=0)')
    replace('                words = ((token*5923+torch.arange(WIDTH)*197+epoch*9991)&65535).short()', '''                raw=((token*5923+torch.arange(WIDTH)*197+epoch*9991)%295/43.-3.25).bfloat16()
                raw[::17,0]=-0.0
                words=raw.view(torch.int16)
                gold=torch.cat([normalized(x,weights(owner),1e-6) for owner,x in enumerate(raw.split(state.rows))])''')
    replace('scores[:state.n],words[:state.n]', 'scores[:state.n],gold[:state.n].view(torch.int16)')
    if late_helper:
        replace('        for mode in ("published","legacy","absent"):', '        for mode in ("legacy",):')
        replace('            for epoch,case in enumerate(cases):','            for epoch,case in enumerate(["ragged"]):')
        replace('    modules.append(module(PACKAGE+"peer_prompt_pack",peer_experts_into=pen["_peer_experts_into"],', '''    def replace_after_capture(workspace,experts,shard,scores,ids,group,out,**options):
        native=pen["_peer_experts_into"]
        original=experts._grouped_humming_and_combine_impl
        if rank==0:
            assert options.get("source_norm") is not None
            def later(shard,ids,scores,out,*,peer_plan):
                assert len(peer_plan)==5
                expected=normalized(raw_input,weights(rank),1e-6)
                assert torch.equal(shard.view(torch.int16),expected.view(torch.int16))
                assert not torch.equal(shard.view(torch.int16),raw_input.view(torch.int16))
                state.counts["helper_called"]+=1
                original(shard,ids,scores,out,peer_plan=peer_plan)
            raw_input=shard
            experts._grouped_humming_and_combine_impl=later
            try: return native(workspace,experts,shard,scores,ids,group,out,**options)
            finally: del experts._grouped_humming_and_combine_impl
        return native(workspace,experts,shard,scores,ids,group,out,**options)
    modules.append(module(PACKAGE+"peer_prompt_pack",peer_experts_into=replace_after_capture,''')
        replace('0,retired_clone=0)','0,retired_clone=0,helper_called=0)')
    replace('                men["_forward_sp_replicated_experts"](model,h,ls,li)', '''                model.pre_expert_norm._forward_hooks={0:True} if rank==(epoch+1)%world else {}
                model._forward_sp_experts(h,ls,li,None)
                assert sum(state.counts[k] for k in ("direct","functional","module"))==1, (state.case,state.counts)
                if model.pre_expert_norm._forward_hooks: assert state.counts["module"]==1''')
    scope = dict(vars(distributed),numeric=numeric,normalized=normalized,libpath=libpath,poison_alias=poison_alias)
    exec(compile(code,"V295 distributed fixture with V296 norm oracle","exec"),scope)
    return scope["fixture"]


def worker(rank, world, stage, folder, dll, alias, late_helper):
    distributed.fixture=build_fixture(dll,alias,late_helper)
    distributed.worker(rank,world,stage,folder)


def main():
    stage=Path(sys.argv[1]).resolve()
    fields=("direct","functional","module","retired_clone","route_pairs","activation_words","empty_local")
    cases={}
    alias="--alias" in sys.argv
    late_helper="--late-helper" in sys.argv
    if late_helper: fields+=("helper_called",)
    with tempfile.TemporaryDirectory(prefix="v296-independent-") as folder:
        root=Path(folder)
        dll=numeric.build(stage,root)
        for world in (4,8):
            mp.spawn(worker,args=(world,str(stage),folder,str(dll),alias,late_helper),nprocs=world)
            for rank in range(world):
                for row in json.loads((root/f"{world}-{rank}.json").read_text()):
                    key=f"tp{world}/{row['mode']}/{row['case']}"
                    target=cases.setdefault(key,dict(calls=0,**{k:0 for k in fields}))
                    target["calls"]+=1
                    for k in fields: target[k]+=row[k]
    totals={k:sum(v[k] for v in cases.values()) for k in ("calls",*fields)}
    if late_helper:
        assert (totals["calls"],totals["helper_called"],totals["functional"])==(12,2,2)
    else:
        assert all(totals[k]>0 for k in fields)
    print(json.dumps(dict(torch=torch.__version__,
        sources={p:distributed.hashlib.sha256((stage/p).read_bytes()).hexdigest() for p in
                 (distributed.MODEL,distributed.PEER,distributed.HUM,distributed.SP,numeric.HEADER)},
        counts=totals,cases=cases,scope="Distributed CPU peer mappings and real Gloo exercise staged SP dispatch, route callbacks and asymmetric normalization. Direct arithmetic executes the CUDA body compiled as a host coroutine. The functional oracle implements the pinned ATen reduction with CPU reciprocal square root. GPU streams and final reduction are modeled."),indent=2))


if __name__=="__main__":
    main()
