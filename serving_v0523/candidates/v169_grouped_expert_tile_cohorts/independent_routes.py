"""Exercise rank-local histograms, common transport gates and cohort metadata using CPU PyTorch."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
PREFIX = "python/sglang/srt/layers/muse_spark/"


def source(stage,name):
    path=PREFIX+name+".py"
    if stage:
        return (stage/path).read_text()
    patch=(HERE/"patch.diff").read_text()
    section=patch.split(f"diff --git a/{path} b/{path}\n",1)[1].split("\ndiff --git",1)[0]
    assert "new file mode" in section
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++ "))+"\n"


def main(stage):
    torch.set_num_threads(2)
    bank = source(stage,"prefill_expert_tiles")
    ns={}
    exec(compile(bank,"<V169 cohort>","exec"),ns)
    transport=source(stage,"sparse_prompt_exchange")
    nodes = [n for n in ast.parse(transport).body if isinstance(n,ast.FunctionDef) and n.name in ("_group_width","_prompt_experts_into")]
    gate=nodes[1]
    stop=next(i for i,n in enumerate(gate.body) if isinstance(n,ast.If) and ast.unparse(n.test)=="width")
    ret=ast.parse("answer = counts, width, local_route_count, expert_counts, descriptor").body[0]
    gate.body=gate.body[:stop]+[ast.copy_location(ast.Return(ret.value),gate.body[stop])]
    ns.update(_local_grouped_eligible=lambda e,_: e.eligible, local_humming_expert_ids=lambda e,ids:e.ids)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),"<V169 gate>","exec"),ns)
    gen=torch.Generator().manual_seed(61091)
    state=({136:{}, 152:{}, 176:{}},(32,16),132)
    rank_gates=plans=two=restores=0
    geometries=[]
    for total in (1,127,128,1024,16383,16384):
        # Random, concentrated, no drops and dropped routes; no challenge inputs.
        global_ids=torch.randint(-4,256,(total,8),generator=gen)
        global_ids[::11]=torch.randint(0,32,(len(global_ids[::11]),8),generator=gen)
        local=[]
        for rank in range(8):
            ids=global_ids-32*rank
            local.append(torch.where((0<=ids)&(ids<32),ids,-1).int())
        padded=math.ceil(total/8)*8
        shard=torch.empty((padded//8,1))
        for masks,votes in ((0xff,0xff),(0xa5,0xff),(0xff,0x7f),(0,0xff)):
            descriptor=[]
            for rank in range(8):
                mask=torch.zeros(padded+1,dtype=torch.uint8)
                if (votes>>rank)&1:
                    mask[:total]=(local[rank]>=0).any(dim=1)
                    mask[-1]=1
                descriptor.append(mask)
            stacked=torch.stack(descriptor)
            refs=stacked[:,:padded].view(8,8,padded//8).sum(-1).T.tolist()
            expected_width=ns["_group_width"](refs,padded) if votes==0xff else 0
            for rank in range(8):
                def gather(dst,src,rank=rank):
                    assert torch.equal(src[0],stacked[rank])
                    dst.copy_(stacked)
                group=SimpleNamespace(world_size=8,rank_in_group=rank,all_gather_into_tensor=gather)
                eligible=bool((votes>>rank)&1)
                histogram=bool((masks>>rank)&1) and eligible
                e=SimpleNamespace(ids=local[rank],eligible=eligible,humming_prefill_tiles=state if (masks>>rank)&1 else None,num_local_experts=32)
                counts,width,live,by_expert,_=ns["_prompt_experts_into"](e,shard,None,global_ids,group,None)
                ref=torch.bincount((local[rank]+1).view(-1).long(),minlength=33)[1:].tolist()
                assert counts==refs and width==expected_width and live==(sum(ref) if eligible else 0)
                assert by_expert==(ref if histogram else None)
                rank_gates+=1
                if not histogram or votes!=0xff: continue
                plan=ns["choose_tiles"](state,by_expert)
                if plan is None: continue
                plans+=1
                keys,slots,runs=ns["sort_tile_cohorts"](local[rank],plan,state)
                slots=slots[slots.numel()-live:]
                actual=local[rank].flatten()[slots]
                covered=set()
                for begin,end,offsets,tuning in runs:
                    for expert in range(32):
                        here=slots[begin+offsets[expert].item():begin+offsets[expert+1].item()]
                        if here.numel():
                            assert expert not in covered
                            covered.add(expert)
                            assert torch.equal(here,torch.where(local[rank].flatten()==expert)[0])
                    assert offsets[0]==0 and offsets[-1]==end-begin
                assert covered=={e for e,n in enumerate(ref) if n}
                assert bool(torch.equal(actual,keys[keys.numel()-live:].remainder(32)))
                assert torch.equal(torch.sort(slots).values,torch.where(local[rank].flatten()>=0)[0])
                two+=len(runs)==2
                restores+=len(slots)
        geometries.append(total)
    # A single heavily contended CUDA bin can hit at most this integer, still exactly representable.
    boundary=torch.full((2**24,),-1,dtype=torch.float32)
    boundary[-1]=31
    hist_call=next(n for n in ast.walk(ast.parse(transport)) if isinstance(n,ast.Call) and
                   ast.unparse(n.func)=="torch.histc")
    hist=eval(compile(ast.Expression(hist_call),"<actual V169 histogram>","eval"),
              {"torch":torch,"local_ids":boundary,"n":32})
    assert hist[0].item()==2**24-1 and hist[-1].item()==1 and (hist[1:-1]==0).all()
    assert plans and two
    return dict(torch=str(torch.__version__),source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in
                (("prefill_expert_tiles",bank),("sparse_prompt_exchange",transport))},
                tokens=geometries,rank_gate_comparisons=rank_gates,selected_plans=plans,
                two_cohort_plans=two,matched_live_route_slots=restores,fp32_histogram_test_slots=2**24,
                h200=False)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",type=Path);args=p.parse_args()
    print(json.dumps(main(args.stage),indent=2))
