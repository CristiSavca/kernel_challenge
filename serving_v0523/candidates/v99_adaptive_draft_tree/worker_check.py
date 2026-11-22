"""Check canonical buffer ownership and every greedy-tree eligibility guard.

Run with a patched SGLang source tree and a CPU Torch interpreter.
"""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def make_function(name, statements, params, globals_):
    src = "from __future__ import annotations\ndef " + name + "(" + params + "):\n"
    src += "\n".join("    " + line for stmt in statements for line in ast.unparse(stmt).splitlines())
    scope = dict(globals_)
    exec(src, scope)
    return scope[name]


def method(src, owner, name):
    cls = next(x for x in ast.parse(src).body if isinstance(x, ast.ClassDef) and x.name == owner)
    return next(x for x in cls.body if getattr(x, "name", None) == name)


def body_between(body, first, last):
    a = next(i for i,x in enumerate(body) if ast.unparse(x).startswith(first))
    b = next(i for i,x in enumerate(body[a:],a) if ast.unparse(x).startswith(last))
    return body[a:b]


def run(root):
    src = (root / "python/sglang/srt/speculative/dflash_worker_v2.py").read_text()
    kernel = (root / "python/sglang/kernels/ops/speculative/dflash_tree.py").read_text()
    tree_ast = next(x for x in ast.parse(kernel).body if getattr(x,"name",None) == "DFlashTree")
    from dataclasses import dataclass
    globals_={"torch":torch,"dataclass":dataclass}
    exec(ast.unparse(tree_ast),globals_)
    Tree=globals_["DFlashTree"]
    gate = make_function("gate",method(src,"DFlashWorkerV2","_can_skip_verify_logits").body,"self,batch,sampling_info",{})
    body = method(src,"DFlashWorkerV2","forward_batch_generation").body
    setup = body_between(body, "draft_tokens = self._draft_block_tokens_buf", "grammar_tree =")
    stage_end = ast.parse("return linear_tokens,original_tree,policy_advantage,tree").body
    flags=NS(speculative_dflash_adaptive_tree=True)
    group=NS(rank_in_group=0,world_size=1)
    stages=[]; selected=[]
    def select(original,chain,advantage):
        selected.append((original.data,chain,advantage))
        copy=original.data.clone()
        n=torch.arange(copy.shape[-1])
        rows=advantage > 0
        copy[0,rows]=chain[rows]
        copy[1,rows]=n
        copy[2,rows]=n-1
        copy[3,rows]=(1 << (n+1))-1
        return Tree(copy)
    choose = make_function("choose", setup+stage_end, "self,bs,block_ids,draft_next,draft_out,batch,draft_input", {
        "torch":torch, "get_spec": lambda: flags,"get_dp_tp_group":lambda:group,"select_dflash_tree":select})
    bs,width=3,8
    info_keys=dict(is_all_greedy=True,has_custom_logit_processor=False,penalizer_orchestrator=None,sampling_mask_batch_indices=None)
    eligible=0; blocked=0; neutral=0; mismatches=0
    with torch.inference_mode():
        for capacity in (bs,bs+1):
            for case in ("yes","neutral","disabled","no_backend","no_sampler","no_tree","eager","logprob","grammar",
                         "route","indexer","selector_sample","stochastic","processor","penalty","penalizer","mask","bias","sampling_mask"):
                selected.clear();stages.clear()
                tree_bank=Tree(torch.arange(4*capacity*width,dtype=torch.int64).reshape(4,capacity,width))
                tree_before=tree_bank.data.clone()
                chain_storage=torch.full((capacity+1,width),-1000,dtype=torch.int64)
                sampler=NS(tree_out=None if case=="no_tree" else tree_bank)
                backend=NS(stage_dflash_tree=lambda *args: stages.append(args))
                worker=NS(_draft_sampler=None if case=="no_sampler" else sampler,
                    _tree_verify_backend=None if case=="no_backend" else backend,_draft_block_tokens_buf=chain_storage,
                    _selector_sample=0 if case=="selector_sample" else None)
                worker._can_skip_verify_logits=lambda b,i: gate(worker,b,i)
                info=NS(**info_keys)
                if case in ("stochastic","processor","penalty","penalizer","mask","bias","sampling_mask"):
                    key,val={"stochastic":("is_all_greedy",False),"processor":("has_custom_logit_processor",True),
                        "penalty":("acc_linear_penalties",0),"penalizer":("penalizer_orchestrator",NS(is_required=True)),
                        "mask":("grammar_mask",0),"bias":("logit_bias",0),"sampling_mask":("sampling_mask_batch_indices",0)}[case]
                    setattr(info,key,val)
                req=NS(return_routed_experts=case=="route",return_indexer_topk=case=="indexer")
                batch=NS(requires_grad=False,sampling_info=info,has_grammar=case=="grammar",return_logprob=case=="logprob",reqs=[req]*bs)
                prior=None if case=="neutral" else torch.tensor([-4,0,6],dtype=torch.int32)
                flags.speculative_dflash_adaptive_tree=case!="disabled"
                ids=torch.arange(bs*width).reshape(bs,width)+20
                nexts=torch.arange(bs*(width-1)).reshape(bs,width-1)+1000
                expected_chain=torch.cat((ids[:,:1],nexts),dim=1)
                linear,original,state,actual=choose(worker,bs,ids,nexts,NS(can_run_graph=case!="eager"),batch,NS(tree_policy_advantage=prior))
                assert torch.equal(linear,expected_chain)
                assert linear.data_ptr()==chain_storage.data_ptr()
                # The folded draft graph can overwrite its own out tensor after this
                # boundary. Learning references only the worker's copy of the chain.
                nexts.fill_(-999);ids.fill_(-998)
                assert torch.equal(linear,expected_chain) and (chain_storage[bs:]==-1000).all()
                if case in ("yes","neutral","disabled"):
                    assert actual is not None
                    if case=="yes":
                        assert prior is state and len(selected)==1 and original is not actual
                        assert torch.equal(actual.tokens[:2],original.tokens[:2]) and torch.equal(actual.tokens[2],linear[2])
                        assert torch.equal(original.data,tree_before[:,:bs])
                        assert actual.data.data_ptr()!=original.data.data_ptr()
                        assert torch.equal(tree_before,tree_bank.data)
                        if capacity>bs:
                            assert original.data.data_ptr()!=tree_bank.data.data_ptr()
                        else:
                            assert original.data.data_ptr()==tree_bank.data.data_ptr()
                        eligible+=1
                    elif case=="neutral":
                        assert original is actual and not selected and state.dtype==torch.int32 and not state.any()
                        neutral+=1
                    else:
                        assert original is state is None and not selected
                        mismatches+=1
                    assert torch.equal(stages[0][0],actual.depths) and torch.equal(stages[0][1],actual.ancestors)
                else:
                    assert actual is original is state is None and not selected
                    assert (not stages) if case=="no_backend" else stages==[(None,None)]
                    blocked+=1
                assert torch.equal(tree_before,tree_bank.data)
    return dict(torch=torch.__version__,graph_buffer_capacities=[3,4],eligible=eligible,neutral=neutral,
                disabled=mismatches,blocked_cases=blocked,
                source="Actual DFlashWorkerV2 greedy/selection block and DFlashTree.first_rows executed with CPU torch; selection transport stubbed.",
                checks=["all greedy guards","padded/nonpadded original storage alias","captured linear output overwritten after canonical copy",
                        "mixtures leave original tree immutable","missing state uses original verification tree","fallback clears state"])


if __name__=="__main__":
    print(json.dumps(run(Path(sys.argv[1])),indent=2))
