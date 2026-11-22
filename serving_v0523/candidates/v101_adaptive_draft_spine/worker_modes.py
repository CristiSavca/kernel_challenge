"""Run production worker ingress for every flag combination, rank, and capture layout."""
import ast
import itertools
import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import torch


def run(root):
    previous=Path(__file__).resolve().parent.parent/"v99_adaptive_draft_tree/worker_check.py"
    helpers=runpy.run_path(str(previous))
    tree_src=(root/"python/sglang/kernels/ops/speculative/dflash_tree.py").read_text()
    worker_src=(root/"python/sglang/srt/speculative/dflash_worker_v2.py").read_text()
    tree_node=next(x for x in ast.parse(tree_src).body if getattr(x,"name",None)=="DFlashTree")
    scope={"torch":torch,"dataclass":dataclass}
    exec(ast.unparse(tree_node),scope)
    tree_class=scope["DFlashTree"]
    body=helpers["method"](worker_src,"DFlashWorkerV2","forward_batch_generation").body
    choose=helpers["body_between"](body,"draft_tokens = self._draft_block_tokens_buf","grammar_tree =")
    result=ast.parse("return linear_tokens, original_tree, policy_advantage, tree").body
    flags,group=NS(),NS()
    seen_select, seen_bcast, seen_stage=[],[],[]
    group.broadcast=lambda data,src: seen_bcast.append((data,src))
    def select(tree, chain, advantage):
        seen_select.append(advantage.shape)
        return tree_class(tree.data.clone())
    choose=helpers["make_function"]("choose",choose+result,"self,bs,block_ids,draft_next,draft_out,batch,draft_input",
        {"get_spec":lambda:flags,"get_dp_tp_group":lambda:group,"select_dflash_tree":select,"torch":torch})
    cases=0
    for adaptive,spine,root_rank,eligible,has_prior,padded in itertools.product((False,True),repeat=6):
        b,w=3,8
        cap=b+int(padded)
        flags.speculative_dflash_adaptive_tree=adaptive
        flags.speculative_dflash_adaptive_spine=spine
        group.rank_in_group=0 if root_rank else 1
        group.world_size=2
        storage=tree_class(torch.arange(4*cap*w).reshape(4,cap,w))
        worker=NS(_draft_block_tokens_buf=torch.zeros((cap,w),dtype=torch.int64),_tree_verify_backend=NS(
                     stage_dflash_tree=lambda *args:seen_stage.append(args)),_draft_sampler=NS(tree_out=storage),
                  _can_skip_verify_logits=lambda *a:True)
        shape=(b,w) if spine else (b,)
        prior=torch.ones(shape,dtype=torch.int32) if has_prior else None
        seen_select.clear();seen_stage.clear();seen_bcast.clear()
        chain,original,state,proposed=choose(worker,b,torch.arange(b*w).reshape(b,w),
             torch.arange(b*(w-1)).reshape(b,w-1),NS(can_run_graph=eligible),
             NS(sampling_info=None,reqs=[NS(return_indexer_topk=False,return_routed_experts=False)]*b),
             NS(tree_policy_advantage=prior))
        assert chain.shape==(b,w) and len(seen_stage)==1
        assert len(seen_bcast)==int(eligible)
        if eligible:
            assert seen_bcast[0][0].shape==(4,b,w) and seen_bcast[0][1]==0
            assert proposed is not None and proposed.data.is_contiguous()
            assert all( torch.equal(item,expected) for item,expected in zip(seen_stage[0],(proposed.depths,proposed.ancestors)))
        else:
            assert proposed is None and seen_stage[0]==(None,None)
        if adaptive and root_rank and eligible:
            assert state.shape==shape and state.dtype==torch.int32
            assert bool(state.any())==has_prior
            assert seen_select==([shape] if has_prior else [])
            assert (original is proposed) != has_prior
        else:
            assert state is original is None and not seen_select
        cases+=1
    return {"torch":torch.__version__,"worker_mode_combinations":cases,"root_rank_and_peer":True,
            "scalar_and_matrix_ingress":True,"graph_eligibility_and_padding":True,"real_nccl":False}


if __name__=="__main__":
    print(json.dumps(run(Path(sys.argv[1])),indent=2))
