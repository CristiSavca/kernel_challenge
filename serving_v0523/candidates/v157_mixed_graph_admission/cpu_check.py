"""Exercise the exported policy against pinned SGLang methods and actual archived capture shapes."""
import ast
import bisect
from collections import Counter, namedtuple
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum, auto
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace as NS
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SG = ROOT / "serving_v0523/sglang"
ARCHIVE = ROOT / "experiments/modal_node/20261113T204343Z-bench"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
ShapeKey = namedtuple("ShapeKey", "size stream_idx variant_label attention_variant", defaults=[None,None,None])
Backend = NS(BREAKABLE="breakable",FULL="full")
get_parallel = lambda: PARALLEL
get_schedule = lambda: SETTINGS
is_cuda = lambda: True
_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2
SOURCES = {}


def bind(path, cls, obj, names):
    name = "python/sglang/srt/model_executor/" + path + ".py"
    src = subprocess.check_output(["git","-C",SG,"show",PIN+":"+name], text=True)
    nodes = next(n.body for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name==cls)
    for fn in (n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names):
        SOURCES[cls+"."+fn.name] = sha256(ast.get_source_segment(src,fn).encode()).hexdigest()
        module = ast.Module(body=[fn],type_ignores=[])
        scope = globals().copy()
        exec(compile("from __future__ import annotations\n"+ast.unparse(module),name,"exec"),scope)
        setattr(obj,fn.name,scope[fn.name])


class ForwardMode(IntEnum):
    EXTEND=1
    TARGET_VERIFY=2
    MIXED=3
    def is_target_verify(self):return self == self.TARGET_VERIFY


class CaptureHiddenMode(IntEnum):
    NULL=0
    FULL=2


class DFlashVerifyInput(NS): pass
class BaseRunner(NS): pass
class PrefillCudaGraphRunner(BaseRunner): pass
class DecodeCudaGraphRunner(BaseRunner): pass
class GraphBackend(NS): pass


def load():
    bind("runner/base_cuda_graph_runner","BaseCudaGraphRunner",BaseRunner,{"_pad_to_bucket"})
    bind("runner/prefill_cuda_graph_runner","PrefillCudaGraphRunner",PrefillCudaGraphRunner,
         {"can_run_graph","can_replay_locally","_shape_key","_has_inactive_dp_rank","_has_uncapturable_chunked_prefix","_uses_eager_prefill_tail"})
    bind("runner/decode_cuda_graph_runner","DecodeCudaGraphRunner",DecodeCudaGraphRunner,
         {"can_run_graph","_resolve_lora_variant","_resolve_attention_variant","_capture_graph_size","_make_graph_key"})
    bind("runner_backend/full_cuda_graph_backend","FullCudaGraphBackend",GraphBackend,{"can_run"})
    src="\n".join(s[1:] for s in (HERE/"delta.diff").read_text().splitlines() if s.startswith("+") and not s.startswith("+++"))+"\n"
    body=[n for n in ast.parse(src).body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(ast.Module(body=body,type_ignores=[]),"exported_v157.py","exec"),globals())


class TensorShape:
    def __init__(self,*dims):self.shape=dims;self.device=torch.device("cpu")
    def __len__(self):return self.shape[0]


def model(grids):
    global SETTINGS,PARALLEL
    SETTINGS=NS(max_prefill_tokens=16384,chunked_prefill_size=16384)
    PARALLEL=NS(dp_size=1,pp_size=1,attn_cp_size=1,attn_dcp_size=1,tp_group=NS(world_size=1))
    pre,ver=grids
    mr=NS(page_size=1,is_draft_worker=False,kv_index_translator=object(),req_to_token_pool=NS(req_to_token=TensorShape(66,131088)),
          spec_algorithm=NS(is_dflash=lambda:True,is_ngram=lambda:False))
    p=PrefillCudaGraphRunner(model_runner=mr,max_bs=64,max_num_tokens=pre[-1],capture_num_tokens=pre,
        prefill_backend_name=Backend.BREAKABLE,_is_full_backend=False,enable_cp_bcg_capture=False,
        _capture_chunked_prefix=False,_captured_attn_metadata_max_bs=None,enable_lora=False,capture_hidden_mode=2,
        max_seq_len=None,max_context_size=None,has_mha_companion_layers=True,_backend_can_run_prefill_cuda_graph=None,
        backend=GraphBackend(_graphs={ShapeKey(size=x) for x in pre}))
    v=DecodeCudaGraphRunner(model_runner=mr,captured_req_width=8,max_bs=ver[-1],capture_bs=ver,ragged_verify_mode=False,
        enable_pdmux=False,require_mlp_tp_gather=False,is_encoder_decoder=False,enable_two_batch_overlap=False,
        require_mlp_sync=False,record_nolora_graph=False,attention_graph_variants=None,disable_padding=False,
        capture_hidden_mode=2,backend=GraphBackend(_graphs={ShapeKey(size=x) for x in ver}))
    mr.prefill_cuda_graph_runner=p;mr.decode_cuda_graph_runner=v
    return mr


def views(mr, p,b,q):
    def view(mode, ids, widths, prefix):
        n=sum(widths); rows=TensorShape(n)
        return NS(batch_size=len(ids),rids=ids,input_ids=rows,positions=rows,out_cache_loc=rows,
            kv_loc_plan=NS(_source=mr.kv_index_translator),forward_mode=mode,capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=None,req_pool_indices_cpu=torch.tensor([i+1 for i in ids]),seq_lens_cpu=torch.tensor(prefix),extend_seq_lens_cpu=widths,extend_prefix_lens_cpu=prefix,return_logprob=False,is_prefill_only=False,
            token_indices_to_pool=None,contains_mm_inputs=lambda:False,global_num_tokens_cpu=None, input_embeds=None,
            replace_embeds=None,lora_ids=None)
    P,T=list(range(p)),list(range(p,p+b))
    lens=[q-p+1]+[1]*(p-1)
    f=view(ForwardMode.EXTEND,P,lens,[101]*p)
    v=view(ForwardMode.TARGET_VERIFY,T,[8]*b,[109]*b)
    v.spec_info=DFlashVerifyInput(draft_token_num=8,num_tokens_per_req=8,topk=1,custom_mask=None,ragged_verify_layout=None)
    m=view(ForwardMode.EXTEND,P+T,lens+[8]*b,[101]*p+[109]*b)
    reqs=[NS(rid=i,is_retracted=False,finished=lambda:False,kv=NS(req_pool_idx=i+1,kv_committed_len=101,kv_allocated_len=117)) for i in T]
    return f,v,m,reqs


def checks():
    stats=json.loads((ROOT/"experiments/iterations/20261118-serving-mixed-graph-capacity.json").read_text())
    grids=stats["target_prefill_graph_rows"],stats["target_verify_graph_requests"]
    mr=model(grids); log=next(ARCHIVE.glob("*v41__server.log")).read_text()
    load=json.loads(next(ARCHIVE.glob("*load.json")).read_text()); order=[r["concurrency"] for r in load["rungs"] for _ in range(r["reps"])]
    pos=-1; counts={n:Counter() for n in order}
    for line in log.splitlines():
        if "Cache flushed successfully!" in line:pos+=1
        found=re.search(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+)",line)
        if pos<0 or not found:continue
        p,q,b=map(int,found.groups())
        if not b:continue
        o=counts[order[pos]]; o["opportunities"]+=1; plan=plan_mixed_dflash_target(mr,*views(mr,p,b,q))
        if plan:
            o["admitted"]+=1;o["pairs"]+=b;o["saved_static_target_rows"]+=plan.prompt_rows+plan.verify_rows-plan.mixed_rows
    for n,s in counts.items():
        approx=stats["rows"][str(n)]["policies"].get("same_budget_whole_cohort",{})
        assert s["admitted"]==approx.get("complete_cohort_passes",0) and s["pairs"]==approx.get("tail_pairs",0)
        assert s["saved_static_target_rows"]==-approx.get("total_target_body_row_change",0)
    faults=["prompt_mode","tree","ragged","width","stale_lengths","request_order","duplicate_rid","tail_req_order",
            "foreign_plan","missing_slot","headroom","context","slot_zero","duplicate_pool_row","retracted","finished",
            "dp","cp","hidden","wrong_logits_mode","encoder","chunks","missing_prefill_graph","missing_decode_variant",
            "uncaptured_shapes","padding_disabled","per_rank_attention","oversized_cohort","no_shared_tail",
            "missing_prefix","stale_prefix","invalid_future_prefix","prompt_prefix_mismatch","truncated_physical_view","missing_mirror","mixed_mirror_for_other_slot","bad_prompt_slot"]
    for fault in faults:
        mr=model(grids); f,v,m,req=views(mr,1,3,582); p=mr.prefill_cuda_graph_runner; d=mr.decode_cuda_graph_runner
        assert plan_mixed_dflash_target(mr,f,v,m,req)
        if fault=="prompt_mode":f.forward_mode=ForwardMode.MIXED
        elif fault=="tree":v.spec_info.custom_mask=object()
        elif fault=="ragged":d.ragged_verify_mode=True
        elif fault=="width":v.spec_info.num_tokens_per_req=7
        elif fault=="stale_lengths":m.extend_seq_lens_cpu[-1]=9
        elif fault=="request_order":m.rids=m.rids[::-1]
        elif fault=="duplicate_rid":m.rids[-1]=m.rids[-2]
        elif fault=="tail_req_order":req.reverse()
        elif fault=="foreign_plan":v.kv_loc_plan._source=object()
        elif fault=="missing_slot":v.out_cache_loc=None
        elif fault=="headroom":req[0].kv.kv_allocated_len-=1
        elif fault=="context":req[0].kv.kv_allocated_len=200000
        elif fault=="slot_zero":req[0].kv.req_pool_idx=0
        elif fault=="duplicate_pool_row":req[0].kv.req_pool_idx=req[-1].kv.req_pool_idx
        elif fault=="retracted":req[-1].is_retracted=True
        elif fault=="finished":req[-1].finished=lambda:True
        elif fault=="dp":PARALLEL.dp_size=2
        elif fault=="cp":p.enable_cp_bcg_capture=True
        elif fault=="hidden":d.capture_hidden_mode=CaptureHiddenMode.NULL
        elif fault=="wrong_logits_mode":m.token_indices_to_pool=object()
        elif fault=="encoder":d.is_encoder_decoder=True
        elif fault=="chunks":SETTINGS.chunked_prefill_size=590
        elif fault=="missing_prefill_graph":p.backend._graphs.clear()
        elif fault=="missing_decode_variant":d.record_nolora_graph=True
        elif fault=="uncaptured_shapes":p.max_num_tokens=600
        elif fault=="padding_disabled":d.disable_padding=True;d.backend._graphs.clear()
        elif fault=="per_rank_attention":d.attention_graph_variants=NS(select=lambda fb:"small-unrecorded")
        elif fault=="oversized_cohort":p.max_bs=3
        elif fault=="no_shared_tail":p.prefill_backend_name="tc_piecewise"
        elif fault=="missing_prefix":m.extend_prefix_lens_cpu=None
        elif fault=="stale_prefix":m.extend_prefix_lens_cpu[-1]=100
        elif fault=="invalid_future_prefix":m.extend_prefix_lens_cpu[-1]=110
        elif fault=="prompt_prefix_mismatch":m.extend_prefix_lens_cpu[0]=102
        elif fault=="truncated_physical_view":m.out_cache_loc=TensorShape(604)
        elif fault=="missing_mirror":m.req_pool_indices_cpu=None
        elif fault=="mixed_mirror_for_other_slot":m.req_pool_indices_cpu[-1]=40
        elif fault=="bad_prompt_slot":m.req_pool_indices_cpu[0]=req[0].kv.req_pool_idx
        assert plan_mixed_dflash_target(mr,f,v,m,req) is None,fault
    return counts,faults


def vote_worker(rank,world,uri):
    load();dist.init_process_group("gloo",init_method=uri,rank=rank,world_size=world,timeout=timedelta(seconds=40))
    global PARALLEL
    PARALLEL=NS(tp_group=NS(world_size=world,cpu_group=dist.group.WORLD))
    plan=DFlashMixedGraphPlan(640,24,640)
    for other,expect in [(plan,plan),(None,None),(DFlashMixedGraphPlan(640,24,704),None),(DFlashMixedGraphPlan(576,24,640),None)]:
        assert agree_mixed_dflash_target(plan if rank==0 else other)==expect
    dist.destroy_process_group()


if __name__=="__main__":
    load()
    counts,faults=checks()
    for world in (2,4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.start_processes(vote_worker,(world,"file://"+str(Path(tmp)/"gloo")),nprocs=world,join=True,start_method="fork")
    print(json.dumps(dict(historical_live_prompt_opportunities=counts,negative_cases=faults,pinned_method_sha256=SOURCES,
                          gloo_world_sizes=[2,4,8],gloo_agreement_patterns=4,physics="CPU metadata, actual Gloo; no CUDA runtime"),indent=2))
