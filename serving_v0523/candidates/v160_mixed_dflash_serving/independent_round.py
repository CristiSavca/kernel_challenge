"""Replay the mixed scheduler selection, actual run_batch and native mixed target on CPU."""
import ast
from contextlib import nullcontext
import dataclasses
from copy import copy
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace as N

import torch

HERE = Path(__file__).resolve().parent
HOST = Path(sys.argv[1])
REL = Path("python/sglang/srt")
HASHES = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source(path):
    raw = path.read_bytes()
    HASHES[path.name] = hashlib.sha256(raw).hexdigest()
    return ast.parse(raw)


bridge = load("review_bridge", HERE.parent / "v159_mixed_verify_forward/bridge_cpu.py")
sched = load("review_sched", HERE / "execution_cpu.py")
ref = bridge.v155
settle = bridge.v156
ref.build("sampling/sampling_batch_info.py","merge_bias_tensor")
pl = bridge.pl
sched.module("sglang.srt.speculative.dflash_mixed_forward",
             start_mixed_prefix_readback=bridge.env["start_mixed_prefix_readback"],
             try_forward_proposed_mixed=bridge.env["try_forward_proposed_mixed"])
bridge.env["MixedDFlashResult"] = sched.MixedDFlashResult
# The actual role result tensors pass through the native copy methods in the
# scheduler harness. Device-to-host is identity for this local CPU replay.
settle.env.update(_async_d2h=lambda value: value, HostAuxiliaryOutput=sched.h.HostAuxiliaryOutput
                  if hasattr(sched.h, "HostAuxiliaryOutput") else N)
settle.env["GenerationBatchResult"] = sched.GenerationBatchResult
class RelayBase:
    dsa_topk_indices = None
    future_dsa_topk_indices_available = False

    def __init__(self, spec_input_type):
        self.spec_input_type = spec_input_type


relay_node = next(n for n in source(HOST / REL / "speculative/dflash_info_v2.py").body
                  if getattr(n, "name", None) == "DFlashDraftInputV2")
relay_env = dict(SpecInput=RelayBase, SpecInputType=N(DFLASH_DRAFT=1),
                 dataclass=dataclasses.dataclass, torch=torch)
exec("from __future__ import annotations\n"+ast.unparse(relay_node),relay_env)
settle.env["DFlashDraftInputV2"] = relay_env["DFlashDraftInputV2"]
relay_env.update(_get_overlap_plan_stream=lambda d:(None,nullcontext()),
                 is_pin_memory_available=lambda dev:False, get_spec=lambda:N(speculative_num_draft_tokens=8),
                 get_exec=lambda:N(mamba=N(enable_mamba_extra_buffer_lazy=False)))
for rel, name in (("mem_cache/allocation_sizing.py","page_aligned_decode_alloc_lens"),
                  ("mem_cache/allocation.py","alloc_for_spec_decode"),
                  ("speculative/spec_utils.py","spec_prepare_for_decode")):
    selected=next(n for n in source(HOST / REL / rel).body if getattr(n,"name",None)==name)
    exec("from __future__ import annotations\n"+ast.unparse(selected),relay_env)
sched.module("sglang.srt.speculative.spec_utils",spec_prepare_for_decode=relay_env["spec_prepare_for_decode"])
sched.Algo.is_dflash_family=lambda self:True


def assign_slots(indices,table,starts,ends,slots,count):
    offset=0
    for idx,start,end in zip(indices.tolist(),starts.tolist(),ends.tolist(),strict=True):
        table[idx,start:end]=slots[offset:offset+end-start].to(table.dtype)
        offset+=end-start
    assert offset==len(slots) and count==len(indices)


relay_env["assign_req_to_token_pool_func"]=assign_slots
def allocate(tree,n):
    assert n==len(tree.free) and not tree.taken
    tree.taken=True
    return tree.free
relay_env["alloc_token_slots"]=allocate
sched.h._is_hip = False


class Allocator:
    page_size = 1
    check_decode_capacity = lambda *args: True
    available_size = lambda self: 1 << 20


def admit_namespace():
    path = HOST / REL / "managers/scheduler_components/mixed_dflash_admission.py"
    ns = dict(copy=copy, DisaggregationMode=N(NULL="none"), SamplingBatchInfo=N(from_schedule_batch=lambda outer,vocab: dataclasses.replace(outer.sampling_info)),torch=torch, dataclasses=dataclasses, hashlib=hashlib, struct=struct,
              BaseTokenToKVPoolAllocator=Allocator, ForwardMode=ref.Mode,
              PrefillCudaGraphRunner=pl.PrefillCudaGraphRunner, DecodeCudaGraphRunner=pl.DecodeCudaGraphRunner,
              get_parallel=lambda: pl.PARALLEL, get_schedule=lambda: pl.SETTINGS)
    body = [node for node in source(path).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


admit = admit_namespace()
sched.module("sglang.srt.managers.scheduler_components.mixed_dflash_admission", admit_mixed_dflash=admit["admit_mixed_dflash"], build_mixed_dflash_batch=admit["build_mixed_dflash_batch"])
astfile = HOST / REL / "managers/scheduler.py"
tree = source(astfile)
admission_block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                       and "speculative_dflash_mixed_prefill" in ast.unparse(n.test)
                       and "admit_mixed_dflash" in ast.unparse(n))
scope = dict(sched.__dict__, chunked_prefill_size=16384,get_spec=lambda: N(speculative_dflash_mixed_prefill=True))
scope["SamplingBatchInfo"] = N(from_schedule_batch=lambda outer, vocab: dataclasses.replace(outer.sampling_info))
exec("def select(self,new_batch,running_batch):\n" + "\n".join(
    "    " + s for s in ast.unparse(admission_block).splitlines()), scope)
select = scope["select"]


def sb_method(name):
    node = next(n for n in tree_sb.body if isinstance(n, ast.ClassDef) and n.name == "ScheduleBatch")
    method = next(n for n in node.body if getattr(n, "name", None) == name)
    namespace = dict(sched.__dict__, torch=torch, strip_beam_tail=lambda b: None,
                     is_pin_memory_available=lambda dev:False, Req=N, get_batch_return_hidden_states_mode=lambda reqs:ref.Capture.NULL,
                     get_exec=lambda: N(mamba=N(enable_mamba_extra_buffer_lazy=False)))
    exec("from __future__ import annotations\n" + ast.unparse(method),namespace)
    return namespace[name]


tree_sb = source(HOST / REL / "managers/schedule_batch.py")
sched.ScheduleBatch.merge_batch = sb_method("merge_batch")
sched.ScheduleBatch.filter_batch = sb_method("filter_batch")
sched.ScheduleBatch.prepare_for_decode = sb_method("prepare_for_decode")
sched.ScheduleBatch.is_empty = lambda self: not self.reqs
fields = {f.name for f in dataclasses.fields(sched.ScheduleBatch)}


def replay(backend, tails, iteration, mirror):
    w, orig_prompt, orig_tail, reference, _, _, entries, steps = settle.prepare(backend, tails, iteration, True)
    runner = w.target_worker.model_runner
    policy = pl.model((list(range(1,129)),[1,2,3,4,8]))
    for name in ("prefill_cuda_graph_runner", "decode_cuda_graph_runner"):
        graph = getattr(policy, name)
        setattr(runner, name, graph)
        graph.model_runner = runner
    runner.page_size = 1
    runner.model_config.is_encoder_decoder = False
    runner.ngram_embedding_manager.engram_hasher = None
    runner.spec_algorithm.is_dflash = lambda: True
    runner.spec_algorithm.is_ngram = lambda: False
    runner.req_to_token_pool = N(req_to_token=torch.zeros((66,128),dtype=torch.int32))
    scheduler = sched.Scheduler(w,mirror)
    scheduler.draft_worker, scheduler.tp_worker = w,w.target_worker
    scheduler.page_size = 1
    scheduler.max_prefill_tokens=16384
    scheduler.enable_hisparse = scheduler.is_mixed_chunk = scheduler.require_mlp_sync = False
    scheduler.dllm_config = None
    scheduler._reqs_finishing_inflight = lambda b: []
    scheduler.tp_group = pl.PARALLEL.tp_group
    allocation = Allocator()
    p, t = len(orig_prompt.reqs), tails
    q, width = orig_prompt.extend_num_tokens, 8
    rng = torch.Generator().manual_seed(100*iteration+t)
    rows = torch.randperm(56,generator=rng)[:p+t]+1
    def normalized(old, first, role):
        ids = rows[first:first+len(old.reqs)]
        rb = sched.ScheduleBatch(**{k:v for k,v in vars(old).items() if k in fields})
        rb.spec_algorithm, rb.device = sched.Algo(), "cpu"
        rb.req_pool_indices, rb.req_pool_indices_cpu = ids, ids.clone()
        rb.model_config = runner.model_config
        rb.enable_overlap = True
        rb.return_hidden_states_mode=ref.Capture.NULL
        rb.token_to_kv_pool_allocator,rb.req_to_token_pool = allocation,runner.req_to_token_pool
        rb.orig_seq_lens = rb.seq_lens.int()
        rb.sampling_info = settle.sampling()
        rb.sampling_info.temperatures = torch.ones((len(old.reqs),1))
        rb.sampling_info.top_ks = torch.ones((len(old.reqs),1),dtype=torch.int32)
        rb.sampling_info.top_ps = torch.ones((len(old.reqs),1))
        rb.sampling_info.min_ps = torch.zeros((len(old.reqs),1))
        # Both policy stages require no optional penalties or grammar.
        rb.sampling_info.penalizer_orchestrator = N(is_required=False,merge=lambda other:None,filter=lambda keep:None)
        for i, req in enumerate(rb.reqs):
            req.rid = f"{role}-{ids[i].item()}"
            req.multimodal_inputs = None
            req.origin_input_ids = []
            req.return_logprob=False
            req.grammar=None
            req.decode_batch_idx=0
            req.sampling_params=N(top_k=1)
            req.finished = lambda:False
            req.is_retracted = False
            req.seqlen = int(rb.seq_lens[i])+1
            req.kv = N(req_pool_idx=int(ids[i]),kv_committed_len=int(rb.seq_lens[i]),kv_allocated_len=96)
            req.refresh_fill_ids = lambda:None
        return rb
    prompt = normalized(orig_prompt,0,"prompt")
    decode = normalized(orig_tail,p,"tail")
    # The schedule-stream host value lags the preceding verify's publication.
    lag = (torch.arange(t)+iteration).remainder(8)+1
    pre_host = decode.seq_lens.clone()
    true = pre_host + lag
    tokens = reference.input_ids[q:].view(t,width)
    others=iter(sorted(set(range(1,3*(q+t*width)+6))-set(reference.out_cache_loc.tolist())))
    free=[]
    for i, req in enumerate(decode.reqs):
        req.kv.kv_allocated_len=req.kv.kv_committed_len
        desired=reference.out_cache_loc[q+i*width:q+(i+1)*width].tolist()
        n=lag[i].item()
        free.extend([next(others) for _ in range(n)]+desired+[next(others) for _ in range(width-n)])
    decode.tree_cache=N(free=torch.tensor(free),taken=False,token_to_kv_pool_allocator=allocation)
    decode.maybe_evict_swa=lambda:None
    prompt.input_ids = None
    prompt.prefill_input_ids_cpu = reference.input_ids[:q].clone()
    prompt.out_cache_loc = reference.out_cache_loc[:q].clone()
    decode.input_ids = None
    decode.spec_info = settle.env["make_draft_input_v2"](bonus_tokens=tokens[:,0],new_seq_lens=pre_host)
    decode.spec_info.future_indices = decode.req_pool_indices
    decode.new_tokens_required_next_decode = lambda: t*width*2
    scheduler.future_map.new_seq_lens_buf[rows[p:]] = true
    scheduler.future_map.output_tokens_buf[rows[p:]] = tokens[:,0]
    original_mk = runner.kv_index_translator.own_plan
    def plan(fb):
        ans=original_mk(fb);ans._source=runner.kv_index_translator;return ans
    runner.kv_index_translator.own_plan = plan
    w.is_current_decode_proposal=bridge.ownership.is_current_decode_proposal.__get__(w)
    w.consume_decode_proposal=bridge.ownership.consume_decode_proposal.__get__(w)
    def native_draft_boundary(b):
        assert torch.equal(b.seq_lens,true) and torch.equal(b.spec_info.bonus_tokens,tokens[:,0])
        pos=b.seq_lens[:,None]+torch.arange(width)
        slots=runner.req_to_token_pool.req_to_token[b.req_pool_indices[:,None],pos].long()
        kv_plan=plan(N(out_cache_loc=slots.flatten()))
        kv_plan.req_pool_indices=b.req_pool_indices
        nonce=w._pending_proposal_token=object()
        return bridge.proposal_cls(tokens,pos,b.seq_lens,b.seq_lens_cpu,slots,kv_plan,None,b.spec_info,b,nonce)
    w.propose_decode_block=native_draft_boundary
    w.verify_proposed_decode_block=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("unexpected standalone"))
    runner.ngram_embedding_manager.update_after_decode=lambda **kw:steps.append("prompt")
    outer, remaining=select(scheduler,prompt,decode)
    assert not remaining.reqs and outer.mixed_dflash_roles==(prompt,decode)
    assert decode.tree_cache.taken and all(r.kv.kv_allocated_len==r.kv.kv_committed_len+2*width for r in decode.reqs)
    assert torch.equal(outer.req_pool_indices,rows)
    report=scheduler.run_batch(outer)
    assert isinstance(report,sched.MixedDFlashResult) and len(entries)==1
    assert report.copy_done.readies==1 and scheduler.future_map.publish_ready.records==1
    expected=[1+(iteration+3*i)%width for i in range(t)]
    assert report.decode_result.accept_lens.tolist()==expected
    actual=entries[0]
    assert torch.equal(actual.positions[q:].view(t,width),true[:,None]+torch.arange(width))
    assert torch.equal(actual.kv_loc_plan.slots,reference.out_cache_loc)
    prompt_ids=settle.pred(reference.input_ids[:q])[torch.tensor(prompt.extend_lens).cumsum(0)-1]
    tail_bonus=settle.pred(tokens[torch.arange(t),torch.tensor(expected)-1])
    expected_lens=torch.cat((prompt.seq_lens,true+torch.tensor(expected)))
    assert torch.equal(scheduler.future_map.output_tokens_buf[rows],torch.cat((prompt_ids,tail_bonus)))
    assert torch.equal(scheduler.future_map.new_seq_lens_buf[rows],expected_lens)
    assert not outer.mixed_dflash_roles and torch.equal(outer.spec_info.future_indices,rows)
    # A chunked prompt can leave the running set before the next forward.
    # Filter varying interior rows and verify the real FutureMap lookups once more.
    survivors = [j for j in range(p+t) if (j + iteration) % 3]
    if not survivors:
        survivors = [p+t-1]
    outer.filter_batch(keep_indices=survivors)
    saved = rows[survivors]
    assert torch.equal(outer.spec_info.future_indices,saved)
    outer.forward_mode=ref.Mode.DECODE
    scheduler.future_map.resolve_seq_lens_cpu(outer)
    sched.resolve_forward_inputs(outer,scheduler.future_map)
    assert torch.equal(outer.seq_lens,expected_lens[survivors])
    assert torch.equal(outer.spec_info.bonus_tokens,torch.cat((prompt_ids,tail_bonus))[survivors])
    return q,len(rows),sum(expected),lag.tolist()


if __name__=="__main__":
    for rel in ("speculative/dflash_mixed_forward.py", "speculative/dflash_mixed_graph_admission.py",
                "speculative/dflash_worker_v2.py", "managers/tp_worker.py", "managers/utils.py",
                "model_executor/runner/prefill_cuda_graph_runner.py", "model_executor/forward_batch_info.py",
                "managers/overlap_utils.py", "layers/logits_processor.py"):
        source(HOST / REL / rel)
    manifest=json.loads((HERE/"validation.json").read_text())
    assert hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest()==manifest["patch_sha256"]
    cnt=emitted=0
    with torch.inference_mode():
        for backend in ("FULL","BREAKABLE"):
            for tails in (1,2,4):
                for i in range(8):
                    for mirror in (True,False):
                        _,_,commits,_=replay(backend,tails,i,mirror)
                        cnt+=1;emitted+=commits
    print(json.dumps(dict(rounds=cnt,native_two_width_reservations=cnt,filtered_next_ticks=cnt,
        tail_commits=emitted, source_hashes=HASHES, full_patch_sha256=manifest["patch_sha256"],
        graph_backends=["Full","Breakable"], initial_host_lag_range=[1,8],
        cpu_mirror_cases=cnt//2, no_cpu_mirror_cases=cnt//2, hardware=False,
        notes="Real reserve and scheduler host methods. Synthetic device allocator, CUDA streams and deterministic draft/target bodies."),
        indent=2))
