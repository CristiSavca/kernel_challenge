"""Compare one mixed result to independently drained native prefill and decode results."""
import ast
import dataclasses
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
DIR = "python/sglang/srt/"
PATCH = (HERE / "host_delta.diff").read_bytes()
PARENT = (HERE.parent / "v147_integrated_experts_prefill/patch.diff").read_bytes()
assert hashlib.sha256(PARENT).hexdigest() == "7a3d4dc6c0e526b60ca50bc975a6036da7796d8eb05f9e8afb70423164425437"
SOURCE_HASHES = {}
PATCHED = {}


@lru_cache
def original(rel):
    path = DIR + rel
    assert ("diff --git a/" + path + " ").encode() not in PARENT
    return subprocess.check_output(
        ["git", "-C", ROOT / "serving_v0523/sglang", "show", PIN + ":" + path], text=True
    )


def load_delta():
    paths = re.findall(rb"^diff --git a/(\S+) b/\1$", PATCH, re.M)
    assert len(paths) == 2
    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            full = Path(tmp) / path.decode()
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(original(path.decode().removeprefix(DIR)))
        subprocess.run(["git", "apply", "--whitespace=error", "-"], cwd=tmp, input=PATCH, check=True)
        for path in paths:
            PATCHED[path.decode().removeprefix(DIR)] = (Path(tmp) / path.decode()).read_text()


@lru_cache
def parsed(rel):
    return ast.parse(PATCHED.get(rel) or original(rel))


def bind(rel, name, owner=None):
    nodes = parsed(rel).body
    if owner:
        nodes = next(n.body for n in nodes if getattr(n, "name", None) == owner)
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    source = ast.unparse(node)
    SOURCE_HASHES["/".join(x for x in (rel, owner, name) if x)] = hashlib.sha256(source.encode()).hexdigest()
    # The outer scheduler tracer is independent of result processing.
    if owner == "Scheduler":
        node.decorator_list = []
    scope = {}
    exec(compile("from __future__ import annotations\n" + ast.unparse(node), rel, "exec"), globals(), scope)
    return scope[name]


load_delta()
batch_src = "managers/schedule_batch.py"
proc_src = "managers/scheduler_components/batch_result_processor.py"
util_src = "managers/utils.py"
sched_src = "managers/scheduler.py"
ForwardMode = bind("model_executor/forward_batch_info.py", "ForwardMode")
BaseFinishReason = bind(batch_src, "BaseFinishReason")
FINISH_LENGTH = bind(batch_src, "FINISH_LENGTH")
FINISH_MATCHED_TOKEN = bind(batch_src, "FINISH_MATCHED_TOKEN")
FINISH_MATCHED_STR = bind(batch_src, "FINISH_MATCHED_STR")
LogitsProcessorOutput = bind("layers/logits_processor.py", "LogitsProcessorOutput")
GenerationBatchResult = bind(util_src, "GenerationBatchResult")
MixedDFlashResult = bind(util_src, "MixedDFlashResult")
_get_speculative_output_stride = bind(proc_src, "_get_speculative_output_stride")
trace = []
copies = []


def noop(*args, **kwargs):
    return None


def _async_d2h(tensor):
    copies.append(tensor.numel())
    return tensor.clone()


def checkpoint_kv_cache(req, tree):
    trace.append(("checkpoint", req.rid, req.kv.kv_committed_len))


def release_kv_cache(req, tree, checkpoint):
    trace.append(("release", req.rid, req.kv.kv_committed_len, checkpoint))


class Sink:
    def __getattr__(self, name):
        return noop


class ScheduleBatch(NS):
    def __getattr__(self, key):
        return None

    def batch_size(self):
        return len(self.reqs)


ScheduleBatch.copy = bind(batch_src, "copy", "ScheduleBatch")
class Request(NS):
    pass


for name in ("finished", "update_finish_state", "_check_vocab_boundary_finish", "_check_str_based_finish",
             "_check_token_based_finish", "_cap_finished_len_at_max_new_tokens", "update_spec_correct_drafts_histogram"):
    setattr(Request, name, bind(batch_src, name, "Req"))


class Grammar:
    def __init__(self, limit):
        self.tokens = []
        self.limit = limit
        self.finished = False

    def accept_token(self, token):
        assert not self.is_terminated()
        self.tokens.append(token)

    def is_terminated(self):
        return len(self.tokens) >= self.limit


class Event:
    def __init__(self):
        self.readies = 0
        self.waits = 0
        self.fail = False

    def record(self):
        self.readies += 1

    def synchronize(self):
        assert self.readies == 1
        self.waits += 1
        if self.fail:
            raise RuntimeError("simulated copy failure")


class Processor(Sink):
    _get_prefill_hidden_capture_mode = staticmethod(lambda batch: None)
    _validate_pp_skip_output_comm = staticmethod(noop)
    _mamba_prefix_cache_update = staticmethod(noop)
    def __init__(self):
        self.is_generation = self.enable_overlap = True
        self.enable_overlap_mlx = False
        self.tree_cache = NS(supports_mamba=lambda:False)
        self.token_to_kv_pool_allocator = NS(
            free_group_begin=lambda:trace.append(("begin",)),
            free_group_end=lambda:trace.append(("end",)),
        )
        self.output_streamer = NS(stream_output=self.stream)
        self.metrics_reporter = Sink()
        self.metrics_reporter.num_generated_tokens = self.metrics_reporter.forward_ct_decode = 0
        self.model_worker = NS(on_verify_complete_cpu=lambda counts, batch_size:trace.append(("verify",counts,batch_size)))
        self.model_config = NS(self_opened_think_start_ids=None)
        self.draft_worker = object()

    def stream(self, requests, logprob, skip=None):
        visible = [(r.rid, r.output_ids[:r.finished_len]) for r in requests if r is not skip and not r.is_retracted]
        trace.append(("stream", visible))


for name in ("process_batch_result_prefill", "process_batch_result_decode", "_resolve_spec_v2_tokens",
             "_normalize_decode_outputs", "_handle_finish_state_updated_req", "_prepare_kv_cache_release",
             "_mamba_can_cache_finished_req", "_maybe_update_reasoning_tokens", "advance_grammar_fsm",
             "_apply_prefill_grammar", "_accept_grammar_tokens", "snapshot_auxiliary_output_starts",
             "move_logprobs_to_cpu", "materialize_sampling_mask_output", "_maybe_collect_routed_experts",
             "_maybe_collect_indexer_topk", "_maybe_collect_customized_info"):
    setattr(Processor, name, bind(proc_src, name, "SchedulerBatchResultProcessor"))


def get_exec():
    return NS(mamba=NS(enable_mamba_extra_buffer_lazy=False))


get_memory = lambda: NS(enable_hisparse=False)
get_observability = lambda: NS(enable_metrics=False)
get_disagg = lambda: NS(disaggregation_decode_enable_offload_kvcache=False)
BaseSpecWorker = type("BaseSpecWorker", (), {})
get_global_indexer_capturer = lambda: None
flush_trace_batch = noop
DisaggregationMode = NS(PREFILL="prefill")


class Scheduler(Sink):
    process_batch_result = bind(sched_src, "process_batch_result", "Scheduler")
    _advance_pending_grammar = bind(sched_src, "_advance_pending_grammar", "Scheduler")
    def __init__(self, processor):
        self.metrics_reporter = Sink()
        self.batch_result_processor = processor
        self.load_publisher = self.load_inquirer = Sink()
        self.enable_fpm = False
        self.tree_cache = NS(flush_pending_backups=lambda:trace.append(("flush",)))
        self.result_queue = []


class Algo:
    is_dflash = lambda self: True
    is_none = lambda self: False


def request(i, role, kind, accept):
    base = [37] if role == "decode" else []
    stop = 200 + i * 10 + max(0, accept // 2 - 1)
    return Request(
        rid=f"{role}{i}", output_ids=base.copy(), kv=NS(kv_committed_len=30+i,mamba_ping_pong_track_buffer=None),
        sampling_params=NS(max_new_tokens=(2 if kind=="cap" else 64),ignore_eos=False,
                           stop_token_ids=({stop} if kind=="stop" else set()),stop_strs=[],stop_regex_strs=[]),
        inflight_middle_chunks=int(kind=="chunk"), finished_reason=(FINISH_LENGTH(1) if kind=="finished" else None),
        finished_len=(1 if kind=="finished" else None), is_retracted=kind=="retracted", to_finish=None,
        discard_output_reason=(FINISH_LENGTH(0) if kind=="discard" else None), return_sampling_mask=False,
        require_reasoning=False, return_hidden_states=False, return_logprob=False, return_routed_experts=False,
        return_indexer_topk=False, beam_group=None, grammar=(Grammar(2) if kind=="grammar" else None),
        multimodal_inputs=None, tokenizer=None, eos_token_ids=None, vocab_size=2000, time_stats=Sink(),
        spec_verify_ct=0, spec_num_correct_drafts=0, spec_correct_drafts_histogram=[], session=None,
    )


def batch(reqs, mode):
    return ScheduleBatch(reqs=reqs, forward_mode=mode, spec_algorithm=Algo(), has_grammar=any(r.grammar for r in reqs),
                         enable_overlap=True, return_logprob=False, return_hidden_states=False)


def fixture(seed):
    width = (2, 4, 8)[seed % 3]
    lens = [1 + ((seed + i*3) % width) for i in range(1 + seed % 6)]
    pkinds = ("normal", "chunk", "finished", "retracted", "discard", "grammar")
    dkinds = ("normal", "cap", "stop", "grammar", "finished", "retracted")
    pr = [request(i, "prompt", pkinds[(seed+i)%len(pkinds)], 1) for i in range(1 + (seed//3)%4)]
    dr = [request(i, "decode", dkinds[(seed//3+i)%len(dkinds)], a) for i,a in enumerate(lens)]
    pids = torch.tensor([100+i for i in range(len(pr))])
    dids = torch.tensor([200+i*10+j if j < a else 1900 for i,a in enumerate(lens) for j in range(width)])
    pres = GenerationBatchResult(next_token_ids=pids,logits_output=LogitsProcessorOutput(None),can_run_cuda_graph=True)
    dres = GenerationBatchResult(next_token_ids=dids,accept_lens=torch.tensor(lens),speculative_num_draft_tokens=width,
                                 logits_output=LogitsProcessorOutput(None),can_run_cuda_graph=True)
    relay = NS(new_seq_lens=torch.tensor([r.kv.kv_committed_len for r in pr] +
                                         [r.kv.kv_committed_len+a for r,a in zip(dr,lens)]),
               bonus_tokens=torch.cat([pids, dids.reshape(len(dr), width)[torch.arange(len(dr)),torch.tensor(lens)-1]]))
    return batch(pr,ForwardMode.EXTEND),batch(dr,ForwardMode.DECODE),pres,dres,relay


def state(requests):
    return [(r.rid,r.output_ids[:],r.finished_len,r.finished_reason.to_json() if r.finished() else None,
             r.inflight_middle_chunks,r.kv.kv_committed_len,r.spec_verify_ct,r.spec_num_correct_drafts,
             list(r.spec_correct_drafts_histogram),r.grammar.tokens[:] if r.grammar else []) for r in requests]


def run(seed, mixed, early_grammar):
    trace.clear()
    copies.clear()
    p,d,pr,dr,relay = fixture(seed)
    all_reqs = p.reqs + d.reqs
    proc = Processor()
    scheduler = Scheduler(proc)
    if mixed:
        result = MixedDFlashResult.from_role_results(p,d,pr,dr,relay)
        assert result.has_sampled_token_ids and result.new_seq_lens is relay.new_seq_lens
        combined = batch(all_reqs,ForwardMode.MIXED)
        result.copy_done = Event()
        result.copy_to_cpu(False,False)
        assert result.copy_done.readies == 1 and copies == [len(p.reqs),dr.next_token_ids.numel(),len(d.reqs)]
        # Later batch filtering is allowed to replace the original row lists.
        p.reqs.clear()
        d.reqs.reverse()
        if early_grammar:
            scheduler.result_queue = [(combined,result)]
            scheduler._advance_pending_grammar()
        scheduler.process_batch_result(combined,result)
        assert trace.count(("flush",)) == 1
    else:
        for cohort, role in ((p,pr),(d,dr)):
            role.copy_done = Event()
            role.copy_to_cpu(False,False)
            if early_grammar:
                proc.advance_grammar_fsm(role,cohort)
        proc.process_batch_result_prefill(p,pr)
        proc.process_batch_result_decode(d,dr)
    return state(all_reqs), [x for x in trace if x!=("flush",)], proc.metrics_reporter.num_generated_tokens


def expect_error(fn, error):
    try:
        fn()
    except error:
        return
    raise AssertionError("expected rejection")


def adverse():
    tests = 0
    for wrong in ("order","drop","duplicate","mode","algorithm","prompt_shape","decode_shape","accept_shape",
                  "width","bonus","lengths","early_copy","different_graph","async_error","missing_readback"):
        p,d,pr,dr,relay = fixture(5)
        if wrong == "order" or wrong == "drop" or wrong in ("async_error","missing_readback"):
            result = MixedDFlashResult.from_role_results(p,d,pr,dr,relay)
            rows = p.reqs+d.reqs
            if wrong == "order": rows = list(reversed(rows))
            if wrong == "drop": rows = rows[:-1]
            result.copy_done = Event()
            result.copy_to_cpu(False,False)
            result.copy_done.fail = wrong=="async_error"
            if wrong == "missing_readback":
                result.copy_done = None
                dr.accept_lens = torch.empty_like(dr.accept_lens,device="meta")
            before = state(p.reqs+d.reqs)
            expect_error(lambda: result.process_roles(batch(rows,ForwardMode.MIXED),Processor()),RuntimeError)
            assert state(p.reqs+d.reqs)==before
        else:
            if wrong=="duplicate": d.reqs[0]=p.reqs[0]
            if wrong=="mode": p.forward_mode=ForwardMode.TARGET_VERIFY
            if wrong=="algorithm": p.spec_algorithm=NS(is_dflash=lambda:False)
            if wrong=="prompt_shape": pr.next_token_ids=pr.next_token_ids[:,None]
            if wrong=="decode_shape": dr.next_token_ids=dr.next_token_ids[:-1]
            if wrong=="accept_shape": dr.accept_lens=dr.accept_lens[:,None]
            if wrong=="width": dr.speculative_output_stride=0
            if wrong=="bonus": relay.bonus_tokens=relay.bonus_tokens[:-1]
            if wrong=="lengths": relay.new_seq_lens=relay.new_seq_lens[:-1]
            if wrong=="early_copy": dr.copy_done=Event()
            if wrong=="different_graph": dr.can_run_cuda_graph=False
            expect_error(lambda:MixedDFlashResult.from_role_results(p,d,pr,dr,relay),ValueError)
        tests += 1
    return tests


def test_role_copy_flags():
    class SideOutput:
        def __init__(self):
            self.values = torch.tensor([13, 21])
        def map_device_tensors(self, copier):
            self.values = copier(self.values)
        def copy_to_host(self, copier):
            return copier(self.values)

    def transfer(settings, mixed):
        p,d,pr,dr,relay = fixture(9)
        p.return_logprob,p.return_hidden_states,d.return_logprob,d.return_hidden_states = settings
        for child in (pr, dr):
            out = child.logits_output
            out.next_token_logprobs = torch.ones(3)
            out.hidden_states = torch.ones(5)
            out.sampling_mask_output = SideOutput()
            out.auxiliary_device_output = SideOutput()
        copies.clear()
        if mixed:
            group = MixedDFlashResult.from_role_results(p,d,pr,dr,relay)
            group.copy_done = Event()
            group.copy_to_cpu(p.return_logprob or d.return_logprob, p.return_hidden_states or d.return_hidden_states)
            assert group.copy_done.readies == 1
        else:
            for batch,child in ((p,pr),(d,dr)):
                child.copy_done = Event()
                child.copy_to_cpu(batch.return_logprob,batch.return_hidden_states)
                assert child.copy_done.readies == 1
        for child in (pr,dr):
            assert child.auxiliary_host_output.tolist() == [13,21]
            assert child.logits_output.auxiliary_device_output is None
        return copies[:]
    for flags in product((False,True), repeat=4):
        assert transfer(flags,True) == transfer(flags,False)
    return 16


if __name__ == "__main__":
    nreq = 0
    for seed in range(96):
        for early_grammar in (False,True):
            result = run(seed,True,early_grammar)
            assert result == run(seed,False,early_grammar)
            assert all(1900 not in row[1] for row in result[0])
            nreq += len(result[0])
    print(json.dumps(dict(mixed_batch_comparisons=192,request_comparisons=nreq,adverse=adverse(),
                          side_output_copy_configurations=test_role_copy_flags(),
                          delta_sha256=hashlib.sha256(PATCH).hexdigest(),torch=torch.__version__,
                          components="Native finish, grammar, result processors, scheduler dispatcher, D2H ordering; CPU tensors, allocator and CUDA-event boundaries instrumented",
                          source_sha256=SOURCE_HASHES),indent=2))

