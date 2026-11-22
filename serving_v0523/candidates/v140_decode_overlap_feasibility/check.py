"""Replay the pinned DFLASH next-length relay and quantify late-tap FC overlap."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from random import Random
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SERVING = HERE.parent.parent
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
def read_json(path):
    return json.loads(path.read_text())


def source(path):
    return subprocess.check_output(
        ["git", "-C", SERVING / "sglang", "show", f"{PIN}:python/sglang/srt/{path}"],
        text=True,
    )


def method(text, name):
    result = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(result) == 1, name
    return result[0]


def body(text, name):
    return ast.get_source_segment(text, method(text, name))


def bind(text, name, ns):
    node = copy.deepcopy(method(text, name))
    node.decorator_list = []
    mod = ast.Module(body=[node], type_ignores=[])
    exec(compile(mod, name, "exec"), ns)
    return ns[name]


class Event:
    def __init__(self):
        self.records = self.waits = 0

    def record(self):
        self.records += 1

    def wait(self):
        self.waits += 1

    def synchronize(self):
        raise AssertionError("host event wait was issued")


class Rows:
    """An indexable device tensor stand-in whose host-read methods always fail."""
    dtype = "int64"

    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def to(self, dtype):
        assert dtype == self.dtype
        return self

    def __setitem__(self, idx, source_rows):
        for slot, value in zip(idx.values, source_rows.values, strict=True):
            self.values[slot] = value

    def __getitem__(self, idx):
        return Rows(self.values[slot] for slot in idx.values)

    def cpu(self):
        raise AssertionError("host tensor copy was issued")


def replay_relay():
    text = source("managers/overlap_utils.py")
    event = Event()
    ns = dict(_DEBUG_ASSERT=False, _is_hip=False)
    publish, resolve = (bind(text, k, ns) for k in ("publish", "resolve_seq_lens_cpu"))
    rng = Random(3405)
    store = Rows([-1] * 157)
    fm = SimpleNamespace(
        new_seq_lens_buf=store, _relay=None, spec_algo=SimpleNamespace(is_some=lambda: True),
        publish_ready=event, needs_confidence_relay=False, needs_cpu_seq_lens=False,
    )
    active_rows = 0
    iterations = 512
    for iteration in range(iterations):
        n = (1, 4, 16, 32)[iteration % 4]
        indices = rng.sample(range(157), n)
        # Every live slot commits >=1 including the bonus, <= native block width.
        expected = {i: rng.randrange(1, 120000) + rng.randint(1, 8) for i in indices}
        publish(fm, Rows(indices), Rows(expected[i] for i in indices))
        rng.shuffle(indices)
        batch = SimpleNamespace(spec_info=SimpleNamespace(future_indices=Rows(indices)))
        resolve(fm, batch)
        assert batch.seq_lens.values == [expected[i] for i in indices]
        assert batch.seq_lens_cpu is None and batch.seq_lens_sum is None
        active_rows += n
    assert event.waits == event.records == iterations
    return dict(cycles=iterations, shuffled_request_rows=active_rows,
                producer_events=event.records, device_waits=event.waits, host_waits_or_reads=0)


def dispatch():
    worker = source("speculative/dflash_worker_v2.py")
    forward = body(worker, "forward_batch_generation")
    verify = forward[forward.index("# --- 2) Target verify."):]
    ordered = ["self.target_worker.forward_batch_generation(", "target_lp.deferred_argmax(",
               "self._accept_block(", "on_publish(new_seq_lens)",
               "self._append_target_hidden_to_draft_kv_by_loc(", "_make_next_draft_input_decode("]
    assert [verify.index(k) for k in ordered] == sorted(verify.index(k) for k in ordered)
    scheduler = source("managers/scheduler.py")
    loop = body(scheduler, "event_loop_overlap")
    launched = loop.index("batch_result = self.run_batch(batch)")
    assert loop.index("if not disable_overlap_for_batch:", launched) < loop.index("pop_and_process()", launched)
    attn = source("layers/attention/flashattention_backend.py")
    assert "self.needs_cpu_seq_lens = self._kv_shard_pool is not None" in attn
    target = source("models/muse_spark.py")
    assert "if layer_idx in self.layers_to_capture:" in target
    assert "aux_hidden_states.append(stream.to(self.operator_dtype))" in target
    replay = source("model_executor/runner/decode_cuda_graph_runner.py")
    assert "output.hidden_states[: self.raw_num_token]" in replay
    patch = (SERVING / "candidates/v64_humming_grouped_sequence_parallel/patch.diff").read_bytes()
    assert hashlib.sha256(patch).hexdigest() == "f6aaf93ce06ccfce9461e06bc3f71dea05dd901c01497c8be31bd2c515da689a"
    assert all(x.encode() not in patch for x in ("managers/scheduler.py", "managers/overlap_utils.py",
                                                "speculative/dflash_worker_v2.py", "flashattention_backend.py"))
    return ordered


def model():
    v64 = read_json(SERVING / "runs/abd217baf0ed49fd.json")["result"]
    h200 = read_json(SERVING.parent / "experiments/runs/b5538a0dd25d46be.json")["response"]["result"]
    records = {r["case"]["m"]: r for r in h200["records"]
               if r["event"] == "benchmark" and r["case"]["layer"] == "pooler"}
    bank_bytes = 8192 * 40960 * 2
    pooler_bytes = 8192 * 4096 * 2
    score = v64["frontier"]["score"]
    rungs = []
    for row in v64["rungs"]:
        users = row["concurrency"]
        m = 8 * users
        r = records[m]
        as_ms = r["baseline_raw_ns"] / 1e6
        boundary_ms = 1000 * row["acceptance"] / row["tok_s_per_user_p50"]
        fc_proxy_ms = as_ms * bank_bytes / pooler_bytes
        target_ms = boundary_ms * (1 - score / 4)
        rungs.append(dict(
            users=users, verify_rows=m, commits=row["acceptance"], p50_decode_tok_s=row["tok_s_per_user_p50"],
            estimated_period_ms=boundary_ms, h200_pooler_raw_us=as_ms*1000,
            h200_pooler_adjusted_us=r["baseline_ns"]/1000, full_fc_proxy_ms=fc_proxy_ms,
            perfect_overlap_decode_speedup_under_proxy=boundary_ms/(boundary_ms-fc_proxy_ms),
            time_to_remove_for_2x_ms=boundary_ms/2, time_to_remove_for_common_4_over_v64_ms=target_ms,
            effective_fc_gb_s_needed_to_cover_common_gain=bank_bytes/1e6/target_ms,
            required_time_over_fc_proxy=target_ms/fc_proxy_ms,
            independent_fc_gflop=2*m*40960*8192/1e9,
            ideal_gather_receive_bytes=m*8192*2*7//8,
            full_fc_hardware_floor_ms=max(bank_bytes / 4.8e12, 2*m*40960*8192/989e12)*1000,
        ))
    # Perfect scheduling of two independent branches saves min(x, y).
    cases = 0
    for lead in [0, 1, 12, 78]:
        for fc in [1, 7, 12, 53]:
            for target in [1, 7, 12, 53]:
                for tail in [0, 1, 3, 80]:
                    serial = lead+fc+target+tail
                    ideal = lead+max(fc, target)+tail
                    assert serial - ideal == min(fc, target)
                    assert serial <= 2*ideal
                    cases += 1
    return dict(score=score, common_gain_needed_for_four=4/score, verified_native_width=8, tap_ids=[1,16,30,44,59],
                layer_count=62, fc_weight_bytes=bank_bytes, pooler_weight_bytes=pooler_bytes,
                overlap_cases=cases, rungs=rungs)


if __name__ == "__main__":
    result = dict(pin=PIN, relay=replay_relay(), verify_order=dispatch(), model=model())
    print(json.dumps(result, indent=2))
