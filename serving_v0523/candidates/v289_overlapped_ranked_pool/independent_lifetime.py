"""Check V289's two-stream dependencies under adversarial graph schedules."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
MODEL = "python/sglang/srt/models/muse_spark.py"
HUMM = "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
FP4 = "python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py"
GROUP = "python/sglang/srt/layers/communication/muse_spark_gemm_ag.py"


def method(stage, path, name):
    program = ast.parse((stage / path).read_bytes())
    return next(n for n in ast.walk(program) if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node):
    return sorted((c.lineno, ast.unparse(c.func)) for c in ast.walk(node) if isinstance(c, ast.Call))


def enumerate_schedules(fork=True, join=True, read_gather_on_side=False):
    """Two graph executions alias all plan slots; main/side advance independently."""
    streams = {
        "main": [(epoch, op) for epoch in (1,2) for op in ("reclaim", "gather", "consume")],
        "side": [(epoch, op) for epoch in (1,2) for op in ("route", "plan")],
    }
    results = {"schedules": 0, "invalid": 0, "first_invalid": None}
    def visit(done, epochs, trace, failed):
        if all(done[k] == len(v) for k,v in streams.items()):
            results["schedules"] += 1
            if failed:
                results["invalid"] += 1
                if results["first_invalid"] is None:
                    results["first_invalid"] = trace
            return
        for stream, events in streams.items():
            idx = done[stream]
            if idx == len(events):
                continue
            epoch, op = events[idx]
            if op == "route" and fork and epochs.get("reclaim", 0) < epoch:
                continue
            if op == "consume" and join and epochs.get("plan", 0) < epoch:
                continue
            state = epochs.copy()
            invalid = False
            if op == "reclaim":
                # A queued write by the previous allocator owner may still run here.
                state["plan_value"] = 0
            if op == "plan":
                invalid = state.get("route") != epoch
                if read_gather_on_side:
                    invalid |= state.get("gather") != epoch
                state["plan_value"] = epoch
            if op == "consume":
                invalid = state.get("plan_value") != epoch or state.get("gather") != epoch
            state[op] = epoch
            visit(done | {stream:idx+1}, state, trace + [f"{op}{epoch}"], failed or invalid)
    visit({"main":0,"side":0}, {}, [], False)
    return results


def main():
    stage = Path(sys.argv[1])
    sources = {p:sha256((stage/p).read_bytes()).hexdigest() for p in (MODEL,HUMM,FP4,GROUP)}
    run = json.loads((HERE/"replay.json").read_bytes())
    assert sources == run["hashes"]
    body = method(stage, MODEL, "_forward_experts")
    ordered = dict((name, line) for line,name in calls(body))
    required = ("allocate_decode_plan", "side_stream.wait_stream", "try_decode_column_gather",
                "self._routing_op()", "prepare_ranked_pool", "fill_decode_plan", "main_stream.wait_stream", "self.experts")
    assert [ordered[k] for k in required] == sorted(ordered[k] for k in required)
    side_context = next(n for n in ast.walk(body) if isinstance(n, ast.With) and "side_stream" in ast.unparse(n.items[0]))
    assert {"self._routing_op()", "prepare_ranked_pool", "fill_decode_plan"} <= {n for _,n in calls(side_context)}
    assert "try_decode_column_gather" not in {n for _,n in calls(side_context)}
    assert next(n.lineno for n in body.body if isinstance(n,ast.Assign) and ast.unparse(n)=="plan_buffers = None" and n.lineno>ordered["main_stream.wait_stream"]) < ordered["self.experts"]
    fill = method(stage,HUMM,"fill_decode_plan")
    assert [a.arg for a in fill.args.args] == ["layer", "topk_ids", "plan"]
    allowed_data = {"packed.data_ptr","packed.is_contiguous","packed.view","topk_ids.numel",
                    "gain.data_ptr","gain.is_contiguous","humming_decode_eligible","_ranked_pool_module","vars","type"}
    metadata = {name for _,name in calls(method(stage,HUMM,"prepare_ranked_pool"))}
    assert metadata == allowed_data
    assert {name for _,name in calls(method(stage,HUMM,"humming_decode_eligible"))} == {
        "bool","hidden_states.is_contiguous","topk_ids.is_contiguous","topk_ids.numel","torch.compiler.is_compiling"}
    assert "_ranked_pool_module()" in ast.unparse(method(stage,GROUP,"can_capture_decode_columns"))
    grouped = ast.unparse(method(stage,HUMM,"humming_grouped_decode_outputs"))
    assert "if decode_plan is None:" in grouped
    assert "ranked_work is not None and ranked_staging is not None" in ast.unparse(method(stage,FP4,"_forward_w4a16"))
    # Confirm rejected captures never pass a side-built plan to the main numerical operators.
    tested = 0
    for case in run["cases"]:
        if "schedule" not in case:
            continue
        ops = list(map(tuple,case["schedule"]))
        if ("side","plan") in ops:
            assert ops.index(("main","allocate")) < ops.index(("side","wait:main"))
            assert ops.index(("side","plan")) < ops.index(("main","wait:side")) < ops.index(("main","ranked_norm_stage"))
        assert ops.count(("side","plan")) + ops.count(("main","plan")) == 1
        tested += 1
    modes = {
        "native_edges": enumerate_schedules(),
        "without_producer_fork": enumerate_schedules(fork=False),
        "without_main_join": enumerate_schedules(join=False),
        "if_side_helper_reads_packed_payload": enumerate_schedules(read_gather_on_side=True),
    }
    assert modes["native_edges"]["invalid"] == 0
    assert all(v["invalid"] for k,v in modes.items() if k != "native_edges")
    print(json.dumps(dict(source_sha256=sources, production_order=required, metadata_calls=sorted(metadata),
                          checked_paired_rank_calls=tested, adversarial_replays=modes,
                          scope="Host source and schedule replay. No real CUDA graph, allocator or timing measurement."),indent=2))


if __name__ == "__main__":
    main()
