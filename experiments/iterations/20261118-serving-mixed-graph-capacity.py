"""Replay historical prompt and verification graph capacities using SGLang's pinned bucket selector."""
import ast
import bisect
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SG = ROOT / "serving_v0523/sglang"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
ARCHIVE = ROOT / "experiments/modal_node/20261113T204343Z-bench"
PREFILL = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+)")


def pinned_selector():
    name = "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py"
    src = subprocess.check_output(["git", "-C", SG, "show", BASE + ":" + name], text=True)
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_pad_to_bucket")
    digest = hashlib.sha256(ast.get_source_segment(src, fn).encode()).hexdigest()
    fn.decorator_list.clear()
    env = dict(bisect=bisect, Sequence=Sequence)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), name, "exec"), env)
    return env[fn.name], digest


def capture_grid(log, pattern):
    captures = re.findall(pattern + r"(\[[^\]]+\])", log)
    assert len(captures) == 8 and len(set(captures)) == 1
    grid = json.loads(captures[0])
    assert sorted(set(grid)) == grid
    return grid


def read_trace(log, run_order):
    which = -1
    for line in log.splitlines():
        if "Cache flushed successfully!" in line:
            which += 1
        if which < 0:
            continue
        match = PREFILL.search(line)
        if match is not None:
            assert "TP0 EP0" in line and "cuda graph: True" in line and "#replay-token:" not in line
            yield run_order[which], tuple(int(i) for i in match.groups())
    assert which + 1 == len(run_order)


def analyze():
    server = next(ARCHIVE.glob("*v41__server.log"))
    log = server.read_text()
    load = json.loads(next(ARCHIVE.glob("*load.json")).read_text())
    info = json.loads(next(ARCHIVE.glob("*server_info.json")).read_text())["internal_states"][0]
    prefill = capture_grid(log, r"Capture target prefill CUDA graph begin\. backend=breakable, num_tokens=")
    verify = capture_grid(log, r"Capture target verify CUDA graph end\. bs=")
    width = info["speculative_num_draft_tokens"]
    cap = min(info["max_prefill_tokens"], info["chunked_prefill_size"])
    assert cap == prefill[-1] == 16384 and width == 8 and verify[-1] == info["max_running_requests"] == 32
    assert info["page_size"] == 1 and not info["enable_mixed_chunk"] and not info["enable_layernorm_sp"]
    choose, digest = pinned_selector()

    def verify_rows(requests):
        return width * choose(requests, verify) if requests else 0

    order = [r["concurrency"] for r in load["rungs"] for _ in range(r["reps"])]
    by_load = {}
    for n in sorted(set(order)):
        by_load[n] = dict(observed=Counter(), policies=defaultdict(Counter), capture_changes=defaultdict(Counter))
    for load_size, (prompt_count, prompt, running) in read_trace(log, order):
        rung = by_load[load_size]
        o = rung["observed"]
        o["prompt_passes"] += 1
        assert 0 < prompt_count <= load_size and prompt_count + running <= load_size
        if not running:
            continue
        o["passes_with_running_requests"] += 1
        o["running_request_prompt_pairs"] += running
        p0 = choose(prompt, prefill)
        v0 = verify_rows(running)
        room = min(running, (cap - prompt)//width)
        same = min(running, (p0 - prompt)//width)
        o["prompt_rows_during_opportunities"] += prompt
        o["standalone_verify_rows_if_every_cohort_runs_once"] += v0
        o["static_prompt_rows_during_opportunities"] += p0
        o["pair_positions_above_max_bucket"] += running - room
        options = []
        for tails in range(room + 1):
            p = choose(prompt + width*tails, prefill)
            v = verify_rows(running - tails)
            options.append((p,v,p + v - p0 - v0))
        nontax = max(i for i,opt in enumerate(options) if opt[2] <= 0)
        least = max(i for i,opt in enumerate(options) if opt[2] == min(x[2] for x in options))
        policies = dict(max_context_capacity=room, same_prompt_bucket=same,
                        same_bucket_all_or_none=running if same == running else 0,
                        same_budget_across_both_graphs=nontax,
                        same_budget_whole_cohort=running if nontax == running else 0,
                        smallest_combined_rows=least, one_tail_only=min(room, 1))
        if room and same < room:
            p1 = options[room][0]
            change = rung["capture_changes"][f"{p0} -> {p1}"]
            change["batches"] += 1
            change["tails"] += room
            change["extra_static_prompt_rows"] += p1-p0
        for key, tails in policies.items():
            stat = rung["policies"][key]
            p,v,delta = options[tails]
            stat["paired_passes"] += int(tails > 0)
            stat["tail_pairs"] += tails
            stat["new_verify_rows"] += width*tails
            stat["complete_cohort_passes"] += int(tails == running)
            stat["complete_cohort_pairs"] += tails if tails == running else 0
            stat["larger_prompt_bucket_passes"] += int(p > p0)
            stat["extra_static_prompt_rows"] += p-p0
            stat["change_in_unused_prompt_padding_rows"] += p-p0-width*tails
            stat["standalone_verify_rows_removed"] += v0-v
            stat["total_target_body_row_change"] += delta
            stat["passes_more_combined_rows"] += int(delta>0)
    for load_size,rung in by_load.items():
        emitted=sum(req["completion"] for req in load["requests"] if req["load"]==load_size)
        rung["observed"]["generated_tokens"] = emitted
        for policy in rung["policies"].values():
            policy["percent_generated_at_four_commits"] = round(400*policy["tail_pairs"]/emitted, 3)
    return dict(archive=str(ARCHIVE.relative_to(ROOT)), substrate="eight H200, TP8 V41 with mixed scheduling and LayerNorm SP disabled",
                verified_file=f"{BASE}:python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py",
                selector_sha256=digest, target_prefill_graph_rows=prefill, target_verify_graph_requests=verify,
                width=width, rows=by_load,
                interpretation="Each historical opportunity holds request membership, prompt chunk and both captured grids fixed. The separated comparison assumes exactly one width-eight target verification step on that same cohort. Positive total_target_body_row_change means more padded target-body rows. Rows do not equal time: full-versus-local attention lengths, expert choice, SP, GPU occupancy, draft proposal, logits, cache work, result settlement and later admissions can differ. Four commits is a scenario. The trace contains neither mixed execution nor V64 physical timing.")


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
