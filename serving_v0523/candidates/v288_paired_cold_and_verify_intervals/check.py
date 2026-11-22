"""Pair physical scheduler intervals from the private V41 and V44 eight-H200 runs."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from statistics import mean, median


HERE = Path(__file__).resolve().parent
SERVING = HERE.parents[1]
ROOT = SERVING.parent
V244 = runpy.run_path(str(HERE.parent / "v244_cold_prompt_scaling/check.py"))
PREFILL = V244["REPORT"]
DECODE = re.compile(
    r"TP0 EP0\] Decode batch, #running-req: (\d+), #full token: (\d+),.*?"
    r"accept len: ([\d.]+),.*?cuda graph: (True|False), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
SOURCE_HASHES = {}
CHUNK = 16384
LAYERS = 62


def read_bytes(path):
    data = path.read_bytes()
    SOURCE_HASHES[str(path.relative_to(ROOT))] = sha256(data).hexdigest()
    return data


def read_json(path):
    return json.loads(read_bytes(path))


def run_inputs(label):
    root = ROOT / "experiments/modal_node"
    client_file = next(root.glob(f"*/*{label}__load.json"))
    log_file = next(client_file.parent.glob(f"*{label}__server.log"))
    manifest = read_json(client_file.parent / "manifest.json")
    variant = next(v for v in manifest["session"]["variants"] if v["name"] == label)
    patch = read_bytes(next(HERE.parent.glob(f"{label}_*/patch.diff")))
    assert sha256(patch).hexdigest() == variant["patch_sha256"]
    clients = {(r["rep"], r["turn"]): r for r in read_json(client_file)["requests"] if r["load"] == 1}
    lines = read_bytes(log_file).decode().splitlines()
    settings = ast.literal_eval(next(line.split("server_args=", 1)[1] for line in lines if "server_args={" in line))
    flush = [i for i, line in enumerate(lines) if "TP0 EP0] Cache flushed successfully!" in line]
    segments = [lines[start:end] for start, end in zip(flush, flush[1:] + [len(lines)])]
    assert (settings["tp_size"], settings["decode_log_interval"], settings["speculative_num_draft_tokens"]) == (8, 40, 8)
    assert settings["moe_runner_backend"] == "marlin" and settings["chunked_prefill_size"] == CHUNK
    return dict(settings=settings, manifest=manifest, clients=clients, segments=segments,
                patch_sha256=variant["patch_sha256"], variant_flags=variant["flags"], lines=lines)


def cold_reports(lines):
    reports = []
    for line in lines:
        match = PREFILL.search(line)
        if match is None:
            continue
        _, seqs, live, cached, running, queued, pending, graph, rate = match.groups()
        assert "#replay-token:" not in line
        signature = tuple(map(int, (seqs, live, cached, running, queued, pending))) + (graph == "True",)
        reports.append((signature, int(live) / float(rate)))
        if int(pending) == 0:
            break
    return reports


def cold_pairs(a, b):
    comparisons = []
    eligible_one_user = 0
    for seg, (aseg, bseg) in enumerate(zip(a["segments"][:5], b["segments"][:5])):
        old, new = cold_reports(aseg), cold_reports(bseg)
        assert len(old) == len(new)
        left = [r[0] for r in old]
        right = [r[0] for r in new]
        for chain in (left, right):
            assert all(row[1] == CHUNK for row in chain[:-1])
            assert all(row[-1] for row in chain[:-1])
        assert sum(r[1] for r in left) == sum(r[1] for r in right)
        assert sum(r[2] for r in left) == sum(r[2] for r in right)
        # A reporting gap is assigned to an interior full chunk only if both its
        # neighbors were full, guarding against a one-batch shift in the scheduler.
        interior = range(1, len(old) - 2)
        strict = [i for i in interior if left[i-1:i+2] == right[i-1:i+2]]
        if seg < 3:
            assert left == right
            assert a["clients"][seg,0]["prompt"] == b["clients"][seg,0]["prompt"] == sum(r[1] for r in left)
            assert min(r[2] for r in left) == max(r[2] for r in left) == 0
            eligible_one_user += len(left)-1
        windows = {}
        for name, indices in (("volume_matched", list(interior)), ("three_report_geometry_matched", strict)):
            old_s = [old[i][1] for i in indices]
            new_s = [new[i][1] for i in indices]
            windows[name] = dict(indices=indices, v41_seconds=old_s, v44_seconds=new_s,
                                 summed_ratio=sum(old_s) / sum(new_s))
        comparisons.append(dict(users=1 if seg<3 else 4, rep=seg if seg<3 else seg-3,
                                live_demand=sum(r[1] for r in left), cached_demand=sum(r[2] for r in left),
                                full_chunks=len(left)-1, terminal_chunk=left[-1][1], windows=windows))
    assert eligible_one_user == 15
    return comparisons


def decode_windows(source):
    collected = {}
    for rep, lines in enumerate(source["segments"][:3]):
        turn = -1
        first_report = None
        ordinal = 0
        for line in lines:
            p = PREFILL.search(line)
            if p:
                first_report = None
                if int(p.group(7)) == 0:
                    turn += 1
                    ordinal = 0
                    if turn:
                        req = source["clients"][rep, turn]
                        assert int(p.group(4)) == req["cached"]
                        assert int(p.group(3)) + req["cached"] == req["prompt"]
                continue
            d = DECODE.search(line)
            if d:
                live, full_tokens, accepted, graph, throughput, queue = d.groups()
                assert int(live) == 1 and int(queue) == 0 and graph == "True"
                token_position = int(full_tokens)
                if first_report is not None:
                    # Both printed rates use the same generated-token counter.
                    # Their quotient is the duration of a verification step.
                    collected[rep,turn,ordinal] = {
                        "step_ms": 1000 * float(accepted) / float(throughput),
                        "mean_accepted": float(accepted), "tokens_s": float(throughput),
                        "first_context_report": first_report, "last_context_report": token_position,
                    }
                    ordinal += 1
                first_report = token_position
        assert turn == 12
    return collected


def paired_decode(a, b):
    old, new = decode_windows(a), decode_windows(b)
    candidates = sorted(old.keys() & new.keys())
    pairs = []
    for rep, turn, ordinal in candidates:
        first = a["clients"][rep,turn]
        second = b["clients"][rep,turn]
        if any(first[k] != second[k] for k in ("prompt", "cached", "asked", "completion")):
            continue
        key = (rep,turn,ordinal)
        pairs.append(dict(rep=rep, turn=turn, ordinal=ordinal, prompt=first["prompt"],
                          cached=first["cached"], v41=old[key], v44=new[key]))
    def summarize(source):
        return {k:median(v[k] for v in source) for k in ("step_ms","mean_accepted","tokens_s")}
    return dict(complete_window_counts=[len(old),len(new)], same_request_and_ordinal=len(candidates),
                matched_client_shapes=len(pairs), per_run_all_windows=[summarize(old.values()),summarize(new.values())],
                matched_medians=[summarize([r[k] for r in pairs]) for k in ("v41","v44")],
                pooled_matched_step_ms_ratio=sum(r["v41"]["step_ms"] for r in pairs)/sum(r["v44"]["step_ms"] for r in pairs),
                max_abs_context_endpoint_difference=max(abs(r["v41"]["last_context_report"]-r["v44"]["last_context_report"]) for r in pairs),
                pairs=pairs)


def main():
    a, b = run_inputs("v41"), run_inputs("v44")
    assert a["settings"] == b["settings"] and len(a["settings"]) == 546
    assert a["manifest"]["image"] == b["manifest"]["image"]
    assert a["variant_flags"] == b["variant_flags"]
    assert a["manifest"]["session"]["replica_flags"] == b["manifest"]["session"]["replica_flags"]
    # The archive has no comparable first-turn reply content from V41.
    assert not any("reply" in r for r in a["clients"].values())
    metrics = read_bytes(SERVING / "sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py").decode()
    producer = read_bytes(SERVING / "sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py").decode()
    assert "self.num_generated_tokens / gap_latency" in metrics and "self.spec_num_accept_tokens / self.spec_num_forward_ct" in metrics
    assert "num_accept_tokens=num_generated_tokens" in producer
    comparisons = cold_pairs(a,b)
    micro = read_json(ROOT / "experiments/modal_serving/20261113T065259Z/result.json")
    assert micro["gpu"] == "NVIDIA H200"
    case = next(r for r in micro["cases"] if r["T"] == CHUNK)
    projected_saving = (case["v40_us"] - case["dense384_us"]) * LAYERS / 1e6
    summary = {}
    for users in (1,4):
        windows = [r["windows"]["three_report_geometry_matched"] for r in comparisons if r["users"] == users]
        old = [v for block in windows for v in block["v41_seconds"]]
        new = [v for block in windows for v in block["v44_seconds"]]
        delta = mean(x-y for x,y in zip(old,new))
        summary[users] = dict(samples=len(old), old_mean_s=mean(old), new_mean_s=mean(new),
                              mean_difference_s=delta, median_difference_s=median(x-y for x,y in zip(old,new)),
                              summed_speedup=sum(old)/sum(new), standalone_explained_fraction_if_serial_equal=projected_saving/delta,
                              unexplained_seconds_under_same_control=delta-projected_saving,
                              required_seconds_beyond_v44_for_v41_chunk_factor={str(k):mean(new)-mean(old)/k for k in (3,4)})
    budget = read_json(HERE.parent / "v271_cold_service_envelope/audit.json")
    large = next(r for r in budget["windows"]["v41"] if r["load"] == 32)
    demand = {str(k):large["conditional_uniform_phase_demand"][str(k)]["required_saved_ms_per_layer_16k_equivalent"] for k in (3,4)}
    conditional = {goal:{str(users):v/(1000*summary[users]["mean_difference_s"]/LAYERS) for users in (1,4)} for goal,v in demand.items()}
    result = dict(patches=[a["patch_sha256"],b["patch_sha256"]],image=a["manifest"]["image"],
                  equal_resolved_server_arguments=546, flags=a["variant_flags"],replica_flags=a["manifest"]["session"]["replica_flags"],
                  phase_totals=comparisons,matched_cold_summary=summary,
                  standalone=dict(local_routes=case["local_routes"],v40_us=case["v40_us"],dense384_us=case["dense384_us"],
                                  optimistic_serial_62_layer_saving_s=projected_saving,
                                  dense_fraction_bits_differing_from_stock=case["dense384_frac_diff_elems"]),
                  decode=paired_decode(a,b),uniform_32user_demand_ms_per_layer=demand,
                  required_multiples_of_observed_chunk_delta_if_transferred=conditional,input_sha256=SOURCE_HASHES)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
