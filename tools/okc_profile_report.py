#!/usr/bin/env python3
"""Summarize "Prefill stage profile" lines from a node server log (the profiler build, prefill graph disabled).

Each profiled forward logs one line per rank. A forward's time per stage is the mean over ranks; a group's is the
mean over its forwards. Groups are (size class, expert path). Usage: okc_profile_report.py server.log [--json out]
"""
import argparse
import json
import re
import statistics
from collections import defaultdict

LINE = re.compile(
    r"\[(?P<ts>\S+ \S+) TP(?P<rank>\d+)[^\]]*\] Prefill stage profile: (?P<kind>\w+) #(?P<idx>\d+), (?P<tokens>\d+) tokens, "
    r"positions (?P<span>\S*), (?P<layers>\d+) layers, experts=(?P<mode>.*?), total (?P<total>[\d.]+) ms; "
    r"(?P<stages>.*?); notes: (?P<notes>.*)$")
BATCH = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?cuda graph: (\w+), "
                   r"input throughput \(token/s\): ([\d.]+)")
CHECK = re.compile(r"Prefill expert check: (.*)$")
BUCKETS = [
    ("attention kernel", ("attn_kernel_swa", "attn_kernel_full")),
    ("attention projections (QKV, output)", ("attn_qkv_gemm", "attn_oproj_gemm")),
    ("attention norms, RoPE, gate", ("attn_qknorm_rope", "attn_onorm_gate", "attn_tail")),
    ("cross-GPU sum after attention", ("attn_oproj_all_reduce",)),
    ("cross-GPU sum after experts", ("moe_all_reduce",)),
    ("norm + residual boundaries", ("attn_boundary", "ffn_boundary")),
    ("router and expert input projection", ("moe_proj_router",)),
    ("experts", ("moe_experts", "exp_before_dense", "exp_sort_gather", "exp_dequant13", "exp_gemm13",
                 "exp_scale_silu", "exp_dequant2", "exp_gemm2", "exp_scale_scatter")),
    ("unpool and other FFN", ("moe_unpool", "ffn_other")),
]


def pairs(text):
    out = {}
    for item in filter(None, (x.strip() for x in text.split(","))):
        name, value = item.rsplit(" ", 1)
        out[name] = float(value)
    return out


def parse(path):
    forwards = defaultdict(list)  # (kind, idx) -> one record per rank
    batches, checks = [], []
    for line in open(path, errors="replace"):
        if (m := LINE.search(line.rstrip("\n"))):
            d = m.groupdict()
            forwards[(d["kind"], int(d["idx"]))].append({
                "rank": int(d["rank"]), "tokens": int(d["tokens"]), "span": d["span"], "layers": int(d["layers"]),
                "mode": d["mode"], "total": float(d["total"]), "stages": pairs(d["stages"]), "notes": pairs(d["notes"])})
        elif (m := BATCH.search(line)):
            batches.append({"seqs": int(m[1]), "new": int(m[2]), "cached": int(m[3]), "graph": m[4] == "True",
                            "tok_s": float(m[5])})
        elif (m := CHECK.search(line)):
            checks.append(m[1].strip())
    return forwards, batches, checks


def merge(records):
    """One forward: mean over ranks of every stage; notes summed over ranks except the extremes."""
    stages = defaultdict(list)
    for r in records:
        for name, ms in r["stages"].items():
            stages[name].append(ms)
    notes = defaultdict(list)
    for r in records:
        for name, value in r["notes"].items():
            notes[name].append(value)
    first = records[0]
    lo, _, hi = first["span"].partition("..")
    return {"tokens": first["tokens"], "span": first["span"], "prefix": int(lo) if lo else None, "layers": first["layers"],
            "mode": first["mode"], "ranks": len(records),
            "total": statistics.mean(r["total"] for r in records),
            "total_spread": max(r["total"] for r in records) - min(r["total"] for r in records),
            "stages": {k: statistics.mean(v) for k, v in stages.items()},
            "notes": {k: (max(v) if k.startswith(("max", "size", "local")) else min(v) if k.startswith("min") else sum(v))
                      for k, v in notes.items()}}


def bucketed(stages):
    known = {s for _, names in BUCKETS for s in names}
    out = {label: sum(stages.get(s, 0.0) for s in names) for label, names in BUCKETS}
    out["unlisted"] = sum(v for k, v in stages.items() if k not in known)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json")
    a = ap.parse_args()
    forwards, batches, checks = parse(a.log)
    merged = {key: merge(recs) for key, recs in sorted(forwards.items())}
    groups = defaultdict(list)
    for (kind, idx), f in merged.items():
        groups[(kind, f["mode"])].append((idx, f))
    report = {"groups": {}, "forwards": [], "checks": checks}
    for (kind, mode), items in sorted(groups.items()):
        fs = [f for _, f in items]
        total = statistics.mean(f["total"] for f in fs)
        names = sorted({s for f in fs for s in f["stages"]})
        stage = {s: statistics.mean(f["stages"].get(s, 0.0) for f in fs) for s in names}
        buckets = bucketed(stage)
        tokens = statistics.mean(f["tokens"] for f in fs)
        layers = fs[0]["layers"]
        print(f"\n== {kind}, experts={mode}: {len(fs)} forwards, mean {tokens:,.0f} tokens, "
              f"{total:,.1f} ms per forward ({tokens / total * 1e3:,.0f} tokens/s, {total / layers:.2f} ms/layer)")
        for label, ms in buckets.items():
            if ms:
                print(f"   {100 * ms / total:5.1f}%  {ms:8.1f} ms  {ms / layers:6.2f} ms/layer  {label}")
        print("   stages (ms): " + ", ".join(f"{s} {stage[s]:.1f}" for s in sorted(stage, key=stage.get, reverse=True)))
        note_names = sorted({n for f in fs for n in f["notes"]})
        if note_names:
            print("   notes (mean per forward): " + ", ".join(
                f"{n} {statistics.mean(f['notes'].get(n, 0) for f in fs):,.0f}" for n in note_names))
        report["groups"][f"{kind}|{mode}"] = {"forwards": len(fs), "tokens": tokens, "total_ms": total, "layers": layers,
                                             "buckets_ms": buckets, "stages_ms": stage}
    # The same expert path with the sums forced to NCCL and left to the default dispatch.
    for (kind, mode), items in sorted(groups.items()):
        if not mode.endswith(" + sums nccl"):
            continue
        other = groups.get((kind, mode[: -len("nccl")] + "default"))
        if not other:
            continue
        def mean(items, pick):
            return statistics.mean(pick(f) for _, f in items)
        def sums(f):
            b = bucketed(f["stages"])
            return b["cross-GPU sum after attention"] + b["cross-GPU sum after experts"]
        n_tot, d_tot = mean(items, lambda f: f["total"]), mean(other, lambda f: f["total"])
        n_sum, d_sum = mean(items, sums), mean(other, sums)
        print(f"\n== {kind}, {mode[: -len(' + sums nccl')]}: sums NCCL {n_sum:,.1f} ms -> default dispatch {d_sum:,.1f} ms "
              f"({n_sum / d_sum:.2f}x); forward {n_tot:,.1f} -> {d_tot:,.1f} ms ({100 * (d_tot / n_tot - 1):+.1f}%); "
              f"{len(items)} vs {len(other)} forwards; custom_sum_checks per forward "
              f"{mean(items, lambda f: f['notes'].get('custom_sum_checks', 0)):,.0f} vs "
              f"{mean(other, lambda f: f['notes'].get('custom_sum_checks', 0)):,.0f}")
        report["groups"][f"{kind}|{mode}"]["sums_ms"] = n_sum
        report["groups"][f"{kind}|{mode[: -len('nccl')]}default"]["sums_ms"] = d_sum
    print("\n== every profiled forward (ms, mean over ranks)")
    print("kind   #   tokens  prefix   mode                   total  attn-kernel  sums(attn+moe)  experts  rank-spread")
    for (kind, idx), f in merged.items():
        b = bucketed(f["stages"])
        sums = b["cross-GPU sum after attention"] + b["cross-GPU sum after experts"]
        print(f"{kind:12s} {idx:3d} {f['tokens']:7d} {str(f['prefix']):>7s}  {f['mode']:36s} {f['total']:8.1f} "
              f"{b['attention kernel']:10.1f} {sums:13.1f} {b['experts']:10.1f} {f['total_spread']:9.1f}")
        report["forwards"].append({"kind": kind, "idx": idx, **{k: f[k] for k in ("tokens", "span", "mode", "total", "ranks")},
                                   "buckets_ms": b, "notes": f["notes"]})
    full = [b for b in batches if b["new"] == 16384 and b["tok_s"] > 2000]
    if full:
        rates = sorted(b["tok_s"] for b in full)
        print(f"\n== scheduler: {len(full)} full 16,384-token prefill batches, median {statistics.median(rates):,.0f} tokens/s "
              f"(p10 {rates[len(rates) // 10]:,.0f}, p90 {rates[9 * len(rates) // 10]:,.0f}), "
              f"cuda graph {sorted({b['graph'] for b in full})}")
        report["scheduler_full_chunk_tok_s_median"] = statistics.median(rates)
    later = [b for b in batches if 256 <= b["new"] < 8192 and b["cached"] > 20000]
    if later:
        ms = sorted(1e3 * b["new"] / b["tok_s"] for b in later if b["tok_s"] > 0)
        print(f"== scheduler: {len(later)} later-turn prefill batches, median {statistics.median(ms):,.0f} ms between batches")
    for c in checks[:8]:
        print("expert check:", c)
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
