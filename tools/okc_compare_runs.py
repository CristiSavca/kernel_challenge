#!/usr/bin/env python3
"""Compare two judged serving runs load by load, and optionally against a predicted score.

Usage:
  okc_compare_runs.py <base.json> <new.json> [--predict 1.057,1.086,1.134,1.200]

Prints measured frontier or indicative check ratios, throughput, TTFT and acceptance, followed by
an optional phase budget estimated from the run's rates and user-supplied
assumptions. The budget reports residual time per estimated verify step.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serving_budget import budget, geomean  # noqa: E402


def load(path):
    with open(path) as source:
        d = json.load(source)
    return d, (d.get("result") or {})


def normalized_points(result):
    """Keep the judge's scored frontier separate from a one-load check."""
    frontier = (result.get("frontier") or {}).get("points", [])
    if frontier:
        return "scored frontier", {point["concurrency"]: point for point in frontier}
    checks = (result.get("vs_stock") or {}).get("rungs", [])
    return "indicative check", {
        point["concurrency"]: {
            "ratio": point.get("at_stock_point"),
            "stock_tok_s_per_gpu": (point.get("stock") or {}).get("tok_s_per_gpu"),
        }
        for point in checks
    }


def percent_change(before, after):
    if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (before, after)):
        return "n/a"
    return f"{100 * (after / before - 1):+.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--predict", default="")
    ap.add_argument("--base-p1", type=float, help="assumed first-turn prefill seconds for the base run")
    ap.add_argument("--new-p1", type=float, help="assumed first-turn prefill seconds for the new run")
    ap.add_argument("--p2", type=float, default=0.15, help="assumed prefill seconds per later turn")
    a = ap.parse_args()
    (db, rb), (dn, rn) = load(a.base), load(a.new)
    print(f"base {os.path.basename(a.base)}: status {db.get('status')} passed {rb.get('passed')} score {(rb.get('frontier') or {}).get('score')}")
    print(f"new  {os.path.basename(a.new)}: status {dn.get('status')} passed {rn.get('passed')} score {(rn.get('frontier') or {}).get('score')}")
    if not rn.get("rungs"):
        print("new run has no rungs; raw result keys:", sorted(rn.keys()))
        print(json.dumps({k: rn.get(k) for k in ("status", "error", "reason", "correctness") if k in rn}, indent=1)[:2000])
        return
    kb, pb = normalized_points(rb)
    kn, pn = normalized_points(rn)
    print(f"ratio sources: base {kb}; new {kn}")
    if kb != "scored frontier" or kn != "scored frontier":
        print("Check ratios are indicative at one load; they do not establish a full serving score.")
    for name, data, result in (("base", db, rb), ("new", dn, rn)):
        print(f"{name} patch {(result.get('patch') or {}).get('sha256')}; flags {json.dumps(data.get('serving_flags', {}), sort_keys=True)}")
    gb = {g["concurrency"]: g for g in rb.get("rungs", [])}
    gn = {g["concurrency"]: g for g in rn["rungs"]}
    pred = [float(x) for x in a.predict.split(",")] if a.predict else []
    predictions = dict(zip(sorted(gn), pred))
    print("\nusers  judge ratio base -> new (change)   raw tok/s/GPU base -> new (change)   predicted  stock tok/s/GPU base/new  first TTFT p90 base -> new   tok/s/user p50  accept")
    same_shape = bool(rb.get("shape")) and rb.get("shape") == rn.get("shape")
    for c in sorted(gb.keys() | gn.keys()):
        b, n = pb.get(c, {}), pn.get(c, {})
        base_rung, new_rung = gb.get(c, {}), gn.get(c, {})
        rb_, rn_ = b.get("ratio"), n.get("ratio")
        complete = (rb.get("passed") is True and rn.get("passed") is True
                    and base_rung.get("truncated") is False and new_rung.get("truncated") is False)
        chg = percent_change(rb_, rn_) if complete and same_shape and kb == kn else "n/a"
        raw_base, raw_new = base_rung.get("tok_s_per_gpu"), new_rung.get("tok_s_per_gpu")
        raw_change = percent_change(raw_base, raw_new) if complete and same_shape else "n/a"
        p = f"{predictions[c]:.3f}" if c in predictions else "  -  "
        print(f"{c:5d}  {rb_} -> {rn_} ({chg})   {raw_base} -> {raw_new} ({raw_change})   {p}      {b.get('stock_tok_s_per_gpu')} / {n.get('stock_tok_s_per_gpu')}"
              f"      {base_rung.get('ttft_first_p90_s')} -> {new_rung.get('ttft_first_p90_s')} s"
              f"    {base_rung.get('tok_s_per_user_p50')} -> {new_rung.get('tok_s_per_user_p50')}"
              f"   {base_rung.get('acceptance')} -> {new_rung.get('acceptance')}")
    if pred:
        if len(pred) == 4 and set(gn) == {1, 4, 16, 32}:
            print(f"predicted score {geomean(pred):.4f}; actual {(rn.get('frontier') or {}).get('score')}")
        else:
            print("Predictions cover selected loads only; no full-score prediction computed.")
    print("\nMeasured repetitions and reported spread by load")
    for name, r in (("base", rb), ("new", rn)):
        rung_stats = []
        for rung in r.get("rungs", []):
            reps, spread = rung.get("reps"), rung.get("spread")
            if reps is None or spread is None:
                continue
            detail = "single rep" if reps == 1 else f"{reps} reps, {spread:.2%} spread"
            rung_stats.append(f"{rung['concurrency']} users ({detail})")
        if rung_stats:
            print(f"{name}  " + ", ".join(rung_stats))

    print(f"\nModeled service shares with later-turn P2 {a.p2:.3f}s. Residual includes all unassigned time.")
    for name, r, p1_override in (("base", rb, a.base_p1), ("new", rn, a.new_p1)):
        try:
            p1, rows = budget(r, p2=a.p2, p1=p1_override)
            shares = " / ".join(f"{100 * x['first_prefill_s'] / x['wall_s']:.0f}%" for x in rows)
            steps = " / ".join(f"{x['residual_step_ms']:.1f}" for x in rows)
            source = "assumption" if p1_override is not None else "one-user rate heuristic"
            print(f"{name}  P1 {p1:.2f}s ({source}), first-prefill shares {shares}, residual ms per estimated step {steps}")
        except (KeyError, TypeError, ValueError) as exc:
            print(f"{name}  budget unavailable ({exc})")
    for name, r in (("base", rb), ("new", rn)):
        c, lt = r.get("correctness") or {}, r.get("later_turns") or {}
        print(f"{name}: correctness passed {c.get('passed')} agreement {c.get('agreement')} decode_kl {c.get('decode_kl_mean')} "
              f"prefill_kl {c.get('prefill_kl_mean')} prefill_top1 {c.get('prefill_top1')} reasons {c.get('reasons')}; "
              f"later turns passed {lt.get('passed')} first {lt.get('first_agreement')} later {lt.get('later_agreement')}; "
              f"gpu memory {r.get('gpu_memory')}; startup {r.get('startup_s')} s")
    print("bounds:", json.dumps((rn.get("correctness") or {}).get("bounds")), json.dumps((rn.get("later_turns") or {}).get("bounds")))


if __name__ == "__main__":
    main()
