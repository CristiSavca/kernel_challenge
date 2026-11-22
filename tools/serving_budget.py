#!/usr/bin/env python3
"""Model per-load serving time and score scenarios from a judged run.

The run records aggregate rates and percentiles. For `c` users, the model
assigns `c * P1` to first-turn prefill and
`c * (turns - 1) * P2` to later-turn prefill. P1 and P2 are seconds of modeled
service time per conversation and per later turn, respectively. The default
P1 uses the difference between reciprocal aggregate and median request rates
at one user as a heuristic. P2 defaults to an assumption of 0.15 seconds.
The residual includes decode and other time outside those estimated prefills.

Score scenarios multiply the published frontier score by the modeled
geometric mean multiplier. They assume fixed work and frontier normalization.
"""

import argparse
import json
import math
from pathlib import Path


def budget(result, p2=0.15, p1=None):
    shape = result["shape"]
    turns, out = shape["turns"], shape["output_mean"]
    if turns < 1 or out <= 0:
        raise ValueError("the run needs positive turn and output token counts")
    if not math.isfinite(p2) or p2 < 0:
        raise ValueError("later-turn prefill time must be finite and non-negative")
    rungs = {rung["concurrency"]: rung for rung in result["rungs"]}
    if not rungs:
        raise ValueError("the run has no rungs")
    if p1 is None:
        if 1 not in rungs:
            raise ValueError("specify --p1 when the run has no one-user rung")
        rung = rungs[1]
        aggregate = rung["tok_s_per_user_e2e"]
        median = rung["tok_s_per_user_e2e_p50"]
        if not all(math.isfinite(rate) and rate > 0 for rate in (aggregate, median)):
            raise ValueError("one-user end-to-end rates must be finite and positive")
        p1 = turns * (out / aggregate - out / median)
        if p1 < 0:
            raise ValueError("the one-user heuristic gives a negative P1; specify --p1")
    if not math.isfinite(p1) or p1 < 0:
        raise ValueError("first-turn prefill time must be finite and non-negative")

    rows = []
    for concurrency, rung in sorted(rungs.items()):
        rate = rung["tok_s_per_user_e2e"]
        acceptance = rung["acceptance"]
        if not all(math.isfinite(value) and value > 0 for value in (rate, acceptance)):
            raise ValueError(f"load {concurrency} needs finite, positive rate and acceptance")
        wall = turns * out / rate
        steps = turns * out / acceptance
        first = concurrency * p1
        later = (turns - 1) * concurrency * p2
        residual = wall - first - later
        if residual < 0:
            raise ValueError(
                f"negative residual at load {concurrency}. Modeled prefill "
                f"{first + later:.3f}s exceeds the reconstructed wall time "
                f"{wall:.3f}s. Adjust --p1 or --p2."
            )
        step_ms = 1e3 * residual / steps
        rows.append(
            {
                "c": concurrency,
                "wall_s": wall,
                "first_prefill_s": first,
                "later_prefill_s": later,
                "residual_s": residual,
                "steps": steps,
                "residual_step_ms": step_ms,
                # Preserve older callers while identifying these as estimates.
                "decode_s": residual,
                "step_ms": step_ms,
            }
        )
    return p1, rows


def predict(rows, prefill_speedup, later_speedup, step_saving_ms, *, residual_speedup=1.0):
    speedups = (prefill_speedup, later_speedup, residual_speedup)
    if not all(math.isfinite(speedup) and speedup > 0 for speedup in speedups):
        raise ValueError("phase speedups must be finite and positive")
    if len(step_saving_ms) != len(rows) or not all(map(math.isfinite, step_saving_ms)):
        raise ValueError("specify one finite step saving per rung, in load order")

    multipliers = []
    for row, saving in zip(rows, step_saving_ms):
        residual = row["residual_s"] / residual_speedup - row["steps"] * saving / 1e3
        if residual < 0:
            raise ValueError(f"step saving gives a negative residual at load {row['c']}")
        modeled_wall = (
            row["first_prefill_s"] / prefill_speedup
            + row["later_prefill_s"] / later_speedup
            + residual
        )
        if modeled_wall <= 0:
            raise ValueError(f"modeled wall time must be positive at load {row['c']}")
        multipliers.append(row["wall_s"] / modeled_wall)
    return multipliers


def geomean(values):
    return math.exp(math.fsum(map(math.log, values)) / len(values))


_PHASES = ("first_prefill_s", "later_prefill_s", "residual_s")
_TARGET_SCENARIOS = (
    ("first prefill", _PHASES[:1]),
    ("all prefill", _PHASES[:2]),
    ("residual", _PHASES[2:]),
    ("first + residual", (_PHASES[0], _PHASES[2])),
    ("all phases", _PHASES),
)


def required_speedup(rows, target_multiplier, phases):
    """Find the uniform speedup of selected phases in a fixed-work scenario."""
    if not math.isfinite(target_multiplier) or target_multiplier <= 0:
        raise ValueError("the target multiplier must be finite and positive")
    if target_multiplier <= 1:
        return 1.0
    shares = [
        (
            sum(row[phase] for phase in _PHASES if phase in phases) / row["wall_s"],
            sum(row[phase] for phase in _PHASES if phase not in phases) / row["wall_s"],
        )
        for row in rows
    ]

    def remaining_time(fraction):
        return geomean([changed * fraction + fixed for changed, fixed in shares])

    required_fraction = 1 / target_multiplier
    if all(fixed > 0 for _, fixed in shares) and remaining_time(0) >= required_fraction:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if remaining_time(mid) > required_fraction:
            hi = mid
        else:
            lo = mid
    return 2 / (lo + hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--p1", type=float, help="assumed first-turn prefill seconds per conversation")
    parser.add_argument("--p2", type=float, default=0.15, help="assumed prefill seconds per later turn")
    parser.add_argument("--first-prefill-speedup", "--prefill-speedup", type=float, default=1.0)
    parser.add_argument("--later-prefill-speedup", "--later-speedup", type=float, default=1.0)
    parser.add_argument("--residual-speedup", type=float, default=1.0)
    parser.add_argument("--step-saving-ms", help="comma-separated savings per modeled step after residual speedup")
    parser.add_argument("--target-score", type=float, action="append", default=[], help="repeat to compare targets")
    args = parser.parse_args()
    try:
        data = json.loads(args.run.read_text())
        result = data.get("result", data)
        p1, rows = budget(result, p2=args.p2, p1=args.p1)
        speedups = (args.first_prefill_speedup, args.later_prefill_speedup, args.residual_speedup)
        savings = (
            [float(value) for value in args.step_saving_ms.split(",")]
            if args.step_saving_ms is not None
            else [0.0] * len(rows)
        )
        multipliers = predict(rows, *speedups[:2], savings, residual_speedup=speedups[2])
        frontier = result.get("frontier") or {}
        score = frontier.get("score")
        targets = []
        if args.target_score:
            if score is None or not math.isfinite(score) or score <= 0:
                raise ValueError("target scenarios require a published frontier score")
            for target in args.target_score:
                speedup = [
                    required_speedup(rows, target / score, phases)
                    for _, phases in _TARGET_SCENARIOS
                ]
                targets.append((target, speedup))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    source = "assumption" if args.p1 is not None else "one-user rate heuristic"
    print(f"Modeled service shares with P1 {p1:.3f}s ({source}) and P2 {args.p2:.3f}s (assumption).")
    print("Wall time is reconstructed from reported rates. Residual includes all unassigned time.")
    print("users  wall/conv  first-prefill  later-prefill  residual  residual/step")
    for row in rows:
        wall = row["wall_s"]
        print(
            f"{row['c']:5d}  {wall:8.1f}s  "
            f"{100 * row['first_prefill_s'] / wall:11.1f}%  "
            f"{100 * row['later_prefill_s'] / wall:11.1f}%  "
            f"{100 * row['residual_s'] / wall:7.1f}%  "
            f"{row['residual_step_ms']:10.1f} ms"
        )

    if speedups != (1.0, 1.0, 1.0) or any(savings):
        relative = geomean(multipliers)
        print("Modeled fixed-work multipliers", " / ".join(f"{value:.3f}x" for value in multipliers))
        print(f"Modeled geometric mean multiplier {relative:.3f}x.")
        if score is not None:
            print(f"Published frontier score {score:.4f}. Scenario with fixed normalization {score * relative:.4f}.")
    if targets:
        print(f"Published frontier score {score:.4f}. Required phase speedups if frontier normalization stays fixed.")
        for target, speedup in targets:
            print(f"Target score {target:g}")
            for (name, _), value in zip(_TARGET_SCENARIOS, speedup):
                outcome = f"{value:.2f}x" if value is not None else "unreachable under these assumptions"
                print(f"  {name:18} {outcome}")


if __name__ == "__main__":
    main()
