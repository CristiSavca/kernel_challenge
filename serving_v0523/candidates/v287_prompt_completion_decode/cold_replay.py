"""Price native decode intervals on complete cold waves, preserving their real prompt work."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from itertools import permutations, product
import runpy
from statistics import median

here = Path(__file__).resolve().parent
root = here.parents[2]
prior = runpy.run_path(str(root / "experiments/iterations/20261119-serving-cold-wave.py"))
archives = runpy.run_path(str(next(here.parent.glob("v244_*/check.py"))))["ARCHIVES"]


def max_assignment(budgets, lengths):
    """Maximum successes under unknown request-to-admission ordering."""
    ordered = sorted(lengths)
    j = 0
    for available in sorted(budgets):
        if j < len(ordered) and ordered[j] <= available:
            j += 1
    return j


def range_of_finishes(budgets, lengths):
    # Integer intervals avoid a strict-inequality tie in the miss matching.
    max_misses = max_assignment([size - 1 for size in lengths], budgets)
    return [len(lengths) - max_misses, max_assignment(budgets, lengths)]


def replay(events, clients, policy, round_cost):
    users = len(clients)
    pulses = []
    previous = 0
    for i, e in enumerate(events):
        n = e["running"]
        assert n >= previous
        fire = n > 0 and (policy == "all_chunks" or n > previous)
        if fire:
            # First report with this ready set supplies the reference time.
            # This can be one useful prompt interval later than eligibility.
            pulses.append((n, e["elapsed"]))
        previous = n
    end = events[-1]["elapsed"]
    final = end + round_cost * len(pulses)
    slots = [sum(n > ord_ for n, _ in pulses) for ord_ in range(users)]
    first = [next((t + (i+1) * round_cost for i, (n,t) in enumerate(pulses) if n > ord_), final)
             for ord_ in range(users)]
    lead = [end-t for t in first]
    output = dict(rounds=len(pulses), request_rounds=sum(slots),
                  extra_serial_time_before_last_prompt_s=round(final-end, 4),
                  median_first_decode_lead_vs_original_last_prompt_s=round(median(lead), 4),
                  p90_first_decode_lead_vs_original_last_prompt_s=round(end - prior["quantile"](first, .9), 4),
                  post_prefill_rounds_by_admission_order=slots)
    out_lengths=[c["completion"] for c in clients]
    assert all(c["completion"] == c["asked"] for c in clients)
    output["possible_finished_clients_by_last_prompt"] = {
        f"constant_{d}_commits_including_bonus": range_of_finishes([1+d*s for s in slots],out_lengths)
        for d in (1, 4, 8)}
    return output


def main():
    windows = []
    inputs = {}
    for label, run, prefix in archives:
        p = root / "experiments/modal_node" / run
        log, data = [p/(prefix + name) for name in ("server.log", "load.json")]
        for path in (log, data):
            inputs[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
        client = json.loads(data.read_bytes())
        assert client["gpus"] == 8
        reps = [(r["concurrency"],idx) for r in client["rungs"] for idx in range(r["reps"])]
        sections = log.read_text().split("Cache flushed successfully!")[1:]
        for section, (users, rep) in zip(sections, reps, strict=False):
            cases = [r for r in client["requests"] if (r["load"],r["rep"],r["turn"]) == (users, rep, 0)]
            if len(cases) != users:
                continue
            events = prior["cold_wave"](section,cases)
            assert all(a["elapsed"] <= b["elapsed"] for a,b in zip(events,events[1:]))
            visible = [c["t0"]+c["ttft"] for c in cases]
            policies = {}
            for name in ("on_completion","all_chunks"):
                costs = {}
                for seconds in (.02,.05,.1,.2):
                    row=replay(events,cases,name,seconds)
                    costs[str(seconds)]={k:v for k,v in row.items() if k.endswith("_s")}
                policies[name]={k:v for k,v in row.items() if not k.endswith("_s")}
                policies[name]["serial_round_cost_sensitivity_s"]=costs
            a,b=policies["on_completion"],policies["all_chunks"]
            assert a["rounds"]<=b["rounds"]
            previous=0
            completion_points=[]
            all_points=[]
            for i,e in enumerate(events):
                n=e["running"]
                if n:
                    all_points.append((i,n))
                    if n>previous:
                        completion_points.append((i,n))
                previous=n
            for ordinal in range(users):
                assert next((i for i,n in completion_points if n>ordinal),None)==next(
                    (i for i,n in all_points if n>ordinal),None)
            windows.append(dict(archive=label, users=users, rep=rep, prompt_calls=len(events),
                                first_prompt_rows=sum(e["new"] for e in events), reported_cold_span_s=round(events[-1]["elapsed"],4),
                                measured_first_text_spread_ms=round(1000*(max(visible)-min(visible)),4),
                                first_requested_output_lengths=sorted(c["completion"] for c in cases), policies=policies))
    matching_checks=0
    for budgets in product(range(3),repeat=3):
        for lengths in product(range(1,4),repeat=3):
            actual=[sum(b>=l for b,l in zip(budgets,order)) for order in permutations(lengths)]
            assert range_of_finishes(budgets,lengths)==[min(actual),max(actual)]
            matching_checks+=1
    print(json.dumps(dict(h200_runs=inputs, matching_oracle_cases=matching_checks, windows=windows,
                  interpretation="Intervals and request lengths are physical V41/V44 records. Fixed prompt chronology and declared per-round costs are counterfactual. At most eight outputs including bonus per native width-eight round. Request-to-admission identities and per-round acceptance are unknown. An assignment range couples all output lengths to distinct ready positions and is conditional on constant commitments; its upper endpoint at eight is a genuine optimistic cap under the frozen schedule. A request already finished in a replay would drop out and may issue another turn; no such later turn is scheduled here. First decode opportunity does not guarantee first visible text or an earlier completion."),indent=2))


if __name__ == "__main__":
    main()
