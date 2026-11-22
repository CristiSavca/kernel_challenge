"""Replay archived H200 client/server observations under explicit mixed-batch geometry assumptions."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import runpy
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[3]
CANDIDATES = ROOT / "serving_v0523/candidates"
ARCHIVE = ROOT / "experiments/modal_node/20261113T204343Z-bench"
GEOMETRY = ROOT / "experiments/iterations/20261118-serving-mixed-graph-capacity"
TP, CHANNELS, TAPS, WIDTH, VOCAB = 8, 8192, 5, 8, 202048
CAP = 16384
BF16_BYTES = 2
PATTERN = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+)")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def budget_geometry(prompts, prompt_rows, tails, captured_rows):
    body = prompt_rows + tails * WIDTH
    selected = prompts + tails * WIDTH
    # Live and padded body rows can differ in the outer eager graph tail.
    # The smaller live-shard assumption makes the row-count test conservative.
    padded_live = math.ceil(body / TP) * TP
    unit = CHANNELS * BF16_BYTES
    selected_gather = TP * selected * unit
    raw_feature_gather = padded_live * TAPS * unit
    narrow_feature_gather = padded_live * unit
    normalized_gate = body >= 128 * TP
    selected_gate = normalized_gate and selected < padded_live // TP
    padded_shard_gate = normalized_gate and selected < math.ceil(captured_rows / TP)
    vocabulary_bytes = selected * VOCAB * 4
    return dict(prompts=prompts, prompt_rows=prompt_rows, tails=tails, live_rows=body,
                captured_rows=captured_rows, logits_rows=selected,
                projected_gate=normalized_gate, selected_endpoint_gate=selected_gate,
                captured_span_selection_gate=padded_shard_gate,
                fp32_full_logits_bytes=vocabulary_bytes,
                tail_full_logit_bytes=tails * WIDTH * VOCAB * 4,
                raw_tap_gather_bytes=raw_feature_gather,
                normalized_feature_gather_bytes=narrow_feature_gather,
                raw_minus_normalized_bytes=raw_feature_gather-narrow_feature_gather,
                source_rows_per_rank=padded_live // TP, endpoint_gather_rows=TP*selected,
                endpoint_selected_bytes=selected_gather, endpoint_full_bytes=padded_live*unit)


def summarize(events):
    rows = [e['logits_rows'] for e in events]
    good = [e for e in events if e['projected_gate']]
    selected = [e for e in events if e['selected_endpoint_gate']]
    return dict(calls=len(events), prompt_requests=sum(e['prompts'] for e in events),
                tail_appearances=sum(e['tails'] for e in events), head_rows=sum(rows),
                rows_per_call=dict(min=min(rows),median=median(rows),max=max(rows)) if rows else None,
                vocabulary_gib_per_rank=sum(e['fp32_full_logits_bytes'] for e in events)/2**30,
                tail_vocabulary_gib_per_rank=sum(e['tail_full_logit_bytes'] for e in events)/2**30,
                normalized_context_calls=len(good), owner_head_eligible_calls=len(selected),
                owner_head_with_captured_span=sum(e['captured_span_selection_gate'] for e in events),
                context_rows=sum(e['live_rows'] for e in good),
                eligible_context_gather_gib=dict(raw=sum(e['raw_tap_gather_bytes'] for e in good)/2**30,
                    projected=sum(e['normalized_feature_gather_bytes'] for e in good)/2**30),
                selected_endpoint_gib_per_rank=sum(e['endpoint_selected_bytes'] for e in selected)/2**30,
                selected_endpoint_input_rows_per_rank=sum(e['logits_rows'] for e in selected),
                selected_endpoint_gathered_rows_per_rank=sum(e['endpoint_gather_rows'] for e in selected))


def main():
    replay = runpy.run_path(str(GEOMETRY.with_suffix('.py')))
    choose, selector_sha = replay['pinned_selector']()
    geom = json.loads(GEOMETRY.with_suffix('.json').read_bytes())
    one, verify = geom['target_prefill_graph_rows'], geom['target_verify_graph_requests']
    trace_path = next(ARCHIVE.glob('*v41__load.json'))
    server = next(ARCHIVE.glob('*v41__server.log'))
    trace = json.loads(trace_path.read_bytes())
    order = [(r['concurrency'], rep) for r in trace['rungs'] for rep in range(r['reps'])]
    per_rep = defaultdict(list)
    slot = -1
    previous = None
    prompt_calls = Counter()
    for line in server.read_text().splitlines():
        if 'Cache flushed successfully!' in line:
            slot += 1
            previous = None
        m = PATTERN.search(line)
        if not m or slot < 0:
            continue
        p, q, t, queued, pending = map(int, m.groups())
        load, rep = order[slot]
        prompt_calls[load] += 1
        assert p + t <= load and q <= CAP
        if t > 0:
            body = q + t * WIDTH
            if body <= CAP and choose(body, one) <= choose(q, one) + WIDTH * choose(t, verify):
                per_rep[load, rep].append(('original',budget_geometry(p, q, t, choose(body, one))))
            elif (q == CAP and p == 1 and queued == 0 and pending > 0 and previous is not None
                  and previous[0] == 1 and previous[3] == 0 and previous[4] == q+pending):
                last = (q+pending-1)%CAP+1
                new_prompt = q-t*WIDTH
                if (last+t*WIDTH <= CAP and choose(last+t*WIDTH, one)==choose(last, one)
                        and choose(new_prompt,one) + WIDTH*choose(t,verify) >= CAP):
                    per_rep[load, rep].append(('continuation',budget_geometry(p,new_prompt,t,CAP)))
        previous = p,q,t,queued,pending
    assert slot + 1 == len(order)
    h = runpy.run_path(str(next(CANDIDATES.glob('v195*/score_headroom.py'))))
    rounds = h['measured_intervals'](trace)
    rerate = h['rerate']
    baseline = rerate(rounds,(1,1,1))
    whole_phases = {phase:rerate(rounds,factors) for phase,factors in {
        'first_turn_ttft_free':(math.inf,1,1),'later_turn_ttft_free':(1,math.inf,1),
        'after_first_token_free':(1,1,math.inf)}.items()}
    continuation = json.loads(next(CANDIDATES.glob('v194*/trace.json')).read_bytes())['by_load']
    result = {}
    for rung in trace['rungs']:
        load = rung['concurrency']
        pairs = [per_rep[key] for key in order if key[0]==load]
        events = [e for pair in pairs for _, e in pair]
        raw = [e for pair in pairs for k,e in pair if k=='original']
        shifted = [e for pair in pairs for k,e in pair if k=='continuation']
        base = geom['rows'][str(load)]['policies'].get('same_budget_whole_cohort',{})
        assert len(raw)==base.get('paired_passes',0)
        assert sum(e['tails'] for e in raw)==base.get('tail_pairs',0)
        assert len(shifted)==continuation[str(load)].get('fit_same_current_and_final_buckets',0)
        assert sum(e['tails'] for e in shifted)==continuation[str(load)].get('running_appearances_fitting',0)
        users=[u for (n,_),group in rounds.items() if n==load for u in group]
        totals=[sum(u[i] for u in users) for i in range(4)]
        shares=[s/sum(totals) for s in totals]
        wr=baseline[load][2]
        assert all(abs(a-b)<0.006 for a,b in zip(wr,rung['wall_s']))
        assert abs(baseline[load][0]-rung['tok_s_per_gpu']) <0.06
        throughput=[]
        for saved_seconds in (0.01,0.05,0.1):
            newwalls=[w-len(events_in_rep)*saved_seconds for w,events_in_rep in zip(wr,pairs)]
            counts=[sum(u[5] for u in rounds[load,rep]) for rep in range(len(wr))]
            rate=mean(count/(TP*w) for count,w in zip(counts,newwalls))
            throughput.append(dict(net_milliseconds_per_fused_call=saved_seconds*1000,
                                   raw_throughput_factor=rate/baseline[load][0]))
        cs=Counter()
        for row in trace['requests']:
            if row['load']==load:cs['requests']+=1;cs['generated']+=row['completion']
        appearances=sum(e['tails'] for e in events)
        result[load]=dict(prompt_calls=prompt_calls[load],paper_schedule=summarize(events),
                          ordinary_budget=summarize(raw),extra_identifiable_continuations=summarize(shifted),
                          client_requests=cs['requests'], generated=cs['generated'],
                          matching_generated_fraction_at_four_commits=4*appearances/cs['generated'],
                          conditional_ceiling_fraction_at_eight_commits=min(1,8*appearances/cs['generated']),
                          measured_client_seconds=dict(zip(('cold_ttft','later_ttft','stream','gaps'),totals)),
                          measured_client_time_shares=dict(zip(('cold_ttft','later_ttft','stream','gaps'),shares)),
                          observed_repetition_wall_seconds=wr,
                          all_phase_removal_raw_rung_factor={k:v[load][0]/baseline[load][0] for k,v in whole_phases.items()},
                          hypothetical_makespan_credit=throughput)
    prompt_rows=3992
    toy_bucket=choose(prompt_rows+3*WIDTH,one)
    assert toy_bucket==choose(prompt_rows,one)==4096
    toy=budget_geometry(2,prompt_rows,3,toy_bucket)
    toy['selected_positions']=[999,prompt_rows-1,*range(prompt_rows,prompt_rows+3*WIDTH)]
    assert len(toy['selected_positions'])==toy['logits_rows'] and toy['selected_endpoint_gate']
    return dict(trace_files={str(p.relative_to(ROOT)):digest(p) for p in
                    (trace_path,server,GEOMETRY.with_suffix('.json'),next(CANDIDATES.glob('v194*/trace.json')))},
                selector_sha256=selector_sha,raw_v41_requests=len(trace['requests']),repetitions=len(order),
                observed_server='Eight H200 TP8 V41, mixed prefill and LayerNorm SP disabled during observation',
                width=WIDTH,tp=TP,vocab=VOCAB,channels=CHANNELS,source_taps=TAPS,toy=toy,loads=result)

if __name__=="__main__":
    print(json.dumps(main(),indent=2,allow_nan=False))
