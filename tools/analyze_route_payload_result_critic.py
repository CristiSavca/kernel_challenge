"""Offline fixed v20 audit. No network, submissions, GPU calls or promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from analyze_sk2_r16_result import ROOT, bounded_file, digest, gm, pass_gate, positive, require

CANDIDATE_JOB = '99924cdcdb1f4952'
CANDIDATE_SHA = '8ab1cf33ef5b4a4eea8721625ad446358fcbd603dfe1277eee17b7e76a53d232'
PARENT_SHA = '97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2'
PARENTS = [('7a8c4941741645ee', 'run'), ('b2a773e2d7194de8', 'check')]
LOADS = ['decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32']
TOKENS = dict(zip(LOADS, [1, 16, 32, 8, 128, 256]))


def clean_rows(rows):
    result = []
    for row in rows:
        require(row.get('flags', []) == [], 'nonempty or invalid exported flags')
        result.append({k: v for k, v in row.items() if k != 'flags'})
    return result


def read_run(job, sha, mode):
    path, raw = bounded_file(f'experiments/runs/{job}.json')
    receipt = json.loads(raw); response = receipt['response']; status = response['status']
    require(receipt['job'] == job and receipt['problem'] == 'muse_decode_layer'
            and receipt['source_sha256'] == sha, 'local receipt identity mismatch')
    ids = [response[k] for k in ('id', 'job_id') if k in response]
    require(bool(ids) and all(v == job for v in ids), 'server job identity mismatch')
    for key, expected in [('board', 'muse_decode_layer'), ('mode', mode)]:
        require(response.get(key) == expected or (key not in response and status == 'received'), 'server identity mismatch')
    source_path, source = bounded_file(receipt['source'])
    require(digest(source) == sha, 'frozen source mismatch')
    provenance = {'job': job, 'mode': mode, 'source_sha256': sha, 'receipt_sha256': digest(raw),
                  'receipt': str(path.relative_to(ROOT)), 'source': str(source_path.relative_to(ROOT))}
    if status in ('received', 'queued', 'running', 'pending'):
        return {'status': 'PENDING', 'saved_status': status, 'provenance': provenance}
    require(status in ('done', 'failed', 'rejected', 'error', 'cancelled', 'infra_error', 'timeout', 'duplicate'),
            'unknown terminal state')
    result = response.get('result') or {}
    try:
        pass_gate(result)
    except ValueError as exc:
        return {'status': 'OFFICIAL_GATE_FAILED', 'reason': str(exc), 'provenance': provenance}
    require(status == 'done', 'inconsistent passing terminal state')
    profile_path, profile_raw = bounded_file(response['profile']['json'])
    profile = json.loads(profile_raw)
    require(profile['job_id'] == job and profile['problem'] == 'muse_decode_layer', 'profile identity mismatch')
    _, profile_source = bounded_file(profile['submission'])
    require(digest(profile_source) == sha, 'profile source mismatch')
    for field, event in [('benchmarks', 'benchmark'), ('profiles', 'profile')]:
        require(clean_rows(profile[field]) == clean_rows([r for r in result['records'] if r['event'] == event]),
                'profile/receipt disagreement beyond absent-empty flags')
    provenance.update(profile=str(profile_path.relative_to(ROOT)), profile_sha256=digest(profile_raw),
        evaluator_inventory_sha256=digest(json.dumps(result['evaluator'], sort_keys=True, separators=(',', ':')).encode()))
    return {'status': 'PASS', 'provenance': provenance, 'result': result}


def scores(result):
    pass_gate(result)
    benchmarks = [r for r in result['records'] if r['event'] == 'benchmark']
    profiles = [r for r in result['records'] if r['event'] == 'profile']
    require({r['index'] for r in benchmarks} == set(range(12)), 'benchmark index coverage')
    require({r['index'] for r in profiles} == set(range(12)), 'profile index coverage')
    rows = {r['index']: r for r in benchmarks}; profiles = {r['index']: r for r in profiles}
    loads = {}
    for i, row in rows.items():
        case = row['case']; name = case['load']; layer = case['layer']
        require(name in LOADS and layer in ('sliding', 'full'), 'unknown case')
        require(case['batch'] * case.get('q_len', 1) == TOKENS[name], 'token shape mismatch')
        require(profiles[i]['case'] == case, 'profile/benchmark case mismatch')
        weight = 46 if layer == 'sliding' else 16
        require(case['weight'] == weight, 'weight mismatch')
        for key in ('median_ns', 'median_raw_ns', 'baseline_ns', 'speedup'):
            positive(row[key], key)
        require(math.isclose(row['speedup'], row['baseline_ns'] / row['median_ns'], rel_tol=1e-9), 'case score mismatch')
        item = loads.setdefault(name, {})
        require(layer not in item, 'duplicate load/layer')
        item[layer] = row
    require(set(loads) == set(LOADS) and all(set(x) == {'sliding', 'full'} for x in loads.values()), 'load coverage')
    weighted = {}
    for name, pair in loads.items():
        numerator = sum(w * pair[layer]['baseline_ns'] for layer, w in [('sliding', 46), ('full', 16)])
        denominator = sum(w * pair[layer]['median_ns'] for layer, w in [('sliding', 46), ('full', 16)])
        weighted[name] = numerator / denominator
    require(math.isclose(gm(weighted.values()), result['speedup'], rel_tol=1e-9), 'six-load score mismatch')
    return rows, weighted


def compare(candidate, parent):
    require(candidate['evaluator'] == parent['evaluator'], 'full evaluator inventory/image mismatch')
    require(candidate['gpu'] == parent['gpu'] == 'H200', 'GPU model mismatch')
    cb, cl = scores(candidate); pb, pl = scores(parent)
    per_case = []
    for i in range(12):
        clean_case = lambda c: {k: v for k, v in c.items() if k != 'seed'}
        require(clean_case(cb[i]['case']) == clean_case(pb[i]['case']), 'case identity mismatch beyond seed')
        per_case.append({'index': i, 'load': cb[i]['case']['load'], 'layer': cb[i]['case']['layer'],
            'normalized_latency_ratio': pb[i]['speedup'] / cb[i]['speedup'],
            'harness_median_ratio': cb[i]['median_ns'] / pb[i]['median_ns'],
            'raw_median_ratio': cb[i]['median_raw_ns'] / pb[i]['median_raw_ns']})
    ratios = {name: cl[name] / pl[name] for name in LOADS}
    aggregate = candidate['speedup'] / parent['speedup']
    require(math.isclose(aggregate, gm(ratios.values()), rel_tol=1e-9), 'aggregate ratio mismatch')
    ci, pi = candidate['speedup_ci'], parent['speedup_ci']
    return {'aggregate_speedup_ratio': aggregate, 'load_speedup_ratios': ratios,
            'load_normalized_latency_ratios': {k: 1 / v for k, v in ratios.items()}, 'cases': per_case,
            'aggregate_at_least_0_5pct': aggregate >= 1.005,
            'all_weighted_load_latencies_at_most_1_02': all(1 / v <= 1.02 for v in ratios.values()),
            'candidate_CI': ci, 'parent_CI': pi, 'reported_CIs_overlap': max(ci[0], pi[0]) <= min(ci[1], pi[1]),
            'same_reported_node': candidate.get('machine', {}).get('node') == parent.get('machine', {}).get('node')}


def activation(result):
    output = []
    for row in result['records']:
        if row['event'] != 'profile':
            continue
        names = re.findall(r'^(_route(?:_split|_a|_b)?)\s+', row.get('report', ''), re.MULTILINE)
        output.append({'index': row['index'], 'load': row['case']['load'], 'layer': row['case']['layer'],
            'visible_route_symbols': names, 'changed_route_symbol_visible': names in [['_route'], ['_route_split']]})
    return output


def analyze():
    candidate = read_run(CANDIDATE_JOB, CANDIDATE_SHA, 'check')
    report = {'candidate': candidate['provenance'], 'automatic_promotion_allowed': False,
              'comparisons': [], 'method': 'Official six weighted-load score; all12 intended affected, no unchanged controls.'}
    if candidate['status'] != 'PASS':
        return {**report, **{k: v for k, v in candidate.items() if k != 'provenance'}}
    comparisons = []
    for job, mode in PARENTS:
        parent = read_run(job, PARENT_SHA, mode)
        require(parent['status'] == 'PASS', 'incomplete parent')
        comparisons.append({'parent': parent['provenance'], **compare(candidate['result'], parent['result'])})
    joint = [name for name in LOADS if all(c['load_speedup_ratios'][name] > 1 for c in comparisons)]
    passed = len(joint) >= 4 and all(c['aggregate_at_least_0_5pct'] and c['all_weighted_load_latencies_at_most_1_02'] for c in comparisons)
    return {**report, 'status': 'NUMERICAL_SCREEN_PASS' if passed else 'NUMERICAL_SCREEN_REJECT',
        'comparisons': comparisons, 'joint_positive_loads': joint, 'joint_positive_count': len(joint),
        'fixed_gate_passed': passed, 'profile_activation': activation(candidate['result']),
        'limits': 'Different runs/seeds; CI overlap is explicit. Visible symbols plus source provenance are not pointer traces or causal router-time attribution.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = analyze()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {'status': 'EVIDENCE_REJECTED', 'reason': str(exc), 'comparisons': [], 'automatic_promotion_allowed': False}
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
