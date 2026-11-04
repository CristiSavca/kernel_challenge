#!/usr/bin/env python3
"""Offline, source-bound V28 decoder screen. No polling or automatic promotion."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = 'muse_decode_layer'
PARENT_SHA = '97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2'
PARENTS = (('7a8c4941741645ee', 'run'), ('b2a773e2d7194de8', 'check'))
PLAN = 'experiments/iterations/20261104-decoder-v28-root-plan.json'
COUNTS = {'setup': 1, 'test': 13, 'benchmark': 12, 'profile': 12}
LOADS = ('decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32')
AFFECTED = frozenset(('decode 32', 'verify 16', 'verify 32'))
GATE = {'affected_load_GM_min': 1.03, 'each_affected_latency_max': 1.03,
        'control_load_GM_latency_min': .97, 'control_load_GM_latency_max': 1.03,
        'each_control_latency_max': 1.03, 'whole_score_ratio_min': 1.015}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def positive(v, label):
    require(type(v) in (int, float) and math.isfinite(v) and v > 0,
            'Invalid positive number: ' + label)
    return v


def gm(values):
    values = list(values)
    require(bool(values), 'Empty geometric mean')
    return math.exp(sum(math.log(positive(x, 'ratio')) for x in values) / len(values))


def read_file(path, root=ROOT):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact outside repository')
    require(path.stat().st_size <= 16 * 1024 * 1024, 'Artifact too large')
    return path, path.read_bytes()


def benchmarks(result):
    require(result.get('passed') is True and result.get('status') == 'ok'
            and result.get('returncode') == 0 and result.get('error') is None,
            'Official numerical/compilation failure')
    require(result.get('gpu') == 'H200', 'Unexpected GPU')
    require(result.get('flags') == [] and result.get('checks') ==
            {'ok': True, 'flagged': [], 'rejected': []}, 'Official checks not clean')
    records = result.get('records', [])
    require(Counter(r.get('event') for r in records) == COUNTS, 'Record coverage mismatch')
    for r in records:
        require(r.get('status', 'pass' if r['event'] == 'profile' else None) == 'pass'
                and not r.get('flags', []), 'Failed or flagged record')
    rows = {r['index']: r for r in records if r['event'] == 'benchmark'}
    profiles = {r['index']: r for r in records if r['event'] == 'profile'}
    require(set(rows) == set(profiles) == set(range(12)), 'Missing/duplicate indices')
    for i, row in rows.items():
        c = row['case']
        load = LOADS[i // 2]
        require(c['load'] == load and c['layer'] == ('sliding' if i % 2 == 0 else 'full')
                and c['batch'] == int(load.split()[1])
                and c.get('q_len', 1) == (8 if load.startswith('verify') else 1)
                and c['weight'] == (46 if i % 2 == 0 else 16)
                and c['ctx'] == 80000 and c['page'] == 1 and c['dist'] == 'soft',
                'Unexpected case metadata')
        require(profiles[i]['case'] == c, 'Benchmark/profile case mismatch')
        for k in ('baseline_ns', 'median_ns', 'baseline_raw_ns', 'median_raw_ns', 'speedup'):
            positive(row.get(k), k)
        require(math.isclose(row['baseline_ns'] / row['median_ns'], row['speedup'], rel_tol=1e-9),
                'Case speedup does not reproduce')
    loads = {}
    for i, load in enumerate(LOADS):
        pair = [rows[2 * i], rows[2 * i + 1]]
        base = sum(r['baseline_ns'] * r['case']['weight'] for r in pair)
        candidate = sum(r['median_ns'] * r['case']['weight'] for r in pair)
        loads[load] = {'baseline_weighted_ns': base, 'candidate_weighted_ns': candidate,
                       'speedup': base / candidate, 'affected': load in AFFECTED}
    score = positive(result.get('speedup'), 'score')
    require(math.isclose(score, gm(r['speedup'] for r in loads.values()), rel_tol=1e-9),
            'Official weighted-load score does not reproduce')
    ci = result.get('speedup_ci')
    require(isinstance(ci, list) and len(ci) == 2
            and positive(ci[0], 'CI low') <= score <= positive(ci[1], 'CI high'),
            'Invalid score interval')
    return rows, loads, profiles


def read_run(job, digest, mode, root=ROOT):
    root = Path(root).resolve()
    require(re.fullmatch('[0-9a-f]{16}', job) and re.fullmatch('[0-9a-f]{64}', digest),
            'Invalid job/source identity')
    rp, raw = read_file(f'experiments/runs/{job}.json', root)
    rec = json.loads(raw)
    response = rec['response']
    ids = [response[k] for k in ('id', 'job_id') if k in response]
    require(rec['job'] == job and ids and all(x == job for x in ids), 'Receipt job mismatch')
    require(rec['problem'] == PROBLEM and rec['source_sha256'] == digest, 'Receipt source/problem mismatch')
    status = response.get('status')
    for key, expected in (('board', PROBLEM), ('mode', mode)):
        require((key not in response and status == 'received') or response.get(key) == expected,
                'Server problem/mode mismatch')
    sp, source = read_file(rec['source'], root)
    require(sha(source) == digest, 'Frozen source changed')
    provenance = {'job': job, 'mode': mode, 'source_sha256': digest,
                  'source': str(sp.relative_to(root)), 'receipt': str(rp.relative_to(root)),
                  'receipt_sha256': sha(raw)}
    if status in ('received', 'queued', 'pending', 'running'):
        return {'status': 'PENDING', 'provenance': provenance,
                'server_identity_confirmed': 'board' in response and 'mode' in response}
    require(status in ('done', 'failed', 'rejected', 'error', 'cancelled', 'infra_error',
                       'timeout', 'duplicate'), 'Unrecognized terminal status')
    result = response.get('result') or {}
    try:
        benchmarks(result)
    except ValueError as e:
        return {'status': 'OFFICIAL_GATE_FAILED', 'reason': str(e),
                'official_error': result.get('error'), 'provenance': provenance}
    require(status == 'done', 'Passing result has inconsistent status')
    pp, raw_profile = read_file(response['profile']['json'], root)
    profile = json.loads(raw_profile)
    require(profile['job_id'] == job and profile['problem'] == PROBLEM, 'Profile identity mismatch')
    require(sha(read_file(profile['submission'], root)[1]) == digest, 'Profile source mismatch')
    normalize = lambda rows: [{**r, 'flags': r.get('flags', [])} for r in rows]
    for field, event in (('benchmarks', 'benchmark'), ('profiles', 'profile')):
        require(normalize(profile[field]) == normalize([r for r in result['records'] if r['event'] == event]),
                'Profile/receipt records disagree')
    provenance.update(profile=str(pp.relative_to(root)), profile_sha256=sha(raw_profile),
                      evaluator_sha256=result['evaluator']['code_sha256'])
    return {'status': 'PASS', 'provenance': provenance, 'result': result}


def compare(current, parent):
    require(current['evaluator'] == parent['evaluator'], 'Evaluator/image changed')
    cr, cl, _ = benchmarks(current)
    pr, pl, _ = benchmarks(parent)
    rows = []
    for i, row in cr.items():
        strip_seed = lambda x: {k: v for k, v in x.items() if k != 'seed'}
        require(strip_seed(row['case']) == strip_seed(pr[i]['case']), 'Parent case metadata differs')
        ratio = row['speedup'] / pr[i]['speedup']
        rows.append({'index': i, 'case': row['case'], 'affected': row['case']['load'] in AFFECTED,
                     'speedup_ratio': ratio, 'normalized_latency_ratio': 1 / ratio,
                     'candidate_us': row['median_ns'] / 1000, 'parent_us': pr[i]['median_ns'] / 1000,
                     'candidate_stock_us': row['baseline_ns'] / 1000,
                     'parent_stock_us': pr[i]['baseline_ns'] / 1000,
                     'candidate_raw_us': row['median_raw_ns'] / 1000,
                     'parent_raw_us': pr[i]['median_raw_ns'] / 1000})
    loads = [{ 'load': name, 'affected': name in AFFECTED,
               'speedup_ratio': cl[name]['speedup'] / pl[name]['speedup'],
               'normalized_latency_ratio': pl[name]['speedup'] / cl[name]['speedup']}
             for name in LOADS]
    affected = gm(x['speedup_ratio'] for x in loads if x['affected'])
    controls = gm(x['normalized_latency_ratio'] for x in loads if not x['affected'])
    whole = current['speedup'] / parent['speedup']
    require(math.isclose(whole, gm(x['speedup_ratio'] for x in loads), rel_tol=1e-9), 'Relative score mismatch')
    gates = {'affected_weighted_load_GM_at_least_1p03': affected >= GATE['affected_load_GM_min'],
             'each_affected_latency_at_most_1p03': all(x['normalized_latency_ratio'] <= 1.03 for x in rows if x['affected']),
             'control_weighted_load_GM_latency_within_3pct': .97 <= controls <= 1.03,
             'each_control_latency_at_most_1p03': all(x['normalized_latency_ratio'] <= 1.03 for x in rows if not x['affected']),
             'whole_score_ratio_at_least_1p015': whole >= 1.015}
    return {'cases': rows, 'loads': loads, 'affected_weighted_load_GM_speedup_ratio': affected,
            'control_weighted_load_GM_latency_ratio': controls, 'whole_score_ratio': whole,
            'gates': gates, 'performance_gate_passed': all(gates.values()),
            'candidate_CI': current['speedup_ci'], 'parent_CI': parent['speedup_ci'],
            'limit': 'Different runs/seeds; normalized official ratios, not direct same-input A/B.'}


def activation(result, marker):
    require(isinstance(marker, str) and marker and len(marker) < 120, 'Invalid MAIN marker')
    rows, _, profiles = benchmarks(result)
    findings = []
    for i, row in rows.items():
        affected = row['case']['load'] in AFFECTED
        text = profiles[i]['report']
        present = marker in text
        small = 'okc::moe_fused<false>' in text
        findings.append({'index': i, 'affected': affected, 'MAIN_marker': present,
                         'small_incumbent_marker': small,
                         'pass': present and not small if affected else small and not present})
    return {'cases': findings, 'passed': all(x['pass'] for x in findings)}


def analyze(job, digest):
    raw = (ROOT / PLAN).read_bytes()
    plan = json.loads(raw)
    require(plan['candidate_sha256'] == digest and plan['fixed_gate'] == GATE, 'Frozen plan mismatch')
    require(plan['source_review_status'] == 'GO_FOR_OFFICIAL_CHECK', 'Source review incomplete')
    for path, expected in plan['evidence_sha256'].items():
        require(sha(read_file(path)[1]) == expected, 'Frozen evidence changed: ' + path)
    candidate = read_run(job, digest, 'check')
    out = {'schema': 1, 'plan': PLAN, 'plan_sha256': sha(raw), 'candidate': candidate['provenance'],
           'automatic_promotion': False, 'full_serving_validated': False}
    if candidate['status'] != 'PASS':
        return {**out, **{k: v for k, v in candidate.items() if k != 'provenance'}, 'fixed_screen_passed': False}
    comparisons = []
    for pid, mode in PARENTS:
        p = read_run(pid, PARENT_SHA, mode)
        require(p['status'] == 'PASS', 'Missing passing exact-parent anchor')
        comparisons.append({'parent': p['provenance'], **compare(candidate['result'], p['result'])})
    active = activation(candidate['result'], plan['main_profile_marker'])
    passed = active['passed'] and all(x['performance_gate_passed'] for x in comparisons)
    return {**out, 'status': 'USEFUL_CHECK_REVIEW_BEFORE_SCORING' if passed else 'CLOSE_FIXED_ATTEMPT',
            'fixed_screen_passed': passed, 'score': candidate['result']['speedup'],
            'score_ci': candidate['result']['speedup_ci'], 'comparisons': comparisons,
            'activation': active, 'evaluator': candidate['result']['evaluator'],
            'checks': candidate['result']['checks'], 'flags': candidate['result']['flags']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('source_sha256')
    args = parser.parse_args()
    print(json.dumps(analyze(args.job, args.source_sha256), indent=2))
