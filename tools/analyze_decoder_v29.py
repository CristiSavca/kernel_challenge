#!/usr/bin/env python3
"""Fixed V29 operational screen, reusing the strict official-receipt reader."""
import argparse
from collections import Counter
import json
import math
import re
from analyze_decoder_v28 import ROOT, PARENT_SHA, PARENTS, LOADS, benchmarks, gm, read_run, read_file, require, sha

PLAN = 'experiments/iterations/20261104-decoder-exact-patch-v29-root-plan.json'
AFFECTED = frozenset(('decode 1', 'decode 16', 'decode 32', 'verify 1'))
GATE = {'affected_load_GM_min': 1.04, 'each_affected_latency_max': 1.03,
        'control_load_GM_latency_min': .97, 'control_load_GM_latency_max': 1.03,
        'each_control_latency_max': 1.03, 'whole_score_ratio_min': 1.025}

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
    gates = {'affected_weighted_load_GM_at_least_1p04': affected >= GATE['affected_load_GM_min'],
             'each_affected_latency_at_most_1p03': all(x['normalized_latency_ratio'] <= 1.03 for x in rows if x['affected']),
             'control_weighted_load_GM_latency_within_3pct': .97 <= controls <= 1.03,
             'each_control_latency_at_most_1p03': all(x['normalized_latency_ratio'] <= 1.03 for x in rows if not x['affected']),
             'whole_score_ratio_at_least_1p025': whole >= 1.025}
    return {'cases': rows, 'loads': loads, 'affected_weighted_load_GM_speedup_ratio': affected,
            'control_weighted_load_GM_latency_ratio': controls, 'whole_score_ratio': whole,
            'gates': gates, 'performance_gate_passed': all(gates.values()),
            'candidate_CI': current['speedup_ci'], 'parent_CI': parent['speedup_ci'],
            'limit': 'Different runs/seeds; normalized official ratios, not direct same-input A/B.'}

def compressed_instances(report, marker):
    # COMP is argument 8; the first 7 are integers. This prefix fits the
    # official shortened names without depending on omitted trailing types.
    pattern = re.escape(marker) + r"<((?:[0-9]+, ){7})(true|false)"
    return Counter(tuple(int(x) for x in m.group(1).rstrip(', ').split(', '))
                   for m in re.finditer(pattern, report) if m.group(2) == 'true')


def activation(result, parent):
    rows, _, profiles = benchmarks(result)
    _, _, anchors = benchmarks(parent)
    findings = []
    for i, row in rows.items():
        affected = row['case']['load'] in AFFECTED
        old = compressed_instances(anchors[i]['report'], 'sk2_kernel')
        new = compressed_instances(profiles[i]['report'], 'sk2_exact_patch_v29')
        residual = compressed_instances(profiles[i]['report'], 'sk2_kernel')
        passed = bool(old) and new == old and not residual if affected else not old and not new and not residual
        findings.append({'index': i, 'affected': affected, 'expected_instances': [list(x) for x in old.elements()],
                         'observed_instances': [list(x) for x in new.elements()],
                         'residual_parent_instances': [list(x) for x in residual.elements()], 'pass': passed})
    return {'cases': findings, 'passed': all(x['pass'] for x in findings)}


def analyze(job, digest):
    raw = (ROOT / PLAN).read_bytes(); plan = json.loads(raw)
    require(plan['candidate_sha256'] == digest and plan['fixed_gate'] == GATE, 'Frozen plan mismatch')
    require(plan['source_review_status'] == 'GO_FOR_OFFICIAL_CHECK', 'Source review incomplete')
    for path, expected in plan['evidence_sha256'].items():
        require(sha(read_file(path)[1]) == expected, 'Frozen evidence changed: ' + path)
    candidate = read_run(job, digest, 'check')
    out = {'schema': 1, 'plan': PLAN, 'plan_sha256': sha(raw), 'candidate': candidate['provenance'],
           'automatic_promotion': False, 'full_serving_validated': False}
    if candidate['status'] != 'PASS':
        return {**out, **{k:v for k,v in candidate.items() if k != 'provenance'}, 'fixed_screen_passed': False}
    comparisons = []; activations = []
    for pid, mode in PARENTS:
        p = read_run(pid, PARENT_SHA, mode)
        require(p['status'] == 'PASS', 'Missing passing exact-parent anchor')
        comparisons.append({'parent': p['provenance'], **compare(candidate['result'], p['result'])})
        activations.append({'parent_job': pid, **activation(candidate['result'], p['result'])})
    passed = all(x['performance_gate_passed'] for x in comparisons) and all(x['passed'] for x in activations)
    return {**out, 'status': 'USEFUL_CHECK_REVIEW_BEFORE_SCORING' if passed else 'CLOSE_FIXED_ATTEMPT',
            'fixed_screen_passed': passed, 'score': candidate['result']['speedup'],
            'score_ci': candidate['result']['speedup_ci'], 'comparisons': comparisons, 'activation': activations,
            'checks': candidate['result']['checks'], 'flags': candidate['result']['flags'],
            'evaluator': candidate['result']['evaluator']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job'); parser.add_argument('source_sha256'); args = parser.parse_args()
    print(json.dumps(analyze(args.job, args.source_sha256), indent=2))
