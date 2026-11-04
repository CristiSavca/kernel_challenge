#!/usr/bin/env python3
"""Offline V30 fixed screen. No polling, allocation, or automatic promotion."""
import argparse
from collections import Counter
import json
import re

from analyze_wgs_shared_a_result import (
    ROOT, PARENT_SHA, PARENTS, benchmark_rows, bounded_file, compare as parent_compare,
    digest, gm, read_run, require,
)

PLAN = 'experiments/iterations/20261104-moe-arch-v30-root-plan.json'
GATE = {'affected_load_GM_min': 1.04, 'each_affected_latency_max': 1.03,
        'control_load_GM_latency_min': .97, 'control_load_GM_latency_max': 1.03,
        'each_control_latency_max': 1.03, 'whole_score_ratio_min': 1.02}
EVALUATOR_SHA = '68c76b0c18f7154905a802d227b084e15c4b95e17d0d67fa40e9fef3d6ab7e11'
PARENT_PINS = {
    'd24dd50196cb4158': ('dc6ecf5b212644f29ae8b9da1dc8307245ea7f304311d6959d0a910545c226f1',
                       '563d259d696c429ea45d51195184f53e0dd9f61b81fe0123877394d2cd71a682'),
    'bcb761dfca264db0': ('9c8f1bc53cbb66947ac22ad61199e90b8e4d1ec9fd14dc88155408e92dfe95e2',
                       'df99864c5d69498f6369a3f2a2a4156ea050ff4e5bc12d1837bf7f107c6c824f'),
}
RESOURCE_GEOMETRIES = {
    'route': (512, 22272, 1), 'coldA': (320, 89856, 2), 'hotA': (576, 231168, 1),
    'coldB': (320, 89856, 2), 'hotB': (576, 231168, 1), 'final': (512, 23296, 1),
}
RESOURCE_RE = re.compile(
    r'V30R name=(\w+) nt=(\d+) dyn=(\d+) regs=(-?\d+) local=(-?\d+) static=(-?\d+) '
    r'maxdyn=(-?\d+) occ=(-?\d+) min=(\d+) api=(-?\d+),(-?\d+),(-?\d+)')
TIME_RE = re.compile(r'\s([0-9]+(?:\.[0-9]+)?)\s*(ns|µs|us|ms|s)\s+'
                     r'([0-9]+(?:\.[0-9]+)?)%\s+(\d+)\s*$')


def validate_result(result):
    rows = benchmark_rows(result)
    tests = {r['index']: r for r in result['records'] if r['event'] == 'test'}
    require(set(tests) == set(range(5)), 'Missing/duplicate numerical test indices')
    for i, bs in enumerate((4, 33, 257, 128, 1)):
        expected = {'bs': bs, 'sharding': 'ep8', **({'dist': 'skewed'} if i == 3 else {})}
        require({k: v for k, v in tests[i]['case'].items() if k != 'seed'} == expected,
                'Numerical test case differs')
    require(result.get('gpu') == 'H200', 'GPU differs from fixed H200 domain')
    encoded = json.dumps(result['evaluator'], sort_keys=True, separators=(',', ':')).encode()
    require(digest(encoded) == EVALUATOR_SHA, 'Evaluator inventory/image differs from exact parents')
    profiles = {r['index']: r for r in result['records'] if r['event'] == 'profile'}
    require(set(profiles) == set(rows), 'Profile index coverage mismatch')
    for i, row in rows.items():
        require(profiles[i]['case'] == row['case'] and isinstance(profiles[i]['report'], str),
                'Profile case/report mismatch')
    return rows, profiles


def read_parent(job, mode, root=ROOT):
    p = read_run(job, PARENT_SHA, mode, root)
    require(p['status'] == 'PASS', 'Exact parent is not complete/pass')
    require((p['provenance']['receipt_sha256'], p['provenance']['profile_sha256']) == PARENT_PINS[job],
            'Pinned parent receipt/profile changed')
    validate_result(p['result'])
    return p


def compare(current, parent):
    validate_result(current); validate_result(parent)
    result = parent_compare(current, parent)
    # The helper's historical SS-A thresholds never govern this iteration.
    result.pop('gates'); result.pop('performance_gate_passed')
    rows = result['cases']
    gates = {
        'affected_GM_at_least_1p04': result['affected_speedup_ratio_GM'] >= GATE['affected_load_GM_min'],
        'whole_score_ratio_at_least_1p02': result['aggregate_speedup_ratio'] >= GATE['whole_score_ratio_min'],
        'every_affected_latency_at_most_1p03': all(r['normalized_latency_ratio'] <= 1.03 for r in rows if r['affected']),
        'control_GM_latency_within_3pct': .97 <= result['control_normalized_latency_ratio_GM'] <= 1.03,
        'every_control_latency_at_most_1p03': all(r['normalized_latency_ratio'] <= 1.03 for r in rows if not r['affected']),
    }
    return {**result, 'gates': gates, 'performance_gate_passed': all(gates.values())}


def resource_records(log):
    require(isinstance(log, str) and len(log.encode()) <= 16 * 1024 * 1024, 'Missing/oversized log')
    rows = {}; problems = []
    for line in log.splitlines():
        if not line.startswith('V30R'):
            continue
        m = RESOURCE_RE.fullmatch(line)
        if not m:
            problems.append('Malformed/truncated resource row'); continue
        name, *values = m.groups()
        if name not in RESOURCE_GEOMETRIES:
            problems.append('Unexpected resource identity: ' + name); continue
        keys = ('nt', 'dyn', 'regs', 'local', 'static', 'maxdyn', 'occ', 'min', 'set_api', 'attr_api', 'capacity_api')
        row = dict(zip(keys, map(int, values)))
        if name in rows and rows[name] != row:
            problems.append('Conflicting resource identity: ' + name)
        rows[name] = row
    for name, row in rows.items():
        nt, dyn, need = RESOURCE_GEOMETRIES[name]
        valid = ((row['nt'], row['dyn'], row['min']) == (nt, dyn, need)
                 and row['set_api'] == row['attr_api'] == row['capacity_api'] == 0
                 and 0 < row['regs'] <= 255 and row['local'] >= 0 and row['static'] >= 0
                 and row['maxdyn'] >= dyn and row['occ'] >= need)
        if not valid:
            problems.append('Invalid attributes/admission: ' + name)
    missing = sorted(set(RESOURCE_GEOMETRIES) - set(rows))
    return {'rows': rows, 'missing': missing, 'problems': problems,
            'passed': not missing and not problems,
            'scope': 'Compiled-pointer theoretical block capacity queried on first module launch; not achieved occupancy or per-case execution. Local allocation and warnings alone are informational.'}


def read_resources(candidate, root=ROOT):
    _, raw = bounded_file(candidate['provenance']['receipt'], root)
    receipt = json.loads(raw); response = receipt['response']
    log = candidate['result'].get('log_tail')
    require(isinstance(log, str), 'Receipt lacks retained log')
    info = {'receipt_tail_sha256': digest(log.encode()), 'saved_log_relation': 'not_available'}
    saved = response.get('output', {}).get('log')
    if saved:
        _, data = bounded_file(saved, root)
        text = data.decode()
        info['saved_log_sha256'] = digest(data)
        if text == log:
            info['saved_log_relation'] = 'identical'
        elif log.startswith(text):
            info['saved_log_relation'] = 'saved_prefix_of_receipt_tail'
        elif text.endswith(log):
            info['saved_log_relation'] = 'receipt_tail_of_saved_log'
            log = text
        else:
            return {**info, 'rows': {}, 'missing': sorted(RESOURCE_GEOMETRIES),
                    'problems': ['Saved log and receipt tail disagree'], 'passed': False}
        # Coherent retained text can establish an explicit row; successful launch
        # alone cannot fill missing resource identities.
    return {**info, 'analyzed_log_sha256': digest(log.encode()), **resource_records(log)}


def kernel_markers(report):
    found = Counter(); malformed = []
    for line in report.splitlines():
        kind = None
        if 'okc::v30_route_gather(' in line: kind = 'route'
        elif 'okc::v30_final(' in line: kind = 'final'
        elif 'okc::v30_tiles<' in line:
            m = re.search(r'okc::v30_tiles<(false|true),\s*(false|true)>', line)
            if m: kind = ('cold' if m[2] == 'true' else 'hot') + ('B' if m[1] == 'true' else 'A')
            else: malformed.append('Incomplete V30 tile identity')
        elif 'okc::moe_fused<' in line:
            m = re.search(r'okc::moe_fused<(false|true)>', line)
            kind = 'parent_' + m[1] if m else None
            if not m: malformed.append('Incomplete parent identity')
        if kind:
            duration = TIME_RE.search(line)
            if not duration or float(duration[1]) <= 0 or int(duration[4]) <= 0:
                malformed.append('Missing positive CUDA duration/calls: ' + kind)
            found[kind] += 1
        elif 'okc::v30_' in line:
            malformed.append('Unknown V30 kernel')
    return found, malformed


def activation(current, parent):
    rows, profiles = validate_result(current); _, old_profiles = validate_result(parent)
    out = []
    expected_main = Counter(RESOURCE_GEOMETRIES.keys())
    for i, row in sorted(rows.items()):
        seen, errors = kernel_markers(profiles[i]['report'])
        old, old_errors = kernel_markers(old_profiles[i]['report'])
        affected = row['case']['bs'] > 16
        expected = expected_main if affected else Counter({'parent_false': 1})
        valid_parent = old == Counter({'parent_true' if affected else 'parent_false': 1}) and not old_errors
        passed = valid_parent and seen == expected and not errors
        out.append({'index': i, 'M': row['case']['bs'], 'affected': affected,
                    'observed': dict(seen), 'expected': dict(expected), 'problems': errors,
                    'parent_profile_matches': valid_parent, 'passed': passed})
    return {'cases': out, 'passed': all(r['passed'] for r in out),
            'limit': 'Visible kernel identities plus frozen source review; not exact runtime pointer tracing or causal attribution.'}


def load_plan(source_sha, root=ROOT):
    _, raw = bounded_file(PLAN, root); plan = json.loads(raw)
    require(plan['candidate_sha256'] == source_sha and source_sha != PARENT_SHA, 'Frozen candidate mismatch')
    require(plan['parent_sha256'] == PARENT_SHA and plan['parents'] ==
            [{'job': j, 'mode': m} for j, m in PARENTS], 'Frozen parents differ')
    require(plan['fixed_gate'] == GATE and plan['source_review_status'] == 'GO_FOR_OFFICIAL_CHECK',
            'Fixed gate/source review differs')
    pins = plan['evidence_sha256']
    required = {plan['candidate_path'], 'tools/analyze_moe_v30.py', 'tools/test_analyze_moe_v30.py',
                'tools/analyze_wgs_shared_a_result.py'}
    require(required <= set(pins), 'Missing source/analyzer/reader pins')
    for path, expected in pins.items():
        require(digest(bounded_file(path, root)[1]) == expected, 'Frozen evidence changed: ' + path)
    require(pins[plan['candidate_path']] == source_sha, 'Candidate path pin mismatch')
    return {'path': PLAN, 'sha256': digest(raw)}


def analyze(job, source_sha, root=ROOT):
    plan = load_plan(source_sha, root)
    current = read_run(job, source_sha, 'check', root)
    out = {'schema': 1, 'plan': plan, 'candidate': current['provenance'], 'automatic_promotion': False,
           'public_decoder_gain_established': False, 'full_serving_validated': False}
    if current['status'] != 'PASS':
        return {**out, **{k: v for k, v in current.items() if k != 'provenance'},
                'fixed_screen_passed': False, 'comparisons': [], 'resources': {}, 'activation': []}
    validate_result(current['result'])
    resources = read_resources(current, root)
    comparisons = []; activations = []
    for jid, mode in PARENTS:
        parent = read_parent(jid, mode, root)
        comparisons.append({'parent': parent['provenance'], **compare(current['result'], parent['result'])})
        activations.append({'parent_job': jid, **activation(current['result'], parent['result'])})
    passed = resources['passed'] and all(r['performance_gate_passed'] for r in comparisons) and all(r['passed'] for r in activations)
    return {**out, 'status': 'USEFUL_CHECK_REVIEW_BEFORE_SCORING' if passed else 'CLOSE_FIXED_ATTEMPT',
            'fixed_screen_passed': passed, 'score': current['result']['speedup'],
            'score_ci': current['result']['speedup_ci'], 'comparisons': comparisons,
            'resources': resources, 'activation': activations, 'flags': current['result']['flags'],
            'checks': current['result']['checks'], 'evaluator': current['result']['evaluator']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job'); parser.add_argument('source_sha256'); args = parser.parse_args()
    try:
        result = analyze(args.job, args.source_sha256)
    except (ValueError, KeyError, TypeError, OSError, UnicodeError) as error:
        result = {'status': 'EVIDENCE_REJECTED', 'reason': str(error), 'fixed_screen_passed': False,
                  'automatic_promotion': False}
    print(json.dumps(result, indent=2))
