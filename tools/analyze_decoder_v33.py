#!/usr/bin/env python3
"""Fixed V33 screen: V29 performance gates plus scalar-storage and full activation gates."""
import argparse
from collections import Counter
import json
import re
from analyze_decoder_v28 import ROOT, PARENT_SHA, PARENTS, benchmarks, read_run, read_file, require, sha
from analyze_decoder_v29 import GATE, AFFECTED, compare, compressed_instances

PLAN = 'experiments/iterations/20261104-decoder-nibble-sidecar-v33-root-plan.json'
MARKER = 'sk2_nibble_sidecar_v33'
RESOURCE_RE = re.compile(r'V33R t=([0-9,]+) x=([0-9,]+) nt=(\d+) dyn=(\d+) '
                         r'attr=([0-9,]+) occ=(-?\d+) api=([0-9,]+)')


def resources(log):
    rows = []; problems = []; pointer_attributes = {}
    require(isinstance(log, str), 'Missing diagnostic log')
    for line in log.splitlines():
        if line.startswith('V33R_STOP'):
            problems.append('Resource coverage stopped'); continue
        if not line.startswith('V33R '):
            continue
        m = RESOURCE_RE.fullmatch(line)
        if not m:
            problems.append('Malformed resource observation'); continue
        t = list(map(int, m[1].split(',')))
        sample = list(map(int, m[2].split(',')))
        attr = list(map(int, m[5].split(',')))
        api = list(map(int, m[7].split(',')))
        require(len(t) == 15 and len(sample) == len(attr) == 4 and len(api) == 2,
                'Resource field count mismatch')
        row = dict(template=t, runtime_sample=sample, threads=int(m[3]),
                   dynamic_shared_bytes=int(m[4]), registers=attr[0], local_bytes=attr[1],
                   static_shared_bytes=attr[2], max_dynamic_shared_bytes=attr[3],
                   capacity=int(m[6]), api=api, compressed=bool(t[7]))
        identity = tuple(t)
        observation = (row['threads'], row['dynamic_shared_bytes'], tuple(attr), row['capacity'], tuple(api))
        if identity in pointer_attributes and pointer_attributes[identity] != observation:
            problems.append('Conflicting compiled-pointer attributes')
        pointer_attributes[identity] = observation
        valid = (api == [0, 0] and row['capacity'] >= 1 and 0 < attr[0] <= 255
                 and attr[3] >= row['dynamic_shared_bytes'])
        if row['compressed'] and (not valid or row['local_bytes'] != 0):
            problems.append('Compressed pointer retains local storage or failed admission')
        rows.append(row)
    signatures = {tuple(x['template'][:7]) for x in rows if x['compressed']}
    if not signatures:
        problems.append('No compressed pointer observations')
    return dict(rows=rows, problems=problems, passed=not problems), signatures


def activation(result, parent, signatures):
    rows, _, profiles = benchmarks(result)
    _, _, anchors = benchmarks(parent)
    findings = []
    for i, row in rows.items():
        affected = row['case']['load'] in AFFECTED
        old = compressed_instances(anchors[i]['report'], 'sk2_kernel')
        new = compressed_instances(profiles[i]['report'], MARKER)
        residual = sum((compressed_instances(profiles[i]['report'], marker)
                        for marker in ('sk2_kernel', 'sk2_exact_patch_v29', 'sk2_exact_patch_v32')), Counter())
        observed = set(new) <= signatures
        passed = (bool(old) and new == old and not residual and observed if affected
                  else not old and not new and not residual)
        findings.append(dict(index=i, affected=affected, passed=bool(passed),
                             expected=[list(x) for x in old.elements()],
                             observed=[list(x) for x in new.elements()],
                             matching_resource_signatures=observed))
    return dict(cases=findings, passed=all(x['passed'] for x in findings))


def analyze(job, digest):
    raw = read_file(PLAN)[1]; plan = json.loads(raw)
    require(plan['candidate_sha256'] == digest and plan['fixed_gate'] == GATE,
            'Frozen source/performance gate mismatch')
    require(plan['source_review_status'] == 'GO_FOR_OFFICIAL_CHECK'
            and plan['compiler_gate'] == 'All observed COMP local_bytes=0; no resource STOP; active signatures covered',
            'Source/compiler review gate mismatch')
    for path, expected in plan['evidence_sha256'].items():
        require(sha(read_file(path)[1]) == expected, 'Frozen evidence changed: ' + path)
    candidate = read_run(job, digest, 'check')
    out = dict(schema=1, plan_sha256=sha(raw), candidate=candidate['provenance'],
               automatic_promotion=False, full_serving_validated=False)
    if candidate['status'] != 'PASS':
        return {**out, **{k:v for k,v in candidate.items() if k != 'provenance'},
                'fixed_screen_passed': False}
    receipt = json.loads(read_file(candidate['provenance']['receipt'])[1])['response']
    log = candidate['result'].get('log_tail')
    saved = read_file(receipt['output']['log'])[1].decode()
    require(isinstance(log, str) and log.startswith(saved), 'Saved log/receipt disagreement')
    resource, signatures = resources(log)
    resource.update(log_tail_sha256=sha(log.encode()), saved_log_sha256=sha(saved.encode()),
                    scope='First normal compiled-pointer observations, not achieved occupancy or isolated causality.')
    comparisons = []; activations = []
    for pid, mode in PARENTS:
        p = read_run(pid, PARENT_SHA, mode)
        require(p['status'] == 'PASS', 'Exact parent is not passing')
        comparisons.append(dict(parent=p['provenance'], **compare(candidate['result'], p['result'])))
        activations.append(dict(parent_job=pid, **activation(candidate['result'], p['result'], signatures)))
    passed = (resource['passed'] and all(x['performance_gate_passed'] for x in comparisons)
              and all(x['passed'] for x in activations))
    return {**out, 'status': 'USEFUL_CHECK_REVIEW_BEFORE_SCORING' if passed else 'CLOSE_FIXED_ATTEMPT',
            'fixed_screen_passed': passed, 'resources': resource, 'comparisons': comparisons,
            'activation': activations, 'score': candidate['result']['speedup'],
            'score_ci': candidate['result']['speedup_ci'], 'checks': candidate['result']['checks'],
            'flags': candidate['result']['flags'], 'evaluator': candidate['result']['evaluator']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('job'); p.add_argument('source_sha256')
    args = p.parse_args(); print(json.dumps(analyze(args.job, args.source_sha256), indent=2))
