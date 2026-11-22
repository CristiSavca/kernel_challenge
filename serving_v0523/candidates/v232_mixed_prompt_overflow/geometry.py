"""Replay archived physical prompt admissions and graph-capture startup observations."""
from bisect import bisect_left
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[3]
archive = runpy.run_path(str(ROOT / 'experiments/iterations/20261118-serving-mixed-graph-capacity.py'))
folder = archive['ARCHIVE']
log_path = next(folder.glob('*v41__server.log'))
log = log_path.read_text()
info = json.loads(next(folder.glob('*v41__server_info.json')).read_bytes())['internal_states'][0]
load = json.loads(next(folder.glob('*v41__load.json')).read_bytes())
cap = info['chunked_prefill_size']
verify = info['cuda_graph_config']['decode']['bs']
prompt = info['cuda_graph_config']['prefill']['bs']
width = info['speculative_num_draft_tokens']
order = [r['concurrency'] for r in load['rungs'] for _ in range(r['reps'])]
observed = list(archive['read_trace'](log, order))
assert cap == prompt[-1] == 16384 and width == 8

def counts(extras, double):
    grid = sorted(set(prompt) | {cap + n * width for n in extras})
    result = {size: Counter() for size in sorted(set(order))}
    for size, (_, q, b) in observed:
        if not b or q + width*b <= cap:
            continue
        x = result[size]
        x['overflow_opportunities'] += 1
        rows = q + width*b
        idx = bisect_left(grid, rows)
        if idx >= len(grid):
            continue
        prompt_rows = prompt[bisect_left(prompt, q)]
        verify_rows = width * verify[bisect_left(verify, b)]
        extra = grid[idx] - prompt_rows - verify_rows
        if extra > (verify_rows if double else 0):
            continue
        x['additional_geometric_admissions_vs_unchanged_native_chunks'] += 1
        x['simultaneous_requests'] += b
        x['separate_verify_rows'] += verify_rows
        x['separate_prompt_rows'] += prompt_rows
        x['net_extra_body_rows'] += extra
        x['peak_extra_body_rows_per_forward'] = max(x['peak_extra_body_rows_per_forward'], extra)
    return result

def capture_samples():
    pat = r'Capturing prefill shape \(num_tokens=(\d+) avail_mem=([\d.]+) GB\):\s*\d+%\|.*?\| (\d+)/74 \[(\d+):(\d+)'
    samples = []
    previous = 0
    for line in log.splitlines():
        match = re.search(pat, line)
        if not match: continue
        rows, free, finished, minutes, seconds = match.groups()
        finished, elapsed = int(finished), 60*int(minutes) + int(seconds)
        if finished == len(samples)+1:
            samples.append(dict(rows=int(rows), elapsed_s=elapsed, delta_s=elapsed-previous,
                                avail_at_shape_start_gb=float(free)))
            previous = elapsed
    assert len(samples)==len(prompt) and samples[0]['rows']==cap
    return samples[:8]

policies = {
    'all_native_strict': (verify,False),
    'geometric_strict': ([1,2,4,8,16,32],False),
    'geometric_bounded': ([1,2,4,8,16,32],True),
    'four_bounded': ([4,8,16,32],True),
    'one_bounded': ([32],True),
}
receipt=dict(archive_log=str(log_path.relative_to(ROOT)),sha256=sha256(log_path.read_bytes()).hexdigest(),
    comparison='Identical native prompt chunks and historical running cohort. V229 may shorten a continuation on final bucket slack. Runtime readiness, allocations and graph-key reach can veto all additional calls.',
    captured_prompt_grid=prompt,captured_verify_grid=verify,actual_first_large_capture_samples=capture_samples(),
    policies={k:dict(verify_sizes=v[0],by_load=counts(*v)) for k,v in policies.items()})
print(json.dumps(receipt,indent=2))
