"""Summarize the compiled fused gate/up images and inspect each steady-state K loop."""
import hashlib
import json
from pathlib import Path
import re
import sys

images, rendered, model = map(Path, sys.argv[1:])
live = {}
src = model / 'python/sglang/srt/layers/muse_spark/live_decode_reduce.py'
exec(compile(src.read_text(), str(src), 'exec'), live)
output = {}
for stem, config in json.loads(rendered.read_text()).items():
    prefix = images / stem
    get = lambda suffix: prefix.with_suffix(suffix).read_bytes()
    sass = get('.sass').decode()
    log = get('.log').decode()
    m, n, _ = config['block']
    stages, bands = config['stages'], config['warp'][1] // 16
    assert hashlib.sha256(get('.cu')).hexdigest() == config['source_sha256']
    frame, stores, loads = map(int, re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log).groups())
    assert frame == stores == loads == 0 and 'C7512' not in log
    assert live['has_live_decode_isa'](sass, m, bands=bands) and 'UTMALDG.4D' in sass
    insts = [(int(addr, 16), op) for addr, op in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?)\s*/\* 0x', sass)]
    loops = {}
    for stop, op in insts:
        jump = re.search(r'\bBRA(?:\.U)? (0x[0-9a-f]+)', op)
        if not jump or (start := int(jump[1], 16)) >= stop:
            continue
        body = [item for address, item in insts if start <= address <= stop]
        tensor = [int(hit[1]) for item in body if (hit := re.search(r'HGMMA\.64x(\d+)x16\.F32\.BF16', item))]
        if len(set(tensor)) != 1 or len(tensor) != stages * 4 * bands:
            continue
        width = tensor[0]
        if width not in loops or len(body) < loops[width]['instructions']:
            loops[width] = dict(instructions=len(body), HGMMA=len(tensor),
                                drains=sum('WARPGROUP.DEPBAR.LE' in x for x in body),
                                local_load_store=sum(bool(re.search(r'\b(?:LDL|STL)\b', x)) for x in body),
                                begin=hex(start), end=hex(stop))
    assert set(loops) == set(range(8,m+1,8)), (m,n,loops)
    assert all(x['local_load_store'] == 0 for x in loops.values())
    output[stem] = config | dict(registers=int(re.search(r'Used (\d+) registers', log)[1]),
                                frame=frame, spill_stores=stores, spill_loads=loads, loops=loops,
                                cubin_sha256=hashlib.sha256(get('.cubin')).hexdigest(),
                                sass_sha256=hashlib.sha256(get('.sass')).hexdigest(),
                                TMA4D=sass.count('UTMALDG.4D'), ptxas_serialized=False,
                                passes_production_instruction_check=True)
print(json.dumps(output, indent=2, sort_keys=True))
