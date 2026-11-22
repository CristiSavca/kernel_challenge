"""Compare one kernel and its inner reduction loop from two cuobjdump --dump-sass files."""
import collections
import gzip
import json
from pathlib import Path
import re
import sys


def instructions(path):
    data = Path(path).read_bytes()
    if Path(path).suffix == '.gz':
        data = gzip.decompress(data)
    sections = data.decode().split('Function : ')
    section = next(s for s in sections if s.startswith('_ZN6sglang19muse_skinny_gate_up10run_kernel'))
    pattern = r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P(?:\d+|T)\s+)?([A-Z0-9_.]+)\b(.*?);'
    decoded = [(int(pc,16),op,arg.strip()) for pc,op,arg in re.findall(pattern,section)]
    hmma = [pc for pc, op, _ in decoded if op.startswith('HMMA.')]
    backedges = [(int(arg,16),pc) for pc,op,arg in decoded if op == 'BRA' and arg.startswith('0x')
                 and int(arg,16) < pc and all(int(arg,16) <= site <= pc for site in hmma)]
    lo,hi=min(backedges,key=lambda edge:edge[1]-edge[0])
    def count(sequence):
        ops=collections.Counter(o for _,o,_ in sequence)
        return dict(instructions=len(sequence),opcodes=dict(sorted(ops.items())))
    return dict(entire_kernel=count(decoded),k_loop=count([r for r in decoded if lo<=r[0]<=hi]),
                k_loop_start=hex(lo),k_loop_backedge=hex(hi))


if __name__=='__main__':
    print(json.dumps(dict(parent=instructions(sys.argv[1]),candidate=instructions(sys.argv[2])),indent=2))
