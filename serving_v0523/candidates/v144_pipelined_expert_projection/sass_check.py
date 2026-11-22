"""Report the hot K backedge and distances from vector LDG to a register source use.

Usage: python sass_check.py base-sm90a.sass candidate-sm90a.sass
Distances count static instruction sites. Hopper control fields follow CuAssembler.
Completion times and dynamic stalls require hardware.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


def sections(text):
    return {b.splitlines()[0]:b for b in text.split("Function : ")[1:]}


def get_phase(body,required_hmma):
    lines={int(pc,16):inst for pc,inst in re.findall(r"^\s*/\*([\da-f]+)\*/\s+(.*?)\s*;\s*/\*",body,re.M)}
    words=re.findall(r"^\s*/\*([\da-f]+)\*/[^\n]+/\* 0x[\da-f]+ \*/\s*/\* 0x([\da-f]+) \*/",body,re.M)
    controls={int(pc,16):int(high,16)>>41 & 0x1ffff for pc,high in words}
    loops=[]
    for pc,op in lines.items():
        target=re.search(r"\bBRA (0x[\da-f]+)",op)
        if target is None: continue
        start=int(target.group(1),16)
        seq=[x for k,x in lines.items() if start<=k<=pc]
        if sum("HMMA." in x for x in seq)==required_hmma and sum("LDG.E." in x for x in seq)>=2:
            loops.append((len(seq),start,pc,seq))
    length,start,end,seq=min(loops)
    reads=[]
    for index,inst in enumerate(seq):
        load=re.search(r"LDG\.E\.(64|128) R(\d+)",inst)
        if not load: continue
        write_barrier=(controls[start+index*16]>>5)&7
        assert write_barrier < 6
        wait_index=next(
            i for i in range(index+1,index+length+1)
            if (controls[start+(i%length)*16]>>11) & (1<<write_barrier)
        )
        span=int(load[1])//32
        regs=set(range(int(load[2]),int(load[2])+span))
        for next_index in range(index+1,index+length+1):
            # Integer and conversion operands are explicit in these hot loops.
            # HMMA's A/B/C operands name the first register of each group.
            op=seq[next_index%length]
            plain=op.split(" ",1)[1] if op.startswith("@") else op
            opcode,_,args=plain.partition(" ")
            operands=args.split(",")
            sources=operands[1:]
            used={int(r) for r in re.findall(r"\bR(\d+)",",".join(sources))}
            if opcode.startswith("HMMA"):
                for q,n in zip(sources,[4,2,4]):
                    r=re.search(r"\bR(\d+)",q)
                    if r:used.update(range(int(r[1]),int(r[1])+n))
            if regs & used:
                reads.append(dict(load_site=index,bytes_per_lane=span*4,write_barrier=write_barrier,
                                  first_explicit_wait_site=wait_index,static_gap_to_wait=wait_index-index,
                                  first_source_site=next_index,static_gap=next_index-index))
                break
        else:raise AssertionError("missing register consumer")
    return dict(instruction_sites=length,loop_begin=hex(start),loop_end=hex(end),bank_loads=reads,
                hmma_sites=[i for i,x in enumerate(seq) if "HMMA" in x],function_sites=len(lines))


def check(path):
    text=path.read_text()
    result={}
    for name,body in sections(text).items():
        if "muse_route_pair17projection_kernelIL" in name:
            gate="ILb1" in name
            result["W13" if gate else "W2"]=get_phase(body,8 if gate else 4)
    return dict(sha256=sha256(text.encode()).hexdigest(),phases=result)


if __name__=="__main__":
    print(json.dumps(dict(parent=check(Path(sys.argv[1])),candidate=check(Path(sys.argv[2]))),indent=2))
