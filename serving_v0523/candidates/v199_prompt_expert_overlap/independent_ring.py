"""Audit release order in the compiled SM90a W2 loop and compare staged scale addressing."""
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

stage=Path(sys.argv[1])
header=stage/"sglang/python/sglang/kernels/jit/csrc/muse_spark/prompt_weight_pipeline.cuh"
adapter=stage/"sglang/python/sglang/srt/layers/muse_spark/prompt_weight_pipeline.py"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resident_bytes(path):
    data=path.read_bytes()
    offset=struct.unpack_from("<Q",data,40)[0]
    stride,count,_=struct.unpack_from("<HHH",data,58)
    sections=[struct.unpack_from("<IIQQQQIIQQ",data,offset+i*stride) for i in range(count)]
    def sec(i):
        h=sections[i]
        return data[h[4]:h[4]+h[5]]
    for i,h in enumerate(sections):
        if h[1]!=2: continue
        tab,names=sec(i),sec(h[6])
        for off in range(0,len(tab),h[9]):
            idx,_,_,section,value,_=struct.unpack_from("<IBBHQQ",tab,off)
            if names[idx:names.index(0,idx)]==b"SMEM_SIZE":
                return struct.unpack_from("<I",sec(section),value-sections[section][3])[0]
    raise ValueError("missing Humming dynamic shared memory symbol")


def drains(tag, expected):
    cub=stage/"rendered"/(tag+".cubin")
    source=stage/"rendered"/(tag+".sass")
    pending=0
    groups=[]
    releases=0
    started=False
    for line in source.read_text().splitlines():
        m=re.search(r"/\*([0-9a-f]+)\*/\s*(?:@[!\w]+\s+)?([A-Z][.A-Za-z0-9_]+)\b",line)
        if not m:continue
        op=m[2]
        if op.startswith("HGMMA."):
            assert op=="HGMMA.64x176x16.F32.BF16"
            pending+=1
            started=True
        if op.startswith("WARPGROUP.DEPBAR") and started:
            groups.append(pending)
            assert pending==expected
            pending=0
        if op.startswith("SYNCS.ARRIVE") and started and pending:
            # The parent releases the raw operand stage after its third drained K16;
            # the new pipeline defers that release until all four have drained.
            raise AssertionError((tag,m[1],pending))
        if op.startswith("SYNCS.ARRIVE") and started: releases+=1
    assert not pending and sum(groups)==32
    return dict(cubin_sha256=digest(cub),sass_sha256=digest(source),groups=groups,
                arrivals_observed_from_first_mma=releases,shared_memory_bytes=resident_bytes(cub))


def address():
    comparisons=0
    for n in (4096,8192):
        for expert in (0,1,15,31):
            for band in (0,1,n//256-1):
                for kstage in (0,1,7,63):
                    base=expert*256*n+(4*kstage)*n+band*256
                    staged={}
                    for loading_thread in range(128):
                        for j in (0,1):
                            idx=loading_thread+128*j
                            address=base+(idx//64)*n+(idx%64)*4
                            staged[idx]=list(range(address,address+4))
                    for math_thread in range(256):
                        lane,warp=math_thread%32,math_thread//32
                        for k in range(4):
                            idx=(lane//4)*2+warp%2+(warp//2)*16
                            for scale in range(4):
                                true=base+k*n+(warp//2)*64+(lane//4)*8+(warp%2)*4+scale
                                assert staged[k*64+idx][scale]==true
                                comparisons+=1
    return comparisons


def weight_bits():
    for byte in (0,*range(8,127)):
        bf=0 if not byte else (byte<<4)+(120<<7)
        number=0. if not byte else 2.**((byte//8)-7)*(1+(byte%8)/8)
        bits=struct.unpack(">I",struct.pack(">f",number))[0]>>16
        assert bits==bf
    return 120


if __name__=="__main__":
    assert digest(header)=="4d9fa4919819afe9d63132c261c178ec8ca245216ebda526e8d6669ee08556f7"
    candidate=drains("pipeline",4); parent=drains("4_256_4",2)
    assert candidate['shared_memory_bytes']-parent['shared_memory_bytes']==8192
    print(json.dumps(dict(header_sha256=digest(header),adapter_sha256=digest(adapter),parent=parent,
                          candidate=candidate,scale_byte_addresses=address(),
                          reachable_block_scale_values=weight_bits(),physical_gpu=False),indent=2))
