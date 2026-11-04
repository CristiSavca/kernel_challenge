#!/usr/bin/env python3
"""Build the fixed distance3 weight-prefetch experiment from scored247888."""
import ast
import hashlib
from pathlib import Path
from build_wgs_shared_a_candidate import HOST as _RESOURCE_HOST

ROOT=Path(__file__).resolve().parents[1]
PARENT_SHA='247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0'
PARENT=ROOT/f'experiments/candidates/muse_moe_w4a16/{PARENT_SHA}.py'
CANDIDATE=ROOT/'muse_moe_w4a16/candidates/native_wgs_l2_prefetch_v17.py'
HOST=(_RESOURCE_HOST.replace('okc_shared_a_resources','okc_l2_prefetch_resources')
      .replace('SSAR','LPFR')
      .replace('shared-A WGS resource admission failed','L2-prefetch WGS resource admission failed'))
PREFETCH=r'''      // Fixed distance3 in this phase's actual rotated sequence; only immutable weights.
      if (i + 3 < g1 - g0) {
        int pgs = gs + 3;
        if (pgs >= g1) pgs = g0 + (pgs - g1);
        const int pu = pgs >> 4, pst = pgs & 15;
        const int pj = pu >> ubl, pub = pu & ((1 << ubl) - 1);
        const unsigned char* psrc = base + (size_t)(((S.jE[pj] << ubl) + pub) * 16 + pst) * (size_t)UNIT_BYTES;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
                     :: "l"(psrc), "r"(UNIT_BYTES) : "memory");
      }
'''

def once(text,old,new):
    assert text.count(old)==1,(old[:100],text.count(old))
    return text.replace(old,new)

def modify_cuda(src):
    a=src.index('__device__ void wgs_wproducer(');b=src.index('// x producer:',a)
    producer=src[a:b]
    anchor='      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }'
    producer=once(producer,anchor,PREFETCH+anchor)
    src=src[:a]+producer+src[b:]
    src=once(src,'void okc_run(std::vector<torch::Tensor>',HOST+'\nvoid okc_run(std::vector<torch::Tensor>')
    return once(src,'    attr = true;','#if OKC_WGS\n    okc_l2_prefetch_resources(sm, e0, e1);\n#endif\n    attr = true;')

def build():
    text=PARENT.read_text();assert hashlib.sha256(text.encode()).hexdigest()==PARENT_SHA
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    cuda=ast.literal_eval(node.value);text=once(text,cuda,modify_cuda(cuda))
    text=once(text,'name=f"okc_moe_kperm3_v9_{tag}_sm{arch}",','name=f"okc_moe_wgs_l2_prefetch_v17_{tag}_sm{arch}",')
    text=once(text,'            verbose=False,','            verbose="-DOKC_WGS=1" in defs,  # retain WGS compiler evidence')
    text=once(text,'# Isolated KPERM3 adaptation over frozen ef7b210b; no SC6/MFIN or pipeline changes.',
        '# Isolated distance3 bulk L2 weight prefetch over scored247888; no SC6 or pipeline changes.\n# Real TMA copies, barriers, weight encoding, arithmetic and dispatch remain the parent implementation.')
    ast.parse(text);return text

if __name__=='__main__':
    text=build();CANDIDATE.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(),CANDIDATE)
