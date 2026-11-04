#!/usr/bin/env python3
"""Build only the WGS final route-scalar owner experiment from scored247888."""
import ast
import hashlib
from pathlib import Path

from build_wgs_shared_a_candidate import HOST as _RESOURCE_HOST

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = '247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0'
PARENT = ROOT / f'experiments/candidates/muse_moe_w4a16/{PARENT_SHA}.py'
CANDIDATE = ROOT / 'muse_moe_w4a16/candidates/native_wgs_final_scalar_v19.py'
HOST = (_RESOURCE_HOST.replace('okc_shared_a_resources', 'okc_final_scalar_resources')
        .replace('SSAR', 'FSR')
        .replace('shared-A WGS resource admission failed', 'final-scalar WGS resource admission failed'))
OWNER_OLD = '''        red[NW * 8 + tid] = tsum;'''
OWNER_NEW = '''        // The existing route owner publishes the scalar used by all output columns.
        float r = 0.f;
        if (S.pos[tk * 8 + tid] >= 0)
          r = p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f);
        red[NW * 8 + tid] = r;'''
CONSUMER_OLD = '''          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);'''
CONSUMER_NEW = '''          const float r = red[NW * 8 + sl];'''
CALL = '#if OKC_WGS\n    okc_final_scalar_resources(sm, e0, e1);\n#endif\n'
HEADER_OLD = '# Isolated KPERM3 adaptation over frozen ef7b210b; no SC6/MFIN or pipeline changes.'
HEADER_NEW = '# Isolated WGS final route-scalar owner reuse over scored247888; same reductions, casts and pipeline.'


def once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)


def modify_cuda(src):
    a = src.index('#if OKC_WGS && OKC_WGSFINSS\n')
    b = src.index('#elif OKC_FINCOL\n', a)
    finalizer = src[a:b]
    finalizer = once(finalizer, OWNER_OLD, OWNER_NEW)
    finalizer = once(finalizer, CONSUMER_OLD, CONSUMER_NEW)
    src = src[:a] + finalizer + src[b:]
    src = once(src, 'void okc_run(std::vector<torch::Tensor>', HOST + '\nvoid okc_run(std::vector<torch::Tensor>')
    return once(src, '    attr = true;', CALL + '    attr = true;')


def build():
    text = PARENT.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == PARENT_SHA
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    cuda = ast.literal_eval(node.value)
    text = once(text, cuda, modify_cuda(cuda))
    text = once(text, 'name=f"okc_moe_kperm3_v9_{tag}_sm{arch}",',
                'name=f"okc_moe_wgs_final_scalar_v19_{tag}_sm{arch}",')
    text = once(text, '            verbose=False,',
                '            verbose="-DOKC_WGS=1" in defs,  # retain WGS compiler evidence')
    text = once(text, HEADER_OLD, HEADER_NEW)
    ast.parse(text)
    return text


if __name__ == '__main__':
    text = build()
    CANDIDATE.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), CANDIDATE)
