#!/usr/bin/env python3
"""Build the isolated new-K/V packing change from exact scored87a9."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = '87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c'
PARENT = ROOT / f'experiments/candidates/muse_attention_decode/{PARENT_SHA}.py'
CANDIDATE = ROOT / 'muse_attention_decode/candidates/native_new_kv_pack_v18.py'
OLD = '''        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
        }'''
NEW = '''        const uint32_t w0 = e4m3x4f(x[0] * sc, x[1] * sc, x[2] * sc, x[3] * sc);
        const uint32_t w1 = e4m3x4f(x[4] * sc, x[5] * sc, x[6] * sc, x[7] * sc);'''
OLD_NAME = 'name=f"okc_attn_v298{tag}_{arch}",'
NEW_NAME = 'name=f"okc_attn_new_kv_pack_v18{tag}_{arch}",'
OLD_NOTE = '# CUDA source and normal compile/launch configuration remain unchanged.'
NEW_NOTE = ('# Isolated new-K/V packing experiment over scored87a9: two existing packed\n'
            '# FP8 helper calls replace the scalar cache-write loop. All other CUDA,\n'
            '# arithmetic, compile flags, launch configuration and stores remain unchanged.')


def once(text, old, new):
    assert text.count(old) == 1, (old[:60], text.count(old))
    return text.replace(old, new)


def build():
    text = PARENT.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == PARENT_SHA
    for old, new in ((OLD, NEW), (OLD_NAME, NEW_NAME), (OLD_NOTE, NEW_NOTE)):
        text = once(text, old, new)
    ast.parse(text)
    return text


if __name__ == '__main__':
    text = build()
    CANDIDATE.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), CANDIDATE)
