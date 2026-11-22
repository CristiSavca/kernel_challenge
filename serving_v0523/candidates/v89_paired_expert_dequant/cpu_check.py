"""Re-run V86's complete CTA simulator using V89's exact packed dequantization helpers."""

import argparse
import importlib.util
import json
import pathlib
import re
import subprocess
import tempfile

import numpy as np

base = pathlib.Path(__file__).resolve().parents[1] / 'v86_skinny_expert_gate_up' / 'cpu_check.py'
spec = importlib.util.spec_from_file_location('v86_source_harness', base)
v86 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v86)

HOST_MUL = """
uint32_t pair_mul(uint32_t x, uint32_t y) {
    uint32_t result = 0;
    for (uint32_t shift = 0; shift <= 16; shift += 16) {
        float lhs = cast<float>(bf16_t(x >> shift));
        float rhs = cast<float>(bf16_t(y >> shift));
        result |= uint32_t(cast<bf16_t>(lhs * rhs)) << shift;
    }
    return result;
}
"""
PAIR_EXPORT = """
extern "C" uint32_t pair(uint32_t w, uint32_t s, uint32_t f) {
    using namespace sglang::muse_skinny_gate_up;
    return operand_pair(w, broadcast_scale(s), f);
}
"""


def run_tests(checkout):
    import ctypes
    header = pathlib.Path('python/sglang/kernels/jit/csrc/muse_spark/expert_skinny_gate_up.cuh')
    production = (checkout / header).read_text()
    # Replace only the two new PTX instructions. Keep their actual operands,
    # immediate constants, packed addressing and the production C++ bit logic.
    emulated, count = re.subn(
        r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
        r'\1 = pair_mul(\2, \3);', production)
    assert count == 2
    with tempfile.TemporaryDirectory() as name:
        stage = pathlib.Path(name)
        (stage/header.parent).mkdir(parents=True)
        (stage/header).write_text(emulated)
        v86.PRELUDE += HOST_MUL
        v86.POSTLUDE += PAIR_EXPORT
        sim = v86.load_kernel(stage, stage)
        lib = ctypes.CDLL(str(stage/'simulate.so'))
        pair = lib.pair
        pair.restype = ctypes.c_uint32
        pair.argtypes = [ctypes.c_uint32]*3
        lut = np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], np.float32)
        rng = np.random.default_rng(89243)
        values = 0
        for byte in range(127):
            prepared = 0 if byte < 8 else byte
            expected = v86.bf16(lut * v86.convert_scales(np.array([prepared],dtype=np.uint8))[0])
            for low in range(16):
                for high in range(16):
                    for frag in range(4):
                        word = int(rng.integers(0,2**32,dtype=np.uint32))
                        word = (word & ~(0x000F000F << (frag*4))) | ((low|(high<<16))<<(frag*4))
                        result = pair(word, prepared, frag)
                        assert result == (int(expected[low]) | (int(expected[high]) << 16))
                        values += 1
        shapes = [(64,64,[0,0,0,0],7),(128,64,[1,0,0,1],4),
                  (128,128,[2,3,7,8],9),(256,128,[0,0,9,18],2),
                  (128,192,[0,17,1,8,3,0,1,7],1),(4096,64,[0,1,0],7),
                  (128,4096,[0,1],3),(64,64,[32,0,0,0],0)]
        total = padding = 0
        for integral in [True,False]:
            for args in shapes:
                n, z = v86.run_case(sim,rng,*args,integral)
                total += n
                padding += z
    return dict(packed_operand_cases=values,cta_cases=len(shapes)*2,
                live_bf16_values=total,unchanged_inactive_prefix=padding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkout',type=pathlib.Path)
    print(json.dumps(run_tests(parser.parse_args().checkout),indent=2))
