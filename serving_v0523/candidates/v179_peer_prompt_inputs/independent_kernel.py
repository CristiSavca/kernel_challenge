"""Compile or interpret the production peer gather using exact BF16 storage bits."""
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import triton
import triton.language as tl
from compile_decode_sm90 import load_jit_module


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compile_cases(fn):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    signature = dict(zip(fn.arg_names, ["*i64","*i64","*i16","i32","constexpr","constexpr"]))
    compiled = []
    for hidden in (1, 7, 126, 257, 1024, 2048, 2049, 4096, 4099, 8192):
        tile = min(2048, triton.next_power_of_2(hidden))
        binary = triton.compile(ASTSource(fn, signature, constexprs=dict(HIDDEN=hidden, BLOCK=tile)),
                                target=GPUTarget("cuda", 90, 32))
        ptx = binary.asm["ptx"]
        assert re.search(r"ld\.global(?:\.[a-z0-9_:]+)*\.b64", ptx), ptx
        assert re.search(r"ld\.global(?:\.[a-z0-9_:]+)*\.b16", ptx), ptx
        assert re.search(r"st\.global(?:\.[a-z0-9_:]+)*\.b16", ptx), ptx
        assert re.search(r"(?:div|rem)\.s64", ptx), ptx
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch, "peer.ptx")
            path.write_text(ptx)
            log = subprocess.run([assembler, "-v", "-arch=sm_90a", path, "-o", str(path.with_suffix(".cubin"))],
                                 capture_output=True, text=True, check=True).stderr
            assert "0 bytes spill stores, 0 bytes spill loads" in log
            regs = int(re.search(r"Used (\d+) registers",log).group(1))
        compiled.append(dict(hidden=hidden,tile=tile,cubin=digest(binary.asm["cubin"]),registers=regs))
    return {"compiled":compiled,"no_spills":True}


def interpreted(fn, source):
    from interpret_decode_kernels import buf

    wrapper = next(n for n in ast.parse(source.read_text()).body
                   if isinstance(n,ast.FunctionDef) and n.name == "read_peer_rows")
    class Output:
        def __init__(self, physical, rows, width):
            self.raw = physical
            self.shape = rows, width

        def view(self, dtype):
            assert dtype == tl.int16
            return buf(self.raw, tl.int16)

    env = {"torch": SimpleNamespace(int16=tl.int16), "triton":triton, "_read_peer_rows":fn}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]), str(source), "exec"),env)
    run = env[wrapper.name]
    cases = guarded = copied = 0
    seen_bits = set()
    rng = np.random.default_rng(179)
    sentinel = 0xA55A
    for peers in (2,4,8):
        for hidden in (1,7,126,257,2049,4096,4099):
            shard = 7
            live = shard - 2
            total = peers * live * hidden
            original = (np.arange(total,dtype="u4") * 7919 + 311).astype("u2").reshape(peers,live,hidden)
            banks = [np.full(7 + shard*hidden + 11, sentinel, dtype="u2") for rank in range(peers)]
            for r, bank in enumerate(banks):
                bank[7:7+live*hidden] = original[r].reshape(-1)
            pointers = np.array([bank[7:].ctypes.data for bank in banks],dtype="i8")
            valid = np.array([r*shard + j for r in range(peers) for j in range(live)],dtype="i8")
            for active in ([], [valid[-1]], valid, rng.permutation(valid),
                           [valid[0],valid[-1],valid[0],valid[1]]):
                idx = np.array(active,dtype="i8")
                out = np.full(7 + idx.size*hidden + 11,sentinel,dtype="u2")
                run(buf(pointers,tl.int64),buf(idx,tl.int64),Output(out[7:7+idx.size*hidden],idx.size,hidden),shard)
                target = original[idx//shard,idx%shard].reshape(-1)
                actual = out[7:7+idx.size*hidden]
                np.testing.assert_array_equal(actual,target)
                assert (out[:7] == sentinel).all() and (out[7+idx.size*hidden:] == sentinel).all()
                copied += target.size
                seen_bits.update(int(i) for i in actual)
                guarded += 18
                cases += 1
            for r, bank in enumerate(banks):
                np.testing.assert_array_equal(bank[7:7+live*hidden].reshape(live,hidden),original[r])
                assert (bank[:7]==sentinel).all() and (bank[7+live*hidden:]==sentinel).all()
    assert len(seen_bits) == (1 << 16)
    # Bias only virtual bases. Every dereferenced address is inside a real host
    # allocation, while the global row ID and row-to-element product exceed int32.
    large = []
    for hidden in (3,4099):
        shard = (1 << 30) + 99
        local = min(shard-1,(1 << 31)//hidden + 43)
        banks = [rng.integers(0,65536,hidden+22,dtype="u2") for _ in range(4)]
        ptrs = np.array([bank[7:].ctypes.data - 2*local*hidden for bank in banks],dtype="i8")
        ids = np.array([3*shard+local,2*shard+local,local],dtype="i8")
        output = np.full(3*hidden+32,sentinel,dtype="u2")
        run(buf(ptrs,tl.int64),buf(ids,tl.int64),Output(output[9:9+3*hidden],3,hidden),shard)
        np.testing.assert_array_equal(output[9:9+3*hidden],np.concatenate([banks[i][7:7+hidden] for i in (3,2,0)]))
        assert (output[:9]==sentinel).all() and (output[9+3*hidden:]==sentinel).all()
        large.append(dict(max_row=int(max(ids)),local_row=int(local),element_offset=int(local*hidden)))
    return dict(cases=cases,copied_bf16_words=copied,distinct_bf16_bit_patterns=len(seen_bits),redzones=guarded,
                int64_cases=large)


if __name__ == "__main__":
    mode, src = sys.argv[1], Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as scratch:
        kernel = load_jit_module(src,Path(scratch))._read_peer_rows
        result = compile_cases(kernel) if mode=="compile" else interpreted(kernel,src)
    print(json.dumps(dict(mode=mode,triton=triton.__version__,physical_gpu=False,source_sha256=digest(src.read_bytes()),**result),indent=2))
