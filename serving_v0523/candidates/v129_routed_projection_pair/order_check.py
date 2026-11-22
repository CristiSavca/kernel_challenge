"""Run the V126 cancellation examples through both actual V129 projections."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import warp_check as work

v86 = work.v86


def check(dll, split):
    width = 128
    code_g = np.zeros((1, 2*width, width), np.uint8)
    code_d = np.zeros((1, width, width), np.uint8)
    x = np.zeros((1,width), np.float32)
    eps = 2.0**-27
    x[0, [0,16,64]] = [1,eps,1]
    x = v86.bf16(x)
    # Serial W13 gate: +1, epsilon, -1. Other gates are one, and their up
    # terms generate activation values A, A, A*epsilon for the down example.
    code_g[0,0,[0,16,64]] = [2,2,10]
    code_g[0,width,0] = 2
    positions = [1,width//split+1,width//split+17]
    for n, src in zip(positions,[0,0,16]):
        code_g[0,n,0] = 2
        code_g[0,width+n,src] = 2
    code_d[0,0,positions] = [2,10,2]
    gate,down = map(v86.pack_codes,[code_g,code_d])
    gs = np.full((1,width//16,2*width),56,np.uint8)
    ds = np.full((1,width//16,width),56,np.uint8)
    glob = v86.bf16([1])
    active = np.full((1,width),0x7fc1,np.uint16)
    output = active.copy()
    plan = np.array([1,0,0,*([-1]*7)],np.int32)
    dll.simulate(*(a.ctypes.data for a in [x,gate,gs,glob,down,ds,glob,active,output,plan]),width,width,1,1)
    a = v86.floats(active[0,positions])
    assert active[0,0] == 0 and a[0] == a[1] and a[0] > 0 and a[2] == a[0]*eps
    assert v86.floats(output[0,0]) == a[2]
    return dict(split_sensitivity_example=split,gate_activation=float(v86.floats(active[0,0])),
                down_output=float(v86.floats(output[0,0])),serial_down=float(a[2]))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        dll, sha = work.compile_device(Path(sys.argv[1]),Path(t))
        print(json.dumps(dict(header_sha256=sha,actual_kernel_with_cpu_mma=True,
                              cases=[check(dll,p) for p in (2,4)]),indent=2))
