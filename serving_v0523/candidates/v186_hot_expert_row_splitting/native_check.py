"""Compile the standalone planner for this CPU and compare its output with the Python oracle."""
import ctypes as ct
from dataclasses import replace
from pathlib import Path
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from check import BASE, BUDGET, OWNERS, T, completion, native_slots
from planner import plan_splits

HERE = Path(__file__).resolve().parent
I, L, U, D = ct.c_int32, ct.c_int64, ct.c_uint64, ct.c_double


class Config(ct.Structure):
    _fields_ = [(name, L) for name in ('tile','weight','hidden','intermediate','workspace')]
    _fields_ += [(name, D) for name in ('owner_ns','helper_ns','link','setup','planning','margin')]


class Move(ct.Structure):
    _fields_ = [(name, I) for name in ('owner','helper','expert','begin','end')]


def verify(library):
    impl = ct.CDLL(library).plan_hot_rows
    impl.argtypes = [I, I, ct.POINTER(I), ct.POINTER(I), ct.POINTER(U), ct.POINTER(Config),
                     ct.POINTER(Move), ct.POINTER(U), ct.POINTER(D)]
    impl.restype = I
    def case(counts, owners, budget, c, timings=False):
        ranks = len(budget)
        config = Config(c.rows_per_tile,c.weight_bytes,c.hidden,c.intermediate,c.helper_workspace_bytes,
                        c.owner_ns_per_issued_row,c.helper_ns_per_issued_row,c.link_bytes_per_ns,
                        c.launch_ns_per_move,c.global_planning_ns,c.minimum_saving_ns)
        n = len(counts)
        out, used, times = (Move*ranks)(), (U*ranks)(), (D*2)()
        args = (ranks,n,(I*n)(*counts),(I*n)(*owners),(U*ranks)(*budget),ct.byref(config),out,used,times)
        result = impl(*args)
        ref = plan_splits(counts, owners, budget, c)
        assert result == len(ref.moves)
        for m, x in zip(out[:result],ref.moves):
            assert tuple(getattr(m,k) for k,t in m._fields_) == (x.owner,x.helper,x.expert,x.begin,x.end)
        assert tuple(used) == ref.extra_bytes_by_rank
        assert np.allclose(times[:],(ref.original_ns,ref.projected_ns), rtol=1e-13, atol=1e-8)
        if timings:
            samples = []
            for _ in range(19):
                start = time.perf_counter_ns()
                for i in range(8):
                    assert impl(*args) == result
                samples.append((time.perf_counter_ns()-start)/8000)
            return round(statistics.median(samples),2)
        return int(bool(result))
    rng = np.random.default_rng(186)
    timings = {}
    for name in ('uniform','single_hot','four_local','observed_even','observed_other_hot'):
        if name.startswith('observed'):
            counts = completion(0 if name == 'observed_even' else 38000)
        else:
            ids = native_slots(rng, T, name)
            counts = np.bincount(ids.ravel(), minlength=256).tolist()
        timings[name] = case(counts,OWNERS,BUDGET,BASE,True)
    positive = 0
    for i in range(400):
        ranks = int(rng.integers(1, 10))
        n = int(rng.integers(0,48))
        counts = rng.integers(0,1200, size=n).tolist()
        owners = rng.integers(0,ranks,size=n).tolist()
        budget = rng.integers(0,2**23,size=ranks).tolist()
        c = replace(BASE, rows_per_tile=int(rng.choice([16,32,128,176])), weight_bytes=int(rng.integers(4,2**17)),
                    helper_workspace_bytes=0, hidden=32, intermediate=64, owner_ns_per_issued_row=165,
                    helper_ns_per_issued_row=int(rng.choice([101,165,282])), global_planning_ns=30,
                    launch_ns_per_move=15, minimum_saving_ns=5)
        positive += case(counts,owners,budget,c)
    assert positive >= 200
    return dict(native_random_cases=400, native_positive=positive, correct_geometry_cases=list(timings),
                native_median_us_with_C_ABI_call=timings, host=os.uname().machine,
                excludes='histogram materialization and D2H, all-rank agreement, native planner integration, H200 transport and GPU kernels')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print(json.dumps(verify(sys.argv[1]),indent=2))
    else:
        with tempfile.TemporaryDirectory(prefix='okc-v186-cpu-') as directory:
            shared = str(Path(directory)/'planner.so')
            cc = shutil.which('c++')
            subprocess.run([cc,'-O3','-std=c++17','-fPIC','-shared',str(HERE/'planner.cc'),'-o',shared], check=True)
            subprocess.run([sys.executable,__file__,shared],check=True)
