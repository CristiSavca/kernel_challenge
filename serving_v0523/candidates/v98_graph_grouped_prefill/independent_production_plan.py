"""Exercise the exact V98 host-side wave planner using CPU PyTorch on edge cases."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def read_planner(stage):
    source = stage / "python/sglang/srt/layers/muse_spark/humming_prefill_waves.py"
    tree = ast.parse(source.read_text())
    names = {"plan_waves", "wave_capacity"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    triton = SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b, next_power_of_2=lambda n: 1 << (n-1).bit_length())
    env = dict(torch=torch, triton=triton)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
    return env["plan_waves"], env["wave_capacity"], hashlib.sha256(source.read_bytes()).hexdigest()


def run(stage):
    plan, capacity, source_sha = read_planner(stage)
    rng = np.random.default_rng(8104939)
    rng_tokens = [1, 3, 8, 31, 32, 33, 129, 8192, 16384, 24576]
    sizes = []
    plans = live_total = checked_outputs = 0
    torch.set_num_threads(2)
    for tokens in rng_tokens:
        k, e = 8, 32
        c = capacity(tokens, k, e, 4096, 4096)
        assert c > 0
        rows = c+k-1
        waves = (tokens*k+c-1)//c
        sizes.append(dict(tokens=tokens, capacity=c, rows=rows, static_waves=waves))
        for mode in ('empty', 'all_local', 'dominant_last', 'one_per_token', 'cross_boundary', 'random'):
            ids = rng.integers(0, e, size=(tokens,k), dtype=np.int32)
            if mode == 'empty':
                ids.fill(-1)
            elif mode == 'dominant_last':
                ids.fill(e-1)
                ids[::3] = e+2
            elif mode == 'one_per_token':
                ids[:, 1:] = -1
            elif mode == 'cross_boundary':
                ids[:max(0,c//k-1),-1] = -1
                ids[-1] = -1
            elif mode == 'random':
                ids[rng.random(ids.shape) < .82] = -1
            live = (ids>=0) & (ids<e)
            count = live.sum(1)
            starts = np.cumsum(count) - count
            owners = np.minimum(starts//c, waves-1)
            bins, slots, edges, ranges = [x.numpy() for x in plan(torch.from_numpy(ids), e, c)]
            assert (ranges[:-1] <= ranges[1:]).all() and (ranges[0],ranges[-1])==(0,tokens)
            token_writes = np.zeros(tokens, np.int8)
            slot_writes = np.zeros((tokens,k), np.int8)
            for w in range(waves):
                first_token, last_token = ranges[w:w+2]
                token_writes[first_token:last_token] += 1
                np.testing.assert_array_equal(owners[first_token:last_token], w)
                off = edges[w*e:(w+1)*e+1]
                n = int(off[-1]-off[0])
                assert 0<=n<=rows
                packed_bounds = off+rows-off[-1]
                expected_counts = np.bincount(ids[first_token:last_token][live[first_token:last_token]],minlength=e)
                np.testing.assert_array_equal(np.diff(np.r_[packed_bounds[:-1],rows]),expected_counts)
                flat = slots[off[0]:off[-1]]
                token,slot = np.divmod(flat,k)
                assert np.all((first_token<=token)&(token<last_token))
                expert = bins[off[0]:off[-1]] % e
                np.testing.assert_array_equal(expert,ids[token,slot])
                row = np.arange(rows-n,rows)
                assert np.all((packed_bounds[expert]<=row)&(row<packed_bounds[expert+1]))
                np.add.at(slot_writes,(token,slot),1)
                # The exact inverse then preserves all original slot ordinals, including repeats of one expert.
                for j in np.unique(expert):
                    span=flat[expert==j]
                    assert np.all(span[1:] > span[:-1])
            np.testing.assert_array_equal(token_writes,np.ones_like(token_writes))
            np.testing.assert_array_equal(slot_writes,live)
            plans += 1
            live_total += int(live.sum())
            checked_outputs += tokens
    return dict(planner_source_sha256=source_sha, torch=str(torch.__version__), plans=plans,
                token_writes=checked_outputs, live_slots=live_total, fixed_sizes=sizes)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    print(json.dumps(run(parser.parse_args().stage),indent=2))
