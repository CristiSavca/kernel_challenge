#!/usr/bin/env python3
"""CPU address, ownership and asynchronous-ring model for SK2 global escapes.

Does not establish CUDA compilation, GPU correctness, races or performance.
"""
import ast
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "muse_decode_layer/submission.py"
CANDIDATE = ROOT / "muse_decode_layer/candidates/native_dense_escape_global_v10.py"
EXPECTED_PARENT = "80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c"


def read_geometry():
    tree = ast.parse(PARENT.read_text())
    names = {"bfp_SK2_CONFIGS", "bfp_SK2_PICK", "bfp_SK2_K"}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bfp__sk2_geom"]
    env = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<geometry-only>", "exec"), env)
    return env


def ring_model(nw, depth, stages, seed, expected_delta=0):
    """Consumer order is preserved; producer/warps/escapes otherwise interleave.

    The memory read and release are separate events. An early-refill mutation
    must be detected even when a consumer has not copied its operand yet.
    """
    rng = random.Random(seed)
    memory = [None] * depth
    arrivals = [0] * depth
    need = [0] * nw
    has_read = [False] * nw
    issue = 0
    # Escape completions are independent and deliberately can happen last.
    for _ in range((2 * nw + 1) * stages * 2):
        if issue == stages and all(k == stages for k in need):
            return True
        actions = []
        slot = issue % depth
        if issue < stages and (issue < depth or arrivals[slot] >= nw + expected_delta):
            actions.append(("produce", -1))
        for w in range(nw):
            if need[w] < stages and memory[need[w] % depth] == need[w]:
                actions.append(("release" if has_read[w] else "read", w))
        if not actions:
            return False
        action, w = rng.choice(actions)
        if action == "produce":
            old = memory[slot]
            if old is not None and any(need[j] <= old for j in range(nw)):
                return False
            memory[slot] = issue
            arrivals[slot] = 0
            issue += 1
        elif action == "read":
            assert memory[need[w] % depth] == need[w]
            has_read[w] = True
        else:
            assert has_read[w]
            arrivals[need[w] % depth] += 1
            need[w] += 1
            has_read[w] = False
    return False


def main():
    parent = PARENT.read_bytes()
    assert hashlib.sha256(parent).hexdigest() == EXPECTED_PARENT
    candidate = CANDIDATE.read_text()
    pt, ct = ast.parse(parent), ast.parse(candidate)
    changed = []
    assert len(pt.body) == len(ct.body)
    for p, c in zip(pt.body, ct.body):
        if ast.dump(p, include_attributes=False) != ast.dump(c, include_attributes=False):
            name = p.name if isinstance(p, ast.FunctionDef) else p.targets[0].id
            changed.append(name)
    assert changed == ["bfp_SK2_SRC", "bfp__get_ext"], changed
    csrc = next(ast.literal_eval(n.value) for n in ct.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "bfp_SK2_SRC" for t in n.targets))
    assert "constexpr bool ESC_GLOBAL = COMP && CL == 1;" in csrc
    assert csrc.count("NW + (COMP && !ESC_GLOBAL ? NESC : 0)") == 4
    assert csrc.count("mbar_init(cbar, 32 * NESC)") == 2
    assert "cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);" in csrc
    assert csrc.index('if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread') < csrc.index("if constexpr (ESC_GLOBAL)")
    env = read_geometry()
    rows = []
    address_count = ownership_count = ring_count = 0
    for cfg in env["bfp_SK2_CONFIGS"]:
        name, layer, _, r, nt, nwr, wsk, upw, _, cl = cfg
        if layer == "router" or cl != 1:
            continue
        for tokr in (8, 16, 32):
            if env["bfp_SK2_PICK"].get((layer, tokr)) != name and name != "pooler":
                continue
            _, depth = env["bfp__sk2_geom"](cfg, tokr)
            ne, kst, nw = min(depth, 4), wsk * upw * 64, nwr * wsk
            k = env["bfp_SK2_K"][layer]
            nst = k // kst
            # Dense, empty, boundary-only and irregular lists; duplicate K but
            # different rows follow the exact retained stable table sequence.
            full_k = np.repeat(np.arange(k, dtype=np.int32), r)
            cases = [np.array([], dtype=np.int32), full_k,
                     np.repeat(np.unique(np.clip(np.concatenate([np.arange(nst + 1) * kst + d for d in (-1, 0, 1)]), 0, k - 1)), r),
                     full_k[(np.arange(len(full_k)) * 37 % 101) < 11]]
            for kk in cases:
                idx = np.arange(len(kk))
                seen = []
                for ew in range(ne):
                    old = np.concatenate([idx[(kk >= st * kst) & (kk < (st + 1) * kst)]
                                          for st in range(nst) if st % ne == ew])
                    new = idx[((kk // kst) % ne) == ew]
                    assert np.array_equal(old, new)
                    seen.extend(new.tolist())
                assert sorted(seen) == idx.tolist()
                ownership_count += len(kk)
            # Independently materialize the 128B-swizzled shared TMA row:
            # logical unit/token/k maps from global token*K + unit*64+k.
            # Test every K coordinate, every physical M up to each TOKR,
            # including odd M and UV's even physical 2T rows.
            kk = np.arange(k, dtype=np.int32)
            stage = kk // kst
            local = kk % kst
            unit, kc = local >> 6, local & 63
            for m in range(1, tokr + 1):
                mb = m if tokr == 8 or m < tokr else tokr
                for token in range(m):
                    shared_addr = unit * mb * 128 + token * 128 + (((kc >> 3) ^ ((unit * mb + token) & 7)) << 4) + (kc & 7) * 2
                    # Undo shared swizzle using independent row-byte decoding.
                    row = shared_addr // 128
                    byte = shared_addr % 128
                    ku = ((byte // 16) ^ (row & 7)) * 8 + ((byte % 16) // 2)
                    decoded_unit, decoded_token = row // mb, row % mb
                    recovered = decoded_token * k + stage * kst + decoded_unit * 64 + ku
                    direct = token * k + kk
                    assert np.array_equal(recovered, direct)
                    assert np.all(direct < m * k)
                    address_count += k
            for seed in range(32):
                assert ring_model(nw, depth, nst + (4 if name == "pooler" else 0), seed)
                ring_count += 1
            rows.append({"name": name, "tokr": tokr, "ring_depth": depth, "escape_warps": ne, "regular_warps": nw})
    # Negative controls ensure the ring model observes missing/extra arrivals.
    assert any(not ring_model(4, 2, 12, s, -1) for s in range(32))
    assert not ring_model(4, 2, 12, 7, 1)
    print(json.dumps({"status": "pass", "parent_sha256": EXPECTED_PARENT,
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "changed_python_nodes": changed, "geometries": rows,
        "patch_entry_ownership_comparisons": ownership_count,
        "tma_global_address_comparisons": address_count,
        "ring_interleavings": ring_count, "negative_controls": 2,
        "gpu_compilation_or_correctness_verified": False,
        "limits": "CPU address/order/event models and AST isolation only; not GPU races, compiler resource attributes, numerical oracle or performance"}, indent=2))


if __name__ == "__main__":
    main()
