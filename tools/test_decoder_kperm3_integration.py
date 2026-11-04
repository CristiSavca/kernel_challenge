#!/usr/bin/env python3
"""Check the isolated decoder integration without importing CUDA/Torch code.

Structural equality bounds the change to the already GPU-checked MoE mechanism.
The byte proof executes the decoder's real packing helpers through the existing
NumPy adapter and independently recovers a complete W13 and W2 expert.
"""
import ast
import copy
import hashlib
from pathlib import Path

import numpy as np
import test_wgs_kperm3_cpu as proof

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/candidates/muse_decode_layer/80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c.py"
CANDIDATE = ROOT / "muse_decode_layer/candidates/native_wgs_kperm3_v12.py"
STANDALONE = ROOT / "experiments/candidates/muse_moe_w4a16/247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0.py"


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    a, b, s = [ast.parse(p.read_text()) for p in (PARENT, CANDIDATE, STANDALONE)]
    # Equality of every preexisting AST node except four precisely scoped edits.
    changed = {"MOE_CUDA_SRC", "_ma_ext", "_wgs_copy", "_ma_prepare"}
    added = {"_wgs_kperm_index", "_wgs_kperm_codes", "_wgs_kperm_rows"}
    def key(n):
        if isinstance(n, ast.FunctionDef):
            return n.name
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            return n.targets[0].id
        return None
    assert [dump(n) for n in a.body if key(n) not in changed] == [dump(n) for n in b.body if key(n) not in changed | added]
    for name in added:
        assert dump(proof.function(b, name)) == dump(proof.function(s, name))
    cuda = proof.literal(s, "CUDA_SRC").replace(
        "#define OKC_BIGM 32  // activates interleaved fat stream at M=32 (measured -0.45 us on Box A)",
        "#define OKC_BIGM 32",
    )
    assert proof.literal(b, "MOE_CUDA_SRC") == cuda
    ext = copy.deepcopy(proof.function(b, "_ma_ext"))
    for n in ast.walk(ext):
        if isinstance(n, ast.Constant) and n.value == "dl_okc_moe_kperm3_v12_":
            n.value = "dl_okc_moe_selmul_"
    assert dump(ext) == dump(proof.function(a, "_ma_ext"))
    prep = copy.deepcopy(proof.function(b, "_ma_prepare"))
    block = next(n for n in prep.body if isinstance(n, ast.If))
    assert ast.unparse(block.body[0]) == "wgs_rowsA = _wgs_kperm_rows(rowsA)"
    block.body.pop(0)
    renamed = 0
    for n in ast.walk(prep):
        if isinstance(n, ast.Name) and n.id == "wgs_rowsA":
            n.id = "rowsA"
            renamed += 1
    assert renamed == 1
    assert dump(prep) == dump(proof.function(a, "_ma_prepare"))
    packing = copy.deepcopy(proof.function(s, "_wgs_copy"))
    packing.args.args.pop()
    packing.args.defaults.pop()
    call = next(n for n in ast.walk(packing) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_pack_codes_nat")
    assert ast.unparse(call.args.pop()) == "positions"
    assert dump(packing) == dump(proof.function(b, "_wgs_copy"))
    assert proof.literal(b, "MOE_MULDEC") is True
    print("PASS exact checked CUDA, packing helper mapping, weight-only preparation, namespace, and all unchanged decoder nodes")

    ns = proof.packing_namespace(b)
    ns["_POS_PACK"] = ns["_POS_MUL"]
    actual = ns["_wgs_copy"]
    def decoder_copy(codes, scales, rows, positions):
        assert positions == ns["_POS_PACK"]
        return actual(codes, scales, rows)
    ns["_wgs_copy"] = decoder_copy
    natural = np.stack((np.arange(4096).reshape(-1, 8), 4096 + np.arange(4096).reshape(-1, 8)), 1).reshape(-1)
    rows_a = ns["_wgs_kperm_rows"](proof.Tensor(natural)).a
    row = np.arange(8192, dtype=np.uint32)[:, None]
    col = np.arange(2048, dtype=np.uint32)[None, :]
    codes = ((row * 37 + col * 11 + ((row ^ col) >> 3)) & 255).astype(np.uint8)
    scales = ((row * 13 + np.arange(256, dtype=np.uint32)[None, :] * 7) & 255).astype(np.uint8)
    proof.check_actual_bytes(ns, codes, scales, rows_a, ns["_POS_PACK"], "decoder W13 MULDEC")
    proof.check_actual_bytes(ns, codes, scales, np.arange(4096), ns["_POS_PACK"], "decoder W2 MULDEC")
    print("candidate_sha256", hashlib.sha256(CANDIDATE.read_bytes()).hexdigest())
    print("GPU numerical correctness and whole-decoder performance remain untested")


if __name__ == "__main__":
    main()
