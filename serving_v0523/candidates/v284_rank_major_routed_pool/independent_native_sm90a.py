"""Check emitted FP32 operations and data dependencies in donor and reference SASS."""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

scratch = Path(sys.argv[1])
output = Path(sys.argv[2])
insn = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?P\d+\s+)?([A-Z][.A-Z0-9_]+)\s+([^;]*?)\s*;")


def sha(file):
    return hashlib.sha256((scratch / file).read_bytes()).hexdigest()


def image(name):
    text = (scratch / (name + ".sass")).read_text()
    code = [(int(pc, 16), op, [s.strip().removesuffix(".reuse") for s in operands.split(",")])
            for pc, op, operands in insn.findall(text)]
    counts = collections.Counter(op for _, op, _ in code)
    shuffles = [(i, a) for i, (_, op, a) in enumerate(code) if op == "SHFL.DOWN"]
    assert [int(a[3], 16) for _, a in shuffles] == [16, 8, 4, 2, 1]
    last_add = None
    for i, a in shuffles:
        if last_add is not None:
            assert last_add == a[2]
        last_add = next(b[0] for _, op, b in code[i+1:i+7]
                        if op == "FADD" and set(b[1:]) == set(a[1:3]))
    squares = [a for _, op, a in code[:shuffles[0][0]]
               if op == "FFMA" and a[1] == a[2]]
    assert len(squares) >= 4 and len(squares) % 4 == 0
    for i, args in enumerate(squares):
        if i % 4:
            assert args[3] == squares[i-1][0]
    rsq = next(i for i, (_, op, a) in enumerate(code) if op == "MUFU.RSQ" and a[1] != "-QNAN")
    rsq_window = code[rsq-7:rsq+5]
    assert any(op == "FSETP.GEU.AND" and "1.175494350822287508e-38" in a for _, op, a in rsq_window)
    assert any(op == "FMUL" and a[2] == "16777216" for _, op, a in code[rsq-3:rsq])
    assert any(op == "FMUL" and a[2] == "4096" for _, op, a in code[rsq+1:rsq+4])
    assert counts["FCHK"] >= 1
    assert all(counts[o] for o in ("FFMA.RZ", "FFMA.RM", "FFMA.RP"))

    # Weighted branches occur first in both binaries. Track both rounded FP32
    # multiplies, terminating at the predicate-free BF16 packed conversion.
    rstd = code[rsq][2][0]
    products = {}
    weighted = []
    for pc, op, a in code[rsq+5:]:
        if op.startswith("F2FP.BF16.F32.PACK_AB"):
            for reg in a[1:]:
                if reg in products and len(products[reg]) == 2:
                    weighted.append([*products[reg], f"{pc:x}"])
        if op == "FMUL":
            if rstd in a[1:]:
                product = [f"{pc:x}"]
            else:
                prior = next((products[arg] for arg in a[1:] if len(products.get(arg, [])) == 1), [])
                product = prior + [f"{pc:x}"] if prior else []
            products[a[0]] = product
        elif a[0].startswith("R") and op not in ("BRA", "CALL", "EXIT", "RET"):
            products.pop(a[0], None)
        if len(weighted) >= 4:
            break
    assert len(weighted) == 4 and len({p[0] for p in weighted}) == 4
    assert len({p[1] for p in weighted}) == 4
    log = (scratch / (name + ".log")).read_text()
    assert "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" in log
    return {"source_sha256": sha(name + ".cu") if (scratch / (name + ".cu")).exists() else None,
            "sass_sha256": sha(name + ".sass"), "ptxas": re.search(r"Used \d+ registers,.*", log)[0],
            "stack_spill_bytes": 0, "static_instructions_with_operands": len(code),
            "static_mufu_rcp_sites": counts["MUFU.RCP"],
            "dependent_square_FFMA_groups_of_four_before_shuffles": len(squares) // 4,
            "shuffle_then_FADD_offsets": [int(a[3], 16) for _, a in shuffles],
            "division_exception_test_sites": counts["FCHK"],
            "rsqrt": "non-FTZ predicate at 2^-126; subnormal x2^24, MUFU.RSQ, x2^12",
            "weighted_output_FMUL_FMUL_BF16_conversion_PCs_hex": weighted}


receipt = {"target": "sm_90a", "nvcc": "12.9.86", "device_execution": False,
           "division_comparison": "static instructions and reciprocal sites; no timing",
           "upstream": json.loads((scratch / "pins.json").read_text()),
           "files": {f: sha(f) for f in ("ranked_pool_norm.cuh", "donor.cu", "golden.cu")},
           "compiled": {name: image(name) for name in ("golden", "donor")}}
if (scratch / "division.sass").exists():
    receipt["before_shift_mask"] = image("division")
    receipt["before_shift_mask"]["header_sha256"] = sha("division.cuh")
    assert receipt["before_shift_mask"]["static_mufu_rcp_sites"] == 6
assert receipt["compiled"]["donor"]["static_mufu_rcp_sites"] == 2
if (scratch / "cubin.sha").exists():
    receipt["cubin_sha256"] = {path: hash_ for hash_, path in
                               (row.split() for row in (scratch / "cubin.sha").read_text().splitlines())}
output.write_text(json.dumps(receipt, indent=2) + "\n")
