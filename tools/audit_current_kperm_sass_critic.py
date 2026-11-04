"""Offline audit of one retained, output-capped CPU compilation; never executes CUDA."""
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/modal/20261104T155445Z-1ed7603c"
OUTPUT = ROOT / "experiments/iterations/20261104-current-kperm-sass-result-critic.json"
SOURCE = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"
NATIVE = "85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c"
SYMBOLS = [f"_ZN3okc9moe_fusedILb{b}EEEvNS_6ParamsE" for b in (1, 0)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def elf_sections(data):
    assert data[:6] == b"\x7fELF\x02\x01"
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, h[6] + i * h[11]) for i in range(h[12])]
    strings = headers[h[13]]
    names = data[strings[4]:strings[4] + strings[5]]
    return {names[s[0]:].split(b"\0", 1)[0].decode(): {"offset": s[4], "bytes": s[5]} for s in headers}


def instructions(text):
    rows, locations, fresh = [], [], False
    for line in text.splitlines():
        if "//## File" in line:
            if not fresh:
                locations = []
            fresh = True
            locations.extend((m[1], int(m[2])) for m in re.finditer(
                r'(?:File |inlined at )"([^"]+)", line (\d+)', line))
            continue
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(?:@!?[PU]+\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\b(.*);", line)
        if m:
            rows.append({"pc": int(m[1], 16), "opcode": m[2], "operands": m[3].strip(),
                         "cuda_lines": sorted({n for f, n in locations if f.endswith("/cuda.cu")})})
            fresh = False
    return rows


def run():
    result = json.loads((RUN / "result.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    request = json.loads((RUN / "request.json").read_text())
    assert result["status"] == "compile_failed" and manifest["status"] == "failed"
    assert manifest["sandbox_terminated"] is True and manifest["gpu"] is None
    assert result["device_execution"] is False and result["source_sha256"] == SOURCE
    assert manifest["resource_kind"] == "cpu_4cpu_32gib"
    assert sha((RUN / "worker.py").read_bytes()) == manifest["worker_sha256"]
    assert sha((RUN / "credit_guard.py").read_bytes()) == manifest["credit_guard_sha256"]
    for name, expected in request["file_sha256"].items():
        assert sha((RUN / name).read_bytes()) == expected
    bundle = result["artifact_bundle"]
    data = (RUN / "compile-artifacts.tar.gz").read_bytes()
    assert len(data) == bundle["bytes"] and sha(data) == bundle["sha256"]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(bundle["files"]) and {m.name for m in members} == set(bundle["files"])
        for member in members:
            expected = bundle["files"][member.name]
            assert member.isfile() and member.size == expected["bytes"]
            value = archive.extractfile(member).read()
            assert sha(value) == expected["sha256"] and value == (RUN / ("compile-" + member.name)).read_bytes()
    baseline = (RUN / "baseline.py").read_text()
    strings = {t.id: ast.literal_eval(n.value) for n in ast.parse(baseline).body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name) and t.id in ("CUDA_SRC", "CPP_SRC")}
    assert sha(baseline.encode()) == SOURCE and sha(strings["CUDA_SRC"].encode()) == NATIVE
    generated = (RUN / "compile-cuda.cu").read_text()
    assert generated == "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n" + strings["CUDA_SRC"]
    assert strings["CPP_SRC"] in (RUN / "compile-main.cpp").read_text()
    child = result["child"]
    assert child == json.loads((RUN / "compile-child.json").read_text())
    assert child["status"] == "compile_child_complete" and child["cuda_initialized"] is False
    assert (child["capability_calls"], child["load_inline_calls"], child["ninja_build_calls"], child["pdl_values"]) == (2, 1, 1, [0])
    ninja = (RUN / "compile-build.ninja").read_text()
    flags = next(l.removeprefix("cuda_cflags = ") for l in ninja.splitlines() if l.startswith("cuda_cflags = "))
    for flag in child["cuda_flags"]:
        assert flag in flags
    assert flags.count("-gencode=") == 1 and "-gencode=arch=compute_90a,code=sm_90a" in flags
    assert all(x in flags for x in ("-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_WGS=1", "-DOKC_PRMTSC=1"))
    assert result["commands"]["compile"]["returncode"] == 0
    dump = result["commands"]["sass.txt"]
    assert dump["output_exceeded"] is True and dump["returncode"] == -9 and dump["timed_out"] is False
    sass = (RUN / "compile-sass.txt").read_text()
    assert len(sass.encode()) == 33554432 and sass.startswith("\t.target\tsm_90a")
    cubin = (RUN / "compile-owned.cubin").read_bytes()
    binary = (RUN / "compile-owned.so").read_bytes()
    embedded_offset = binary.find(cubin)
    assert embedded_offset >= 0
    sections = elf_sections(cubin)
    starts = [sass.index("\t.section\t.text." + symbol) for symbol in SYMBOLS]
    texts = [sass[starts[0]:starts[1]], sass[starts[1]:]]
    rows = [instructions(text) for text in texts]
    size = sections[".text." + SYMBOLS[0]]["bytes"]
    assert [r["pc"] for r in rows[0]] == list(range(0, size, 16))
    assert ".L_x_1871:" in texts[0] and ".L_x_1873:" not in texts[1]
    assert rows[1][-1]["pc"] + 16 < sections[".text." + SYMBOLS[1]]["bytes"]
    active = rows[0]
    dq = [r for r in active if set(r["cuda_lines"]) & {407, 408, 409, 410}
          and r["opcode"] in {"LOP3.LUT", "SHF.R.W.U32", "IMAD"}]
    dq_counts = dict(Counter(r["opcode"] for r in dq))
    assert dq_counts == {"SHF.R.W.U32": 1024, "LOP3.LUT": 2560, "IMAD": 512}
    scale = [r for r in active if 2790 in r["cuda_lines"] and r["opcode"] in {"PRMT", "IMAD"}]
    assert dict(Counter(r["opcode"] for r in scale)) == {"PRMT": 128, "IMAD": 128}
    mul = [r for r in active if r["opcode"] == "HMUL2.BF16_V2"]
    broadcast = Counter("H0_H0" if "H0_H0" in r["operands"] else "H1_H1" if "H1_H1" in r["operands"] else "other" for r in mul)
    assert broadcast == {"H0_H0": 1024, "H1_H1": 1024}
    assert not any(set(r["cuda_lines"]) & set(range(414, 433)) for r in active)
    deps = Counter(r["operands"] for r in active if r["opcode"] == "WARPGROUP.DEPBAR.LE")
    assert deps == {"gsb0, 0x1": 256, "gsb0, 0x0": 16}
    log = (RUN / "compile-compile.stdout.log").read_text() + (RUN / "compile-compile.stderr.log").read_text()
    resources = (RUN / "compile-resources.txt").read_text()
    for symbol in SYMBOLS:
        assert re.search(re.escape(symbol) + r":\s+REG:96 STACK:0 SHARED:1024 LOCAL:0", resources)
    warnings = re.findall(r"C751\d", log)
    billing = json.loads((RUN / "billing-after.json").read_text())["billing"]
    return {
        "date": "2026-11-04", "reviewer": "attention_paper_reset",
        "verdict": "COMPILE_AND_BINARY_VALID_BIG1_SASS_COMPLETE_BIG0_TRUNCATED_DIAGNOSTIC_REMAINS_FAILED",
        "run": str(RUN.relative_to(ROOT)), "source_sha256": SOURCE, "native_source_sha256": NATIVE,
        "artifact_hashes_verified": bundle["files"], "archive_sha256": bundle["sha256"],
        "reviewer_script_sha256": sha(Path(__file__).read_bytes()),
        "manifest_and_result_sha256": {n: sha((RUN / n).read_bytes()) for n in ("manifest.json", "result.json")},
        "binary_binding": {"exact_cubin_occurs_in_owned_so_at_byte": embedded_offset, "generated_cuda_prefix_lines": 3,
                           "cuda_flags": child["cuda_flags"], "actual_compiler_versions": result["versions"],
                           "one_build": True, "cuda_initialized": False, "device_execution": False},
        "execution": {"compile_seconds": result["commands"]["compile"]["elapsed_seconds"],
                      "dump_failure": dump, "sandbox_terminated": True, "no_retry_or_recompile": True},
        "resources_both_BIG": {"registers": 96, "stack_bytes": 0, "local_bytes": 0,
                               "static_shared_bytes": 1024, "ptxas_spill_load_store_bytes": 0,
                               "static_shared_is_not_dynamic_launch_allocation": True},
        "coverage": [{"BIG": b, "section_bytes": sections[".text." + symbol]["bytes"],
                      "retained_instruction_count": len(items), "last_retained_pc": hex(items[-1]["pc"]),
                      "complete": b == 1} for b, symbol, items in zip((1, 0), SYMBOLS, rows)],
        "BIG1_static_inventory": {"scope": "One complete text section, including compiled NTT/phase/FIRST alternatives; not dynamic counts.",
            "dq4_integer_ops": dq_counts, "decoded_word_instances": 512,
            "per_word_integer_ops": {"LOP3": 5, "SHF": 2, "IMAD": 1},
            "per_word_BF16x2_multiplies": 4, "HMUL2_BF16_total": len(mul), "broadcast_operands": dict(broadcast),
            "scale_pair_ops": dict(Counter(r["opcode"] for r in scale)),
            "scale_pair_reconstruction_instances": 128, "static_stage_bodies": 32,
            "HGMMA_opcodes": dict(Counter(r["opcode"] for r in active if r["opcode"].startswith("HGMMA."))),
            "WARPGROUP_DEPBAR_limits": dict(deps),
            "DQ7_inactive": "Build defines both MULDEC and DQ7; preprocessor selects MULDEC and no annotations from alternate dq4 lines414..432 appear.",
            "C751x_in_complete_compile_logs": warnings},
        "examples": {hex(pc): {"opcode": row["opcode"], "operands": row["operands"], "cuda_lines": row["cuda_lines"]}
                     for pc in (0x6ef0, 0x6f00, 0x6f10, 0x6f50, 0x6f70, 0x6fa0, 0x6fb0, 0x6fc0, 0x6fe0, 0x6ff0, 0x7020, 0x7030, 0x7050, 0x7070, 0x70d0)
                     for row in active if row["pc"] == pc},
        "interpretation": [
            "Expected broadcasts are already folded into H0_H0/H1_H1 HMUL2 operands; no extra broadcast instruction-removal opportunity found.",
            "Scale reconstruction is the expected PRMT+IMAD pair; r3 bit decode uses expected mask/IMAD0x1084/mask. No concrete duplicate computation identified.",
            "Line annotations are not exact instruction semantics: scheduled HGMMA/address operations can carry neighboring source lines, so counts require both mapped source and opcode. Some H0 operands carry a bmul2h source annotation.",
            "Actual wait1 group instructions are present and C751x warnings absent for this build; neither establishes achieved asynchronous overlap or official evaluator behavior.",
            "Math-pressure counters are aggregate, from a separate run. Static counts do not measure issue cycles, hot-path frequency, critical-path time or predicted speedup.",
            "No claim of complete BIG0 SASS, official binary identity, performance improvement, candidate approval or retry authorization."
        ],
        "latest_saved_billing_observation": {"metered_cost": billing["metered_cost"],
            "ephemeral_apps": billing["metered_cost_breakdown"]["ephemeral_apps"],
            "preserve_highest_observed_usd": "4.16434237", "provider_call_by_reviewer": False}
    }


if __name__ == "__main__":
    report = run()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verdict": report["verdict"], "output": str(OUTPUT.relative_to(ROOT)),
                      "sha256": sha(OUTPUT.read_bytes()), "coverage": report["coverage"]}, indent=2))
