#!/usr/bin/env python3
"""Source-bound Lean proof of the MULDEC u32 fragment (not a PTX interpreter)."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_SHA = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"
ANCHOR = ROOT / "experiments/candidates/muse_moe_w4a16" / (ANCHOR_SHA + ".py")
TOOLCHAIN = ROOT / ".toolchains/lean-4.35.0-darwin_aarch64"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
BASE_THEOREMS = [f"OKC.Dq4.decode{i}_correct" for i in range(4)] + [
    "OKC.Dq4.positions_bijective", "OKC.Dq4.masked3", "OKC.Dq4.multiply3_codes"]
CANDIDATE_THEOREMS = [f"OKC.Candidate.decode{i}_correct" for i in range(4)]
LIMITATIONS = [
    "Source-level unsigned32 arithmetic fragment only; no PTX/SASS parsing or compiler proof.",
    "The Python AST/source-to-Lean translation and independent bit-placement specification are reviewed trusted links.",
    "Packing AST identity is checked; tensor layout, coordinate permutations and Python execution are not proved.",
    "Assumes ordinary uint32 semantics, builtin funnel shifts, immutable packing maps and distinct output registers; surrounding Python/C++ environment is not certified.",
    "Does not certify dispatch, the non-MULDEC tiny path, BF16/FP8 arithmetic, scales, MMA, memory safety, synchronization or termination.",
    "No performance or complete-kernel correctness claim; official GPU checks remain required.",
]


class VerificationError(RuntimeError):
    pass


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def one_assignment(tree: ast.Module, name: str) -> ast.Assign:
    writes = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Store) and n.id == name]
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id == name]
    if len(writes) != 1 or len(nodes) != 1:
        raise VerificationError(f"Expected one top-level assignment to {name}")
    return nodes[0]


def canonical(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def packing_nodes(tree: ast.Module) -> dict[str, str]:
    out = {}
    for name in ("_stored_pos_mul", "_pack_codes", "_pack_codes_nat"):
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == name]
        if len(nodes) != 1 or nodes[0] not in tree.body:
            raise VerificationError(f"Expected one top-level function {name}")
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name
               for n in ast.walk(tree)):
            raise VerificationError(f"Reassigned packing function {name}")
        out[name] = canonical(nodes[0])
    for name in ("_POS_MUL", "_REG_HJ"):
        out[name] = canonical(one_assignment(tree, name))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(node.value, ast.Name) and node.value.id in ("_POS_MUL", "_REG_HJ")):
            raise VerificationError("Direct packing-map mutation is outside the proof binding")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("_POS_MUL", "_REG_HJ")):
            raise VerificationError("Packing-map method calls are outside the proof binding")
    return out


def extract(source: bytes, anchor: bytes | None = None) -> dict:
    """Never executes the candidate. Unknown syntax fails closed for this fragment."""
    if len(source) > 2 * 1024 * 1024:
        raise VerificationError("Candidate exceeds 2 MiB source limit")
    anchor = ANCHOR.read_bytes() if anchor is None else anchor
    if sha(anchor) != ANCHOR_SHA:
        raise VerificationError("Reviewed packing anchor digest mismatch")
    try:
        tree, reference = ast.parse(source), ast.parse(anchor)
    except (SyntaxError, ValueError) as exc:
        raise VerificationError(f"Python source parsing failed: {exc}") from exc
    packing, expected = packing_nodes(tree), packing_nodes(reference)
    if packing != expected:
        changed = [k for k in expected if packing.get(k) != expected[k]]
        raise VerificationError("Packing specification binding changed: " + ", ".join(changed))
    assignment = one_assignment(tree, "CUDA_SRC")
    if not isinstance(assignment.value, ast.Constant) or type(assignment.value.value) is not str:
        raise VerificationError("CUDA_SRC must be a literal string")
    cuda = assignment.value.value
    # C/C++ comments are whitespace, not deletion (r/**/0 must not become r0).
    # C++ translation phase 2 precedes comment removal. A trailing backslash
    # can otherwise hide a real statement inside a continued // comment.
    spliced = re.sub(r"\\\r?\n", "", cuda)
    stripped = re.sub(r"/\*.*?\*/|//[^\n]*", " ", spliced, flags=re.S)
    if re.search(r"^\s*#\s*define\s+(?:uint32_t|__funnelshift_r|q|M|r[0-3])\b", stripped, re.M):
        raise VerificationError("Macro override of modeled arithmetic identifier")
    signature = (r"__device__\s+__forceinline__\s+void\s+dq4\s*\(\s*uint32_t\s+q\s*,"
                 r"\s*uint32_t\s*&\s*r0\s*,\s*uint32_t\s*&\s*r1\s*,"
                 r"\s*uint32_t\s*&\s*r2\s*,\s*uint32_t\s*&\s*r3\s*\)\s*\{"
                 r"\s*#if\s+OKC_MULDEC\s*\n(?P<body>.*?)\n\s*#else\b")
    matches = list(re.finditer(signature, stripped, re.S))
    if len(matches) != 1 or len(re.findall(r"\bvoid\s+dq4\s*\(", stripped)) != 1:
        raise VerificationError("Expected one recognized dq4 unsigned32 MULDEC branch")
    body = matches[0]["body"]
    num = r"(0[xX][0-9a-fA-F]+|[0-9]+)[uU]?"
    grammar = (r"\s*constexpr\s+uint32_t\s+M\s*=\s*" + num + r"\s*;"
               r"\s*r0\s*=\s*q\s*&\s*M\s*;"
               r"\s*r1\s*=\s*__funnelshift_r\s*\(\s*q\s*,\s*q\s*,\s*" + num + r"\s*\)\s*&\s*M\s*;"
               r"\s*r2\s*=\s*__funnelshift_r\s*\(\s*q\s*,\s*q\s*,\s*" + num + r"\s*\)\s*&\s*M\s*;"
               r"\s*r3\s*=\s*\(\s*\(\s*q\s*&\s*" + num + r"\s*\)\s*\*\s*" + num + r"\s*\)\s*&\s*M\s*;\s*")
    parsed = re.fullmatch(grammar, body)
    if not parsed:
        raise VerificationError("Unsupported dq4 fragment syntax; no certificate issued")
    if any(len(s) > 1 and s.startswith("0") and not s.lower().startswith("0x")
           for s in parsed.groups()):
        raise VerificationError("Leading-zero/octal literals are outside the modeled grammar")
    constants = dict(zip(("mask", "rotate1", "rotate2", "multiply_mask", "multiplier"),
                         (int(s, 16 if s.lower().startswith("0x") else 10) for s in parsed.groups())))
    if any(not 0 <= v < 2**32 for v in constants.values()):
        raise VerificationError("Constants must be uint32")
    if any(constants[k] >= 32 for k in ("rotate1", "rotate2")):
        raise VerificationError("Only funnel shifts below 32 are modeled")
    return {"source_sha256": sha(source), "cuda_sha256": sha(cuda),
            "fragment_sha256": sha(body), "constants": constants,
            "packing_anchor_sha256": ANCHOR_SHA,
            "packing_ast_sha256": {k: sha(v) for k, v in packing.items()}}


def generated_lean(binding: dict) -> str:
    c = binding["constants"]
    text = "import OKC.Dq4\nnamespace OKC.Candidate\nopen OKC.Dq4\n"
    text += "set_option maxRecDepth 100000\nset_option maxHeartbeats 2000000\n"
    definitions = [f"q &&& {c['mask']}", f"q.rotateRight {c['rotate1']} &&& {c['mask']}",
                   f"q.rotateRight {c['rotate2']} &&& {c['mask']}",
                   f"((q &&& {c['multiply_mask']}) * {c['multiplier']}) &&& {c['mask']}"]
    for i, expr in enumerate(definitions):
        text += f"def actual{i} (q : Word) : Word := {expr}\n"
        text += f"theorem decode{i}_correct (q : Word) : actual{i} q = spec{i} q := by\n"
        text += f"  simpa only [actual{i}, OKC.Dq4.decode{i}, OKC.Dq4.mask] using (OKC.Dq4.decode{i}_correct q)\n"
        text += f"#print axioms decode{i}_correct\n"
    return text + "end OKC.Candidate\n"


def audit_axioms(output: str, required: list[str]) -> dict[str, list[str]]:
    records = {}
    pattern = r"'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)"
    for name, raw in re.findall(pattern, output, re.S):
        if name in records:
            raise VerificationError(f"Duplicate axiom audit: {name}")
        axioms = [a.strip() for a in raw.split(",") if a.strip()]
        if set(axioms) - ALLOWED_AXIOMS:
            raise VerificationError(f"Unapproved proof dependencies for {name}: {axioms}")
        records[name] = axioms
    if set(records) != set(required):
        raise VerificationError(f"Missing/unexpected theorem audits: {sorted(set(records) ^ set(required))}")
    return records


def run_lean(args: list[str], directory: Path, env: dict, timeout: int = 180) -> str:
    # Local files bound memory consumption; Lean receives its own memory limit.
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(args, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise VerificationError(f"Lean exceeded {timeout}s bound") from exc
        size = log.tell()
        log.seek(0)
        output = log.read(512 * 1024).decode(errors="replace")
        if size > 512 * 1024:
            raise VerificationError("Lean log exceeded 512 KiB")
        if code or re.search(r"\b(sorryAx|declaration uses 'sorry')\b", output):
            raise VerificationError(f"Lean rejected fragment (exit {code}):\n{output[-6000:]}")
        return output


def verify(source: Path) -> dict:
    start = time.monotonic()
    data = source.read_bytes()
    binding = extract(data)
    contract = json.loads((ROOT / "formal/proof-contract.json").read_text())
    proof_files = {}
    for relative, digest in contract["proof_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise VerificationError("Unsafe proof contract path")
        contents = (ROOT / "formal" / path).read_bytes()
        if sha(contents) != digest:
            raise VerificationError(f"Reviewed proof file changed: {relative}")
        proof_files[relative] = contents
    lean, lake = TOOLCHAIN / "bin/lean", TOOLCHAIN / "bin/lake"
    if not lean.is_file() or not lake.is_file():
        raise VerificationError("Pinned Lean toolchain missing; see formal/README.md")
    for binary in (lean, lake):
        if sha(binary.read_bytes()) != contract["toolchain_binaries_sha256"][binary.name]:
            raise VerificationError(f"Pinned toolchain binary changed: {binary.name}")
    env = {**os.environ, "PATH": str(TOOLCHAIN / "bin") + os.pathsep + os.environ.get("PATH", "")}
    # Ignore caller-selected Lean paths; no existing .olean can satisfy this build.
    for name in ("LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT", "LAKE_HOME"):
        env.pop(name, None)
    parent = ROOT / "formal/.lake/proof-runs"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-", dir=parent) as temporary:
        work = Path(temporary)
        for relative, contents in proof_files.items():
            path = work / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        generated = generated_lean(binding)
        (work / "Candidate.lean").write_text(generated)
        build_output = run_lean([str(lake), "build", "OKC"], work, env)
        base_axioms = audit_axioms(build_output, BASE_THEOREMS)
        candidate_output = run_lean([str(lake), "env", "lean", "-M4096", "Candidate.lean"], work, env)
        candidate_axioms = audit_axioms(candidate_output, CANDIDATE_THEOREMS)
    if source.read_bytes() != data:
        raise VerificationError("Source changed during proof")
    return {"schema": 1, "status": "proved_fragment", "official": False,
            "recorded_at": datetime.now(timezone.utc).isoformat(), "source_path": str(source.resolve()),
            **binding, "domain": "all 4294967296 uint32 words",
            "verifier_sha256": sha(Path(__file__).read_bytes()),
            "proof_contract_sha256": sha((ROOT / "formal/proof-contract.json").read_bytes()),
            "proof_files_sha256": contract["proof_files"], "toolchain": contract["toolchain"],
            "toolchain_binaries_sha256": contract["toolchain_binaries_sha256"],
            "generated_lean_sha256": sha(generated), "axioms": {**base_axioms, **candidate_axioms},
            "build_log": build_output, "candidate_log": candidate_output,
            "elapsed_seconds": round(time.monotonic() - start, 3), "limitations": LIMITATIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.source.resolve())
    except (VerificationError, OSError, ValueError) as exc:
        result = {"schema": 1, "status": "rejected", "official": False, "error": str(exc)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "proved_fragment" else 1)


if __name__ == "__main__":
    main()
