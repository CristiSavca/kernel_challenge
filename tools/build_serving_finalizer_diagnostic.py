#!/usr/bin/env python3
"""Embed the serving finalizer in an otherwise exact official Marlin baseline.

This only creates a readable private-check candidate; it never uploads or runs it.
The candidate owns its global binding; no imported module is modified.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1"
BASE_COMMIT = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
HELPER = "python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py"
OLD_HELPER_SHA = "9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591"
LICENSE_SHA = "1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747"
PREFIX = """# okc: problem=muse_moe_w4a16
# Private diagnostic: stock Marlin with an isolated serving finalizer change.
# SGLang source at a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f, Apache-2.0.
# Upstream license: https://github.com/sgl-project/sglang/blob/main/LICENSE
# This candidate is for component validation; it is not our native incumbent.
"""
SEPARATOR = "\n\n# Readable serving helper. This definition belongs to this candidate module.\n"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    base = (ROOT / "muse_moe_w4a16/baseline.py").read_bytes()
    assert sha(base) == BASE_SHA
    helper = (ROOT / "serving/sglang" / HELPER).read_bytes()
    original = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:{HELPER}"], cwd=ROOT / "serving/sglang")
    assert sha(original) == OLD_HELPER_SHA and helper != original
    base_tree = ast.parse(base)
    helper_tree = ast.parse(helper)
    definitions = lambda tree: {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    assert not definitions(base_tree) & definitions(helper_tree)
    assert "fused_post_expert_norm_weighted_sum" in definitions(helper_tree)
    assert {"prepare", "kernel_fn"} <= definitions(base_tree)
    assert not {"prepare", "kernel_fn"} & definitions(helper_tree)
    license_text = (ROOT / "serving/sglang/LICENSE").read_bytes()
    assert sha(license_text) == LICENSE_SHA
    license_comment = "\n# SGLang's complete Apache-2.0 license follows.\n" + "\n".join(
        "# " + line if line else "#" for line in license_text.decode().splitlines()) + "\n"
    source = PREFIX.encode() + base + SEPARATOR.encode() + helper + license_comment.encode()
    compile(source, "serving_finalizer_v22.py", "exec")
    assert source[len(PREFIX):len(PREFIX) + len(base)] == base
    candidate = ROOT / "muse_moe_w4a16/candidates/serving_finalizer_v22.py"
    candidate.write_bytes(source)
    report = {
        "candidate": str(candidate.relative_to(ROOT)), "candidate_sha256": sha(source),
        "baseline_sha256": BASE_SHA, "serving_base": BASE_COMMIT,
        "original_helper_sha256": OLD_HELPER_SHA, "helper_sha256": sha(helper),
        "helper_path": "serving/sglang/" + HELPER,
        "complete_upstream_license_sha256": LICENSE_SHA,
        "all_original_baseline_bytes_preserved": True,
        "helper_adds_no_prepare_or_kernel_fn": True,
        "new_helper_definition_in_candidate_module_only": True,
        "source_bytes": len(source), "gpu_validated": False,
        "limits": "Compilation here is Python syntax only. Requires separate source review and official GPU correctness/performance; broader helper API and full serving remain unvalidated.",
    }
    out = ROOT / "experiments/iterations/20261104-serving-finalizer-diagnostic-build.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
