"""Detect newly unbound Python globals, including V88's real graph-warmup failure.

Run with Python 3.14. Optional arguments name more candidate versions, such as v206.
Only the temporary Git index and object directory are written.
The analysis checks lexical bindings. Conditional imports and CUDA execution are
separate startup checks.
"""

import builtins
import hashlib
import json
import os
import subprocess
import symtable
import sys
import tempfile
from functools import lru_cache
from pathlib import Path


CANDIDATES = Path(__file__).resolve().parent.parent
STOCK = CANDIDATES.parent / "sglang"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
MODULE_NAMES = set(dir(builtins)) | {
    "__file__", "__name__", "__package__", "__spec__", "__loader__", "__builtins__",
    "__annotations__", "__cached__", "__path__", "__class__",
}


def analyze(source, path):
    module = symtable.symtable(source, path, "exec")
    defined = MODULE_NAMES | {
        symbol.get_name() for symbol in module.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    missing = set()
    scopes = 0
    reads = 0
    pending = [(module, "module")]
    while pending:
        table, scope = pending.pop()
        scopes += 1
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                reads += 1
                if symbol.get_name() not in defined:
                    missing.add((scope, symbol.get_name()))
        pending.extend((child, f"{scope}.{child.get_name()}") for child in table.get_children())
    return missing, scopes, reads


def check(scratch, additional):
    objects = scratch / "objects"
    objects.mkdir()
    env = {
        **os.environ,
        "GIT_INDEX_FILE": str(scratch / "index"),
        "GIT_OBJECT_DIRECTORY": str(objects),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(STOCK / ".git/objects"),
    }

    def git(*args):
        return subprocess.check_output(["git", "-C", STOCK, *args], env=env, text=True)

    tracked = set(git("ls-tree", "-r", "--name-only", PIN).splitlines())

    @lru_cache(maxsize=None)
    def native_missing(path):
        if path not in tracked:
            return set()
        missing, _, _ = analyze(git("show", f"{PIN}:{path}"), path)
        return missing

    result = {}
    require_manifest = ("v204", "v205", *additional)
    for label in ("v88", "v88a", *require_manifest):
        folder, = CANDIDATES.glob(label + "_*")
        patch = folder / "patch.diff"
        git("read-tree", PIN)
        git("apply", "--cached", str(patch))
        tree = git("write-tree").strip()
        paths = [
            path for path in git("diff", "--cached", "--name-only", PIN).splitlines()
            if path.endswith(".py")
        ]
        issues = {}
        scopes = 0
        reads = 0
        for path in paths:
            missing, count, references = analyze(git("show", ":" + path), path)
            scopes += count
            reads += references
            newly_missing = missing - native_missing(path)
            if newly_missing:
                issues[path] = sorted(newly_missing)
        result[label] = {
            "tree": tree,
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "python_files": len(paths),
            "lexical_scopes": scopes,
            "global_references_in_scopes": reads,
            "unbound_relative_to_pin": issues,
        }
        if label in require_manifest:
            assert tree == json.loads((folder / "validation.json").read_bytes())["tree"]

    assert result["v88"]["unbound_relative_to_pin"] == {
        "python/sglang/srt/models/dflash.py": [("module.candidate_topk", "is_cuda")]
    }
    assert all(not result[label]["unbound_relative_to_pin"] for label in ("v88a", *require_manifest))
    return result


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="okc-names-") as directory:
        print(json.dumps(check(Path(directory), sys.argv[1:]), indent=2))

