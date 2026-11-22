"""Verify both exports, the unchanged V173 flags, and the normal medium-density dispatch."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
PARENT = HERE.parent / "v173_medium_expert_tma"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
URL = "https://openkernelchallenge.dev/api/v1/serving"
CU = "python/sglang/kernels/jit/csrc/muse_spark/"
ADAPTER = "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
MEDIUM = CU + "expert_medium_route_pair.cuh"
TMA = CU + "expert_tma_route_pair.cuh"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    live = json.loads(subprocess.check_output(["curl", "-fsSL", URL]))
    assert live["base"]["commit"] == BASE
    rules = live["rules"]
    full, delta, baseline = [p.read_bytes() for p in (HERE/"patch.diff", HERE/"delta.diff", PARENT/"patch.diff")]
    parent = json.loads((PARENT/"validation.json").read_text())
    assert sha(baseline) == parent["patch_sha256"]
    flags = parent["launch_flags"]
    denied = [p for patterns in live["launch_flags"]["denied"].values() for p in patterns]
    assert not any(fnmatch.fnmatchcase(flag, pattern) for flag in flags for pattern in denied)
    added = [line[1:] for line in full.decode().splitlines() if line.startswith("+") and not line.startswith("+++ ")]
    assert len(full) <= rules["max_patch_bytes"] and max(map(len, added)) <= rules["max_line_chars"]
    assert not any(word in line for line in added for word in rules["forbidden_words"])
    assert not any(line.startswith((b"old mode ", b"new mode ", b"Binary files ", b"GIT binary patch", b"rename ")) for line in full.splitlines())
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp)/"index"))

        def git(*args, data=None):
            return subprocess.check_output(["git", "-C", STAGE, *args], input=data, env=env)

        def apply(tree, patch, reverse=False):
            git("read-tree", tree)
            git("apply", "--cached", "--whitespace=error-all", *(["--reverse"] if reverse else []), "-", data=patch)
            return git("write-tree").decode().strip()

        parent_tree = apply(BASE, baseline)
        assert parent_tree == parent["tree"]
        tree = apply(BASE, full)
        assert tree == apply(parent_tree, delta)
        assert apply(tree, delta, True) == parent_tree
        assert apply(tree, full, True) == git("rev-parse", BASE+"^{tree}").decode().strip()
        changed = git("diff", "--name-only", parent_tree, tree).decode().splitlines()
        assert set(changed) == {CU+"expert_route_pair.cuh", TMA}
        paths = git("diff", "--name-only", BASE, tree).decode().splitlines()
        for row in git("diff", "--raw", BASE, tree).decode().splitlines():
            old, new = row.split()[:2]
            assert new in ("100644", "100755") and (old == ":" + new or (old == ":000000" and new == "100644"))
        for path in paths:
            assert path.startswith(rules["allowed_prefix"]) and not any(path.startswith(x) for x in rules["forbidden_prefixes"])
            assert path.endswith(tuple(rules["allowed_suffixes"]))
            data = git("show", tree+":"+path)
            assert data == (STAGE/path).read_bytes()
            if path.endswith(".py"):
                compile(data, path, "exec")
        a = ast.parse((STAGE/ADAPTER).read_bytes())
        funcs = {f.name: ast.unparse(f) for f in a.body if isinstance(f, ast.FunctionDef)}
        assert "MuseTmaRoutePair::run_medium" in funcs["_tma_route_pair_module"]
        assert "route[0].run_medium" in funcs["humming_expert_outputs"]
        src = (STAGE/TMA).read_text()
        assert "dev)(multirow_projection<Gate>, p, maps, stride)" in src
        assert "MuseMediumRoutePair::run_with<false>" in src
        assert "dev)(muse_medium_route_pair::projection<Gate>, p, stride)" in src
        compiler = json.loads((HERE/"sm90.json").read_text())
        hashes = {name:sha((STAGE/(CU+name)).read_bytes()) for name in compiler["build_inputs"]["new"]}
        assert hashes == compiler["build_inputs"]["new"]
        assert json.loads((HERE/"dispatch.json").read_text())["dispatch_cases"] == 108
        assert json.loads((HERE/"launch.json").read_text())["cases"] == 25600
        emu = json.loads((HERE/"independent_warp.json").read_text())
        assert emu["header_sha256"] == [hashes[n] for n in ("expert_route_pair.cuh","expert_tma_route_pair.cuh")]
        assert emu["medium_sha256"] == hashes["expert_medium_route_pair.cuh"]

    print(json.dumps(dict(candidate="V174", base=BASE, parent="V173", parent_tree=parent_tree, tree=tree,
                          patch_sha256=sha(full), bytes=len(full), delta_bytes=len(delta), delta_sha256=sha(delta),
                          parent_sha256=sha(baseline), paths=paths, changed_from_parent=changed, launch_flags=flags,
                          strict_apply_reverse_full_delta=True, official_export="okc serve diff",
                          official_rules=URL, checked_flag_patterns=len(denied), max_added_line_chars=max(map(len, added)),
                          physical_H200=False, submitted=False), indent=2))


if __name__ == "__main__":
    main()
