"""Rebuild the V171 export and CLI against the pinned tree and published serving rules."""
import argparse
import ast
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import msgspec


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
PARENT = HERE.parent / "v168_streamed_experts_sparse_return"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
FLAG = "--speculative-dflash-sp-prefill-context"
FIELD = FLAG[2:].replace("-", "_")
SRT = "python/sglang/srt/"
ORDER = SRT + "arg_groups/field_order.py"
SPEC = SRT + "arg_groups/fields/spec.py"
API = "https://openkernelchallenge.dev/api/v1/serving"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def field_order(data):
    return ast.literal_eval(next(node.value for node in ast.parse(data).body if isinstance(node, ast.Assign)))


def check_cli(files, temp):
    def module(name, filename):
        path = temp / (name.split(".")[-1] + ".py")
        path.write_bytes(files(filename))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value

    prefix = "sglang.srt.arg_groups"
    util = module(prefix + ".arg_utils", SRT + "arg_groups/arg_utils.py")
    module(prefix + ".choices", SRT + "arg_groups/choices.py")
    spec = module(prefix + ".fields.spec", SPEC).Spec
    parser = argparse.ArgumentParser()
    util.add_cli_args_from_dataclass(parser, spec, fields=[FIELD])
    assert getattr(spec(), FIELD) is False
    assert getattr(parser.parse_args([]), FIELD) is False
    assert getattr(parser.parse_args([FLAG]), FIELD) is True


def main():
    baseline, delta, full = [p.read_bytes() for p in (PARENT/"patch.diff", HERE/"delta.diff", HERE/"patch.diff")]
    parent = json.loads((PARENT/"validation.json").read_text())
    assert sha(baseline) == parent["patch_sha256"]
    live = json.loads(subprocess.check_output(["curl", "-fsSL", API]))
    assert live["base"]["commit"] == BASE
    rules = live["rules"]
    flags = {**parent["launch_flags"], FLAG: ""}
    denied = [item for patterns in live["launch_flags"]["denied"].values() for item in patterns]
    assert not any(fnmatch.fnmatchcase(flag, pattern) for flag in flags for pattern in denied)
    added = [line[1:] for line in full.decode().splitlines() if line.startswith("+") and not line.startswith("+++ ")]
    assert len(full) <= rules["max_patch_bytes"] and max(map(len, added)) <= rules["max_line_chars"]
    assert not any(word in line for line in added for word in rules["forbidden_words"])
    assert not any(line.startswith((b"old mode ", b"new mode ", b"GIT binary patch", b"Binary files ", b"rename "))
                   for line in full.splitlines())
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        env = dict(os.environ, GIT_INDEX_FILE=str(temp/"index"))

        def git(*args, data=None):
            return subprocess.check_output(["git", "-C", STAGE, *args], env=env, input=data)

        def apply(tree, patch, reverse=False):
            git("read-tree", tree)
            git("apply", "--cached", "--whitespace=error-all", *(["--reverse"] if reverse else []), "-", data=patch)
            return git("write-tree").decode().strip()

        parent_tree = apply(BASE, baseline)
        assert parent_tree == parent["tree"]
        tree = apply(BASE, full)
        assert tree == apply(parent_tree, delta)
        assert apply(tree, delta, reverse=True) == parent_tree
        assert apply(tree, full, reverse=True) == git("rev-parse", BASE+"^{tree}").decode().strip()
        raw = git("diff", "--raw", BASE, tree).decode().splitlines()
        assert all(line.split()[:2] in ([":100644", "100644"], [":100755", "100755"], [":000000", "100644"]) for line in raw)
        paths = git("diff", "--name-only", BASE, tree).decode().splitlines()
        changed = git("diff", "--name-only", parent_tree, tree).decode().splitlines()

        def contents(path):
            return git("show", tree+":"+path)

        for path in paths:
            assert path.startswith(rules["allowed_prefix"]) and not any(path.startswith(p) for p in rules["forbidden_prefixes"])
            assert path.endswith(tuple(rules["allowed_suffixes"]))
            assert contents(path) == (STAGE/path).read_bytes()
            if path.endswith(".py"):
                compile(contents(path), path, "exec")
        assert set(changed) == {ORDER, SPEC, *(SRT+p for p in (
            "layers/dflash_context_projection.py", "layers/layernorm_sp.py", "layers/logits_processor.py",
            "model_executor/runner/prefill_cuda_graph_runner.py", "models/muse_spark.py", "speculative/dflash_worker_v2.py"))}
        old_fields = field_order(git("show", parent_tree+":"+ORDER))
        assert tuple(field_order(contents(ORDER))) == (*old_fields, FIELD)
        check_cli(contents, temp)
        hashes = {path: sha(contents(SRT + path)) for path in json.loads((HERE/"independent_worker.json").read_text())["sources"]}
        assert hashes == json.loads((HERE/"independent_worker.json").read_text())["sources"]
    print(json.dumps(dict(candidate="V171", base=BASE, parent="V168", parent_tree=parent_tree, tree=tree,
                          parent_sha256=sha(baseline), patch_sha256=sha(full), delta_sha256=sha(delta),
                          bytes=len(full), delta_bytes=len(delta), official_export="okc serve diff", official_rules=API,
                          paths=paths, changed_from_parent=changed, launch_flags=flags, forbidden_flag_patterns=len(denied),
                          max_added_line_chars=max(map(len, added)), strict_apply_reverse_full_delta=True,
                          parent_positional_fields_preserved=len(old_fields), cli_msgspec=msgspec.__version__,
                          physical_H200=False, submitted=False), indent=2))


if __name__ == "__main__":
    main()
