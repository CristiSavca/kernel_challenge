"""Reconstruct both donor orders in a private index and audit V312's public export."""

import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
SERVING = HERE.parents[1]
CANDIDATES = HERE.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def donor(n):
    return next(CANDIDATES.glob(f"v{n}_*"))


def read_json(path):
    return json.loads(path.read_bytes())


def verify(export_stage=None):
    contract = read_json(donor(309) / "public_contract.json")
    rules = contract["rules"]
    pin = contract["base"]["commit"]
    score_submission = (SERVING / "runs/v204_nov19_scored.submit.txt").read_text()
    scored = json.loads(score_submission.splitlines()[0])
    assert "ad275bf67d9a4a0a" in score_submission and scored["passed_check"] == "7fcdf9abf39b486a"
    flags = read_json(HERE / "flags.json")
    cli = [shlex.split(s) for s in (HERE / "flags").read_text().splitlines()]
    assert all(len(parts) in (1, 2) for parts in cli)
    assert {s[0]: " ".join(s[1:]) for s in cli} == flags == scored["flags"] and len(cli) == 6
    denied = [p for patterns in contract["launch_flags"]["denied"].values() for p in patterns]
    assert all(not fnmatch.fnmatchcase(flag, pattern) for flag in flags for pattern in denied)
    patch = (HERE / "patch.diff").read_bytes()
    delta = (HERE / "delta.diff").read_bytes()
    assert delta == (donor(311) / "delta.diff").read_bytes()
    sources = {}
    manifests = {}
    trees = {}
    repo = SERVING / "sglang"
    git_dir = subprocess.check_output(["git", "-C", repo, "rev-parse", "--absolute-git-dir"], text=True).strip()
    with tempfile.TemporaryDirectory() as directory:
        obj = Path(directory) / "objects"
        obj.mkdir()
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"),
                   GIT_OBJECT_DIRECTORY=str(obj), GIT_ALTERNATE_OBJECT_DIRECTORIES=str(Path(git_dir) / "objects"))
        def git(*args, data=None):
            return subprocess.check_output(["git", "-C", repo, *args], input=data, env=env)
        def apply(tree, change, reverse=False):
            git("read-tree", tree)
            git("apply", "--cached", "--whitespace=error-all", *(["-R"] if reverse else []), "-", data=change)
            return git("write-tree").decode().strip()

        baseline = git("rev-parse", pin + "^{tree}").decode().strip()
        for n in (204, 308, 309, 310, 311):
            manifest = read_json(donor(n) / "validation.json")
            data = (donor(n) / "patch.diff").read_bytes()
            tree = apply(pin, data)
            assert tree == manifest["tree"] and manifest["base"] == pin and sha(data) == manifest["patch_sha256"]
            assert manifest["launch_flags"] == flags
            trees[n], manifests[n] = tree, manifest["patch_sha256"]
        assert manifests[204] == scored["sha256"]
        tree = apply(pin, patch)
        assert apply(trees[310], delta) == tree and apply(tree, delta, reverse=True) == trees[310]
        assert apply(tree, patch, reverse=True) == baseline
        assert apply(trees[204], delta) == trees[311]
        order = trees[311]
        prompt = trees[204]
        for n in (308,309,310):
            step = (donor(n) / "delta.diff").read_bytes()
            assert sha(step) == read_json(donor(n) / "validation.json")["delta_sha256"]
            next_prompt = apply(prompt, step)
            assert next_prompt == trees[n] and apply(next_prompt, step, reverse=True) == prompt
            prompt = next_prompt
            order = apply(order, step)
        assert order == tree
        for n in (310,309,308):
            order = apply(order, (donor(n) / "delta.diff").read_bytes(), reverse=True)
        assert order == trees[311] and apply(order,delta,reverse=True)==trees[204]
        assert git("diff","--no-renames","-U1",pin,tree) == patch
        paths = git("diff","--name-only",pin,tree).decode().splitlines()
        changed_from_prompt = git("diff","--name-only",trees[310], tree).decode().splitlines()
        assert changed_from_prompt == ["python/sglang/kernels/ops/attention/paged_verify_attention.py"]
        lines = patch.decode().splitlines()
        added = [s[1:] for s in lines if s.startswith("+") and not s.startswith("+++")]
        longest = max(map(len, added))
        assert longest <= rules["max_line_chars"] and len(patch) <= rules["max_patch_bytes"]
        assert not any(word in line for line in added for word in rules["forbidden_words"])
        assert not any(s.startswith(("rename ", "old mode ", "new mode ", "GIT binary patch", "Binary files", "deleted file mode")) for s in lines)
        for path in paths:
            assert path.startswith(rules["allowed_prefix"]) and not path.startswith(tuple(rules["forbidden_prefixes"]))
            assert path.endswith(tuple(rules["allowed_suffixes"]))
            mode = git("ls-tree",tree,path).split()[0]
            parent_mode = git("ls-tree",pin,path).split()[:1] or [b"100644"]
            assert [mode] == parent_mode and mode in (b"100644", b"100755")
            data = git("show",tree+":"+path)
            assert data and b"\0" not in data
            if path.endswith(".py"):
                ast.parse(data)
            origin = 311 if path in changed_from_prompt else 310
            assert data == git("show",trees[origin]+":"+path)
            if export_stage is not None:
                assert (export_stage / path).read_bytes() == data
            sources[path] = {"donor": origin, "sha256": sha(data)}

    decode = read_json(HERE / "decode_dispatch.json")
    prompt = read_json(HERE / "prompt_dispatch.json")
    assert decode == read_json(donor(311) / "dispatch.json") and prompt == read_json(donor(310) / "dispatch.json")
    assert decode["counts"]["full_forward_extend_pairs"] == 40 and len(prompt["cases"]) == 8
    target_attn = sources[changed_from_prompt[0]]["sha256"]
    assert decode["sources_sha256"]["v311"] == target_attn
    assert target_attn == read_json(donor(311) / "sm90.json")["source_sha256"]["candidate"]
    aot = read_json(donor(310) / "aot.json")
    layers = "python/sglang/srt/layers/muse_spark/"
    assert aot["tiled_source_sha256"] == sources[layers+"humming_prompt_route_tiles.py"]["sha256"]
    assert aot["wave_source_sha256"] == sources[layers+"humming_prefill_waves.py"]["sha256"]
    inherited = {}
    for n, filename in [(310, "aot.json"), (310, "interpret.json"), (311, "sm90.json"), (311, "interpreter.json")]:
        data = (donor(n) / filename).read_bytes()
        record = json.loads(data)
        if n == 310:
            assert record["wave_source_sha256"] == aot["wave_source_sha256"]
            assert record["tiled_source_sha256"] == aot["tiled_source_sha256"]
        else:
            assert record["source_sha256"]["candidate"] == target_attn
        inherited[f"v{n}/{filename}"] = sha(data)
    return dict(candidate="V312", base=pin, common_parent="V204", donor_patch_sha256=manifests, donor_trees=trees, tree=tree,
                official_export="okc serve diff, Git diff context 1", patch_sha256=sha(patch), delta_sha256=sha(delta),
                bytes=len(patch), max_allowed_bytes=rules["max_patch_bytes"], files=len(paths), max_added_line_chars=longest,
                exact_donor_sources=sources, forward_reverse_both_orders=True, launch_flags=flags,
                scored_v204_receipt="ad275bf67d9a4a0a", official_contract="https://openkernelchallenge.dev/api/v1/serving",
                contract_rules_sha256=sha(json.dumps(rules, sort_keys=True).encode()),
                inherited_kernel_evidence_sha256=inherited,
                rerun_sha256={f:sha((HERE/f).read_bytes()) for f in ("decode_dispatch.json","prompt_dispatch.json","flags")},
                tested_on_eight_h200=False, submitted=False)


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else None), indent=2))
