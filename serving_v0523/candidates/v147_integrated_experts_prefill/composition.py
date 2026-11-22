"""Verify the two official parent patches commute over their scored V64 anchor."""
import argparse
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
BACKEND = "python/sglang/srt/layers/attention/flashattention_backend.py"
SPAN = "python/sglang/srt/layers/attention/prefill_kv_span.py"
PARENTS = {64: "v64_humming_grouped_sequence_parallel", 144: "v144_pipelined_expert_projection",
           146: "v146_zero_copy_window_prefill"}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main(stage):
    patches = {key: (HERE.parent / folder / "patch.diff").read_bytes() for key, folder in PARENTS.items()}
    patches[147] = (HERE / "patch.diff").read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))

        def git(*args, contents=None):
            return subprocess.check_output(["git", "-C", stage, *args], env=env, input=contents)

        def apply(start, patch, reverse=False):
            git("read-tree", start)
            git("apply", "--cached", "--whitespace=error", *(["-R"] if reverse else []), "-", contents=patch)
            return git("write-tree").decode().strip()

        trees = {key: apply(BASE, patch) for key, patch in patches.items()}
        for key, patch in patches.items():
            assert apply(trees[key], patch, True) == git("rev-parse", BASE + "^{tree}").decode().strip()
        for parent, other in ((144, 146), (146, 144)):
            delta = git("diff", "--no-ext-diff", trees[64], trees[other])
            assert apply(trees[parent], delta) == trees[147]
            assert apply(trees[147], delta, True) == trees[parent]
        delta = (HERE / "delta.diff").read_bytes()
        assert apply(trees[144], delta) == trees[147] and apply(trees[147], delta, True) == trees[144]

        def paths(old, new):
            return git("diff", "--name-only", old, new).decode().splitlines()

        changed = paths(trees[144], trees[147])
        assert changed == sorted([BACKEND, SPAN])
        intersection = sorted(set(paths(trees[64], trees[146])) & set(paths(trees[64], trees[144])))
        assert intersection == [BACKEND]
        all_paths = paths(BASE, trees[147])
        hashes = {}
        for p in all_paths:
            assert p.startswith("python/sglang/")
            contents = git("show", trees[147] + ":" + p)
            assert contents == (stage / p).read_bytes()
            hashes[p] = digest(contents)
            if p.endswith(".py"):
                compile(contents, p, "exec")
            if p == SPAN:
                assert contents == git("show", trees[146] + ":" + p)
            elif p != BACKEND:
                assert contents == git("show", trees[144] + ":" + p)

    parent_meta = {k: json.loads((HERE.parent / PARENTS[k] / "validation.json").read_text()) for k in (144, 146)}
    for k, meta in parent_meta.items():
        assert meta["tree"] == trees[k] and meta["patch_sha256"] == digest(patches[k])
    assert digest(patches[64]) == parent_meta[144]["scored_anchor_sha256"]
    assert parent_meta[144]["launch_flags"] == parent_meta[146]["launch_flags"]
    flags = parent_meta[144]["launch_flags"]
    fixed = (HERE.parent.parent / "flags").read_text().split("# Fixed,")[1].split("\n", 1)[1]
    patterns = [s.strip() for s in fixed.replace("#", "").replace("\n", "").split(",") if s.strip()]
    assert not any(fnmatchcase(flag, pattern) for flag in flags for pattern in patterns)
    return dict(candidate="V147", base=BASE, parent="V144", anchor="V64", prefill_source="V146",
                patch_sha256=digest(patches[147]), bytes=len(patches[147]), delta_sha256=digest(delta),
                delta_bytes=len(delta), tree=trees[147], launch_flags=flags, fixed_flag_patterns=len(patterns),
                parent_patch_sha256={str(k):digest(patches[k]) for k in PARENTS},
                parent_trees={str(k):trees[k] for k in PARENTS}, changed_from_parent=changed,
                overlapping_paths=intersection, paths=all_paths, checked_python=sum(p.endswith(".py") for p in all_paths),
                source_sha256=hashes, both_merge_orders_and_reverse=True, official_export="okc serve diff .",
                physical_H200=False, judge_submitted=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("stage", type=Path)
    print(json.dumps(main(args.parse_args().stage), indent=2))
