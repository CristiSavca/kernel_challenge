"""Rebuild the combined tree using a private index and immutable serving exports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
ROOT = "python/sglang/srt/"
ORDER = ROOT + "arg_groups/field_order.py"
SPEC = ROOT + "arg_groups/fields/spec.py"
MUSE = ROOT + "models/muse_spark.py"


def artifact(number):
    folder, = HERE.parent.glob(f"v{number}_*")
    return folder


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(checkout, index, include_hot=True):
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}

    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", checkout, *args], input=data, env=env)

    def apply(patch, *args):
        git("apply", "--cached", "--whitespace=error-all", *args, "-", data=patch)

    def tree():
        return git("write-tree").decode().strip()

    donors = {}
    for n in (168, 169, 171, 172, 173, *((174,) if include_hot else ())):
        folder = artifact(n)
        meta = json.loads((folder/"validation.json").read_bytes())
        patch = (folder/"patch.diff").read_bytes()
        assert meta["patch_sha256"] == sha(patch)
        git("read-tree", PIN)
        apply(patch)
        donors[n] = tree()
        assert donors[n] == meta["tree"]

    git("read-tree", donors[173])
    for n in (169, 171):
        apply((artifact(n)/"delta.diff").read_bytes())
    apply((artifact(172)/"delta.diff").read_bytes(), *("--exclude="+p for p in (ORDER, SPEC, MUSE)))
    # Keep all V171 positional slots. V172 is the only newly appended slot.
    for p in (ORDER, SPEC, MUSE):
        old = git("show", donors[168]+":"+p)
        donor = git("show", donors[172]+":"+p)
        current = git("show", ":"+p)
        if p == ORDER:
            anchor = b'    "speculative_dflash_sp_prefill_context",\n'
            assert current.count(anchor) == 1
            merged = current.replace(anchor, anchor+b'    "speculative_dflash_share_verify_schedule",\n')
        elif p == SPEC:
            anchor = b'    speculative_domino_candidate_pool_size:'
            start = b'    speculative_dflash_share_verify_schedule:'
            block = donor[donor.index(start):donor.index(anchor)]
            assert current.count(anchor) == 1 and start not in current
            merged = current.replace(anchor, block+anchor)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                paths = [Path(tmp)/name for name in ("combined", "v168", "v172")]
                for path, data in zip(paths, (current, old, donor), strict=True):
                    path.write_bytes(data)
                merged = subprocess.check_output(["git", "merge-file", "-p", *paths])
            compile(merged, p, "exec")
        obj = git("hash-object", "-w", "--stdin", data=merged).decode().strip()
        git("update-index", "--cacheinfo", "100644,"+obj+","+p)
    # This per-invocation attribute must not shift the parent dataclass constructor.
    batch_path = ROOT + "model_executor/forward_batch_info.py"
    data = git("show", ":"+batch_path)
    attr = b"    # Set only for the duration of one target model call.\n    _fa3_verify_schedules: Optional[dict] = None\n"
    end = b"    forward_metadata_replan_equivalent: bool = False\n"
    assert data.count(attr) == data.count(end) == 1
    data = data.replace(attr, b"").replace(end, end+b"\n"+attr)
    obj = git("hash-object", "-w", "--stdin", data=data).decode().strip()
    git("update-index", "--cacheinfo", "100644,"+obj+","+batch_path)
    if include_hot:
        apply(git("diff", donors[173], donors[174]))
    return tree(), donors
