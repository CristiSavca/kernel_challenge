"""Bind materialized test inputs to the official full-patch digest and Git tree."""
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bind(stage, directory, paths):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "validation.json").read_bytes())
    digest = sha((directory / "patch.diff").read_bytes())
    assert digest == manifest["patch_sha256"]
    source = {}
    for path in paths:
        committed = subprocess.check_output(["git", "-C", str(stage), "show", manifest["tree"] + ":" + path])
        data = (stage / path).read_bytes() if (stage / path).exists() else committed
        assert committed == data
        source[path] = sha(data)
    return dict(candidate=manifest["candidate"], tree=manifest["tree"], patch_sha256=digest,
        baseline=manifest["parent"], baseline_tree=manifest["parent_tree"],
        baseline_sha256=manifest["parent_sha256"], source_sha256=source)
