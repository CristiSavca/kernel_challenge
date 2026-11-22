"""Find Python name errors introduced by a frozen serving patch. Requires Pyflakes."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from pyflakes import __version__
from pyflakes.checker import Checker
from pyflakes.messages import UndefinedLocal, UndefinedName


BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", repo, *args])


def unresolved(source, path):
    if not source:
        return []
    messages = Checker(ast.parse(source, filename=path), filename=path).messages
    return sorted(
        (m.lineno, m.message % m.message_args)
        for m in messages if isinstance(m, (UndefinedName, UndefinedLocal))
    )


def audit(sglang, patches):
    results = {}
    baseline = {}
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        git(stage, "init", "-q")
        objects = git(sglang, "rev-parse", "--path-format=absolute", "--git-path", "objects").decode().strip()
        (stage/".git/objects/info/alternates").write_text(objects + "\n")
        for patch in patches:
            git(stage, "read-tree", BASE)
            git(stage, "apply", "--cached", str(patch.resolve()))
            paths = git(stage, "diff", "--cached", "--name-only", "--diff-filter=ACM", BASE).decode().splitlines()
            files = [name for name in paths if name.endswith(".py")]
            new = []
            for path in files:
                if path not in baseline:
                    has_file = bool(git(stage, "ls-tree", BASE, "--", path))
                    src = git(stage,"show",BASE + ":" + path) if has_file else b""
                    baseline[path] = Counter(text for _, text in unresolved(src, path))
                prior = baseline[path].copy()
                for line, text in unresolved(git(stage,"show",":" + path),path):
                    if prior[text]:
                        prior[text] -= 1
                    else:
                        new.append(dict(path=path,line=line,message=text))
            results[patch.parent.name] = dict(sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                                              checked_python=len(files),new_findings=new)
    return dict(pyflakes=__version__,base=BASE,candidates=results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patches",nargs="+",type=Path)
    ap.add_argument("--sglang",type=Path,default=Path(__file__).resolve().parents[1]/"serving_v0523/sglang")
    print(json.dumps(audit((args := ap.parse_args()).sglang,args.patches),indent=2))
