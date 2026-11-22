"""Compose measured V68 with the live-prompt-only FP8 write-and-read path."""

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

HERE = Path(__file__).resolve().parent
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
V64 = "b5a37c2923f8d1ab24c1c5cf7920a21e2c086774"
V68 = "f9966bf0dfca287d58d8d24ea2d055b3c2162702"
V178B = "f7b733938289fa943c2d2a675ce26e4e88a28cf7"
V178B_SHA = "ca7a01b3d4cce18622d23735cfdbe66ab1e4d4abb60ffa635c2cab1fed9a74dc"
BACK = "python/sglang/srt/layers/attention/flashattention_backend.py"
HELPER = "python/sglang/srt/layers/attention/direct_prompt_kv.py"


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def between(source, first, end):
    assert source.count(first) == 1, first
    lo = source.index(first)
    return source[lo:source.index(end, lo)]


def merge_backend(base, measured, prompt):
    head = "from sglang.srt.layers.attention.base_attn_backend import"
    imp = between(prompt, "from sglang.srt.layers.attention.direct_prompt_kv import (", head)
    out = once(measured, head, imp + head)
    policy = between(prompt, "        self._direct_fp8_prompt = (", "        if self._get_fa_runtime_policy is None:")
    at = "        self._hopper_verify_sms = ("
    out = once(out, at, policy + at)
    scope = "        cp_active = is_cp_active(forward_batch)\n\n        if k is not None:"
    out = once(out, scope, scope.replace("\n\n", "\n        direct_prompt_kv = None\n\n"))
    start = "                else:\n                    k_scale = k_descale if self.kv_cache_is_mxfp8 else layer.k_scale\n"
    end = "        # Use precomputed metadata across all layers"
    # The same precision/scale operations are used by prefill and verification.
    # Touch only the ordinary MHA writer in forward_extend.
    def writer(text):
        extend = text.index("    def forward_extend(")
        lo = text.index(start, extend)
        return text[lo:text.index(end, lo)]
    out = once(out, writer(base), writer(prompt))
    prep = between(prompt, "                if direct_prompt_kv is not None:\n                    key_cache,",
                   "                result = flash_attn_with_kvcache(")
    at = "                if result is None:\n                    result = flash_attn_with_kvcache("
    out = once(out, at, at.replace("                    result =", textwrap.indent(prep, "    ") + "                    result ="))
    ast.parse(out)
    return out


def construct(stage, index, prompt_patch):
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", stage, *args], env=env, input=data)

    for number, tree, patch in ((64, V64, None), (68, V68, None), ("178b", V178B, prompt_patch)):
        raw = Path(patch if patch else next(HERE.parent.glob(f"v{number}_*/patch.diff"))).read_bytes()
        if number == "178b":
            assert hashlib.sha256(raw).hexdigest() == V178B_SHA
        git("read-tree", PIN)
        git("apply", "--cached", "--whitespace=error-all", "-", data=raw)
        assert git("write-tree").decode().strip() == tree
    assert set(git("diff","--name-only",V64,V178B).decode().splitlines()) == {BACK, HELPER}
    git("read-tree", V68)
    backend = merge_backend(*(git("show", tree + ":" + BACK).decode() for tree in (V64,V68,V178B)))
    for path, data in ((BACK, backend.encode()), (HELPER, git("show", V178B + ":" + HELPER))):
        obj = git("hash-object","-w","--stdin",data=data).decode().strip()
        git("update-index","--add","--cacheinfo","100644," + obj + "," + path)
    return git("write-tree").decode().strip()


if __name__ == "__main__":
    stage = Path(sys.argv[1]).resolve()
    donor = Path(sys.argv[2]) if len(sys.argv) > 2 else next(HERE.parent.glob("v178b_*/patch.diff"))
    with tempfile.TemporaryDirectory() as tmp:
        print(construct(stage, Path(tmp) / "index", donor))
