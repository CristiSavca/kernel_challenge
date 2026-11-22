"""Verify exact private-patch composition in temporary Git indexes."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
COMPONENTS = ROOT.parent
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
FILES = {
    "v64":"v64_humming_grouped_sequence_parallel/patch.diff",
    "v106":"v106_paged_swa_suffix_prefill/patch.diff",
    "v108":"v108_replicated_sp_draft_context/patch.diff",
    "v110":"v110_verify_fp8_qkv_store/patch.diff",
}
FLASH = "python/sglang/srt/layers/attention/flashattention_backend.py"


def check(stage):
    def git(*args, env=None):
        return subprocess.check_output(["git","-C",str(stage),*args],env=env)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = dict(os.environ, GIT_INDEX_FILE=str(tmp/"component.index"))
        def tree():
            return git("write-tree",env=env).decode().strip()
        def read(value):
            git("read-tree",value,env=env)
        def apply(path, *flags):
            git("apply","--cached",*flags,str(path),env=env)
        def changed(a,b):
            return set(git("diff","--name-only",a,b).decode().splitlines())
        parents={}
        hashes={}
        for key, file in FILES.items():
            src=COMPONENTS/file
            read(PIN)
            apply(src)
            parents[key]=tree()
            hashes[key]=hashlib.sha256(src.read_bytes()).hexdigest()
        read(parents["v106"])
        apply(COMPONENTS/"v108_replicated_sp_draft_context/delta.diff")
        apply(COMPONENTS/"v110_verify_fp8_qkv_store/delta.diff","--3way")
        full=tree()
        # Repeat in reverse feature order. This also holds when a clean
        # integrated checkout has no component-specific Git refs or objects.
        read(parents["v110"])
        apply(COMPONENTS/"v108_replicated_sp_draft_context/delta.diff")
        patch=tmp/"v106_from_v64.diff"
        patch.write_bytes(git("diff","--full-index",parents["v64"],parents["v106"]))
        apply(patch,"--3way")
        assert tree()==full
        read(PIN); apply(ROOT/"patch.diff"); assert tree()==full
        apply(ROOT/"patch.diff","-R"); assert tree()==git("rev-parse","HEAD^{tree}").decode().strip()
        read(parents["v106"]); apply(ROOT/"delta.diff"); assert tree()==full
        components={n:changed(parents["v64"],t) for n,t in parents.items() if n!="v64"}
        assert (components["v106"] & components["v110"]) == {FLASH}
        assert not (components["v108"] & (components["v106"] | components["v110"]))
        exclusive=[]
        for key, names in components.items():
            for name in sorted(names-{FLASH}):
                assert git("show",f"{full}:{name}")==git("show",parents[key]+":"+name)
                exclusive.append(name)
        def funcs(t):
            text=git("show",f"{t}:{FLASH}").decode()
            cls=next(n for n in ast.parse(text).body if getattr(n,"name",None)=="FlashAttentionBackend")
            return {n.name:ast.unparse(n)+"\n" for n in cls.body if isinstance(n,ast.FunctionDef)}
        tables={k:funcs(parents[k]) for k in ("v64","v106","v110")}; merged=funcs(full)
        shared=[]; features=[]
        for name,actual in merged.items():
            a=tables["v106"].get(name,""); base=tables["v64"].get(name,""); b=tables["v110"].get(name,"")
            if a==b: assert actual==a
            elif a==base: assert actual==b; features.append(name)
            elif b==base: assert actual==a; features.append(name)
            else:
                group=[]
                for i,text in enumerate((a,base,b)):
                    p=tmp/f"{i}.method";p.write_text(text);group.append(str(p))
                expected=subprocess.check_output(["git","merge-file","-p",*group]).decode()
                assert expected==actual,name
                shared.append(name)
        assert shared==["__init__","forward_extend"]
        return {"pin":PIN,"component_hashes":hashes,"component_trees":parents,"integrated_tree":full,
                "commutes_in_both_component_orders":True,"reverse_restores_pin":True,"v106_delta_exact":True,
                "single_source_component_paths_byte_identical":sorted(exclusive),
                "backend_methods_compared":len(merged),"shared_methods_exact_ast_three_way":shared,
                "single_component_changed_methods":features}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    print(json.dumps(check(parser.parse_args().stage),indent=2))
