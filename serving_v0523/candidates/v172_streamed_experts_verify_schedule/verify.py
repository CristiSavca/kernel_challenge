"""Reconstruct V172 with new Git indices and exercise the pinned option parser."""
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

HERE = Path(__file__).resolve().parent
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
V168 = ("v168_streamed_experts_sparse_return", "95816b34347ec23816b3def52e99265449ce0c8df523e0a36f587603ae57b574")
V151 = ("v151_integrated_verify_scheduler", "2923c97cf66ca37c2c27ded28b53658e1a6d6ea5da3e828907da46fb7290b543")
V147 = ("v147_integrated_experts_prefill", "7a3d4dc6c0e526b60ca50bc975a6036da7796d8eb05f9e8afb70423164425437")
KEY = "speculative_dflash_share_verify_schedule"
FLAG = "--speculative-dflash-share-verify-schedule"
P = "python/sglang/srt/"
SPEC = P+"arg_groups/fields/spec.py"
ORDER = P+"arg_groups/field_order.py"
MUSE = P+"models/muse_spark.py"
FORWARD = P+"model_executor/forward_batch_info.py"
BACKEND = P+"layers/attention/flashattention_backend.py"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def option(contents, scratch):
    import msgspec
    def load(name, content):
        file = scratch/(name.rsplit(".",1)[-1]+".py")
        file.write_bytes(content)
        spec = importlib.util.spec_from_file_location(name,file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    prefix = "sglang.srt.arg_groups"
    helper = load(prefix+".arg_utils",contents(P+"arg_groups/arg_utils.py"))
    load(prefix+".choices",contents(P+"arg_groups/choices.py"))
    fields = load(prefix+".fields.spec",contents(SPEC)).Spec
    parser = argparse.ArgumentParser()
    helper.add_cli_args_from_dataclass(parser,fields,fields=[KEY])
    assert getattr(fields(),KEY) is False and getattr(parser.parse_args([]),KEY) is False
    assert getattr(parser.parse_args([FLAG]),KEY) is True
    return msgspec.__version__

def main(stage):
    url = "https://openkernelchallenge.dev/api/v1/serving"
    live = json.loads(subprocess.check_output(["curl","-fsSL",url]))
    assert live["base"]["commit"] == BASE
    rules = live["rules"]
    full, delta = [(HERE/name).read_bytes() for name in ("patch.diff","delta.diff")]
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ,GIT_INDEX_FILE=tmp+"/index")
        def git(*args, data=None):
            return subprocess.check_output(["git","-C",stage,*args], env=env, input=data)
        def apply(start, patch, reverse=False):
            git("read-tree",start)
            git("apply","--cached","--whitespace=error-all",*(["--reverse"] if reverse else []),"-",data=patch)
            return git("write-tree").decode().strip()
        def paths(a,b):
            return git("diff","--name-only",a,b).decode().splitlines()
        def contents(tree,path):
            return git("show", tree+":"+path)
        def method(tree,path,cls,name):
            code=ast.parse(contents(tree,path))
            owner=next(n for n in code.body if getattr(n,"name","")==cls)
            return ast.dump(next(n for n in owner.body if getattr(n,"name","")==name))
        trees={}
        for folder, expected in (V147,V151,V168):
            patch=(HERE.parent/folder/"patch.diff").read_bytes()
            assert sha(patch)==expected
            trees[folder.split("_")[0]]=apply(BASE,patch)
        parent, donor = trees["v168"],trees["v151"]
        donor_delta=(HERE.parent/V151[0]/"delta.diff").read_bytes()
        assert apply(trees["v147"],donor_delta)==donor
        raw=apply(parent,donor_delta)
        tree=apply(BASE,full)
        assert tree==apply(parent,delta)
        assert apply(tree,delta,True)==parent and apply(tree,full,True)==git("rev-parse",BASE+"^{tree}").decode().strip()
        assert paths(raw,tree)==[ORDER]
        def ordered(t):
            a=ast.parse(contents(t,ORDER))
            return ast.literal_eval(next(n.value for n in a.body if isinstance(n,ast.Assign)))
        assert ordered(tree)==(*ordered(parent),KEY)
        assert len(ordered(tree))==len(set(ordered(tree)))
        assert set(paths(parent,tree))=={BACKEND,FORWARD,MUSE,SPEC,ORDER}
        assert all(contents(tree,p)==contents(donor,p) for p in (BACKEND,FORWARD,SPEC))
        assert all(method(tree,MUSE,"MuseSparkTextForCausalLM",n)==method(donor,MUSE,"MuseSparkTextForCausalLM",n)
                   for n in ("__init__","forward"))
        assert method(tree,MUSE,"MuseSpark1xAttention","forward")==method(parent,MUSE,"MuseSpark1xAttention","forward")
        all_paths=paths(BASE,tree)
        for p in all_paths:
            data=contents(tree,p)
            mode=git("ls-tree",tree,p).split()[0]
            old_mode=git("ls-tree",BASE,p).split()
            assert p.startswith(rules["allowed_prefix"]) and not any(p.startswith(prefix) for prefix in rules["forbidden_prefixes"])
            assert p.endswith(tuple(rules["allowed_suffixes"])) and mode==(old_mode[0] if old_mode else b'100644')
            assert mode in (b'100644',b'100755') and data and data==(stage/p).read_bytes()
            if p.endswith(".py"): compile(data,p,"exec")
        flags=json.loads((HERE.parent/V168[0]/"validation.json").read_text())["launch_flags"]|{FLAG:""}
        fixed=[pattern for entries in live["launch_flags"]["denied"].values() for pattern in entries]
        assert not any(fnmatch.fnmatchcase(flag,p) for p in fixed for flag in flags)
        cli=option(lambda p:contents(tree,p),Path(tmp))
        added=[l[1:] for l in full.decode().splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert len(full)<=rules["max_patch_bytes"] and max(map(len,added))<=rules["max_line_chars"]
        assert not any(word in line for line in added for word in rules["forbidden_words"])
        assert not any(line.startswith(("old mode ","new mode ","rename from ","rename to ","GIT binary patch","Binary files "))
                       for line in full.decode().splitlines())
        print(json.dumps(dict(candidate="V172",base=BASE,parent_sha256=V168[1],parent_tree=parent,donor_sha256=V151[1],
            donor_delta_sha256=sha(donor_delta),donor_tree=donor,tree=tree,patch_sha256=sha(full),delta_sha256=sha(delta),
            bytes=len(full),delta_bytes=len(delta),strict_forward_reverse=True,paths=all_paths,changed_from_parent=paths(parent,tree),
            parent_positional_fields_preserved=len(ordered(parent)),pilot_flag_default=False,real_cli_msgspec=cli,
            byte_identical_donor_files=[BACKEND,FORWARD,SPEC],fixed_flag_patterns=len(fixed),max_added_line_chars=max(map(len,added)),
            launch_flags=flags,official_export="okc serve diff",rules_url=url,physical_h200=False,judge_submitted=False),indent=2))

if __name__=="__main__":
    main(Path(sys.argv[1]).resolve())
