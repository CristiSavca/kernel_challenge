"""Reconstruct the official complete export from V182, V171's endpoint, and V185's preprojection."""
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
C = HERE.parent
STAGE = Path(sys.argv[1]).resolve()
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
V185_DIGEST = "f611fec5bb999b6e16cde149ae5d5b890492a587be24b3624930913d7b45bc41"
V185_TREE = "296d7bd19fa14322e75f41d84d454d8387679da2"
FLAG = "--speculative-dflash-sp-prefill-context"
URL = "https://openkernelchallenge.dev/api/v1/serving"
SRT = "python/sglang/srt/"
ORDER = SRT + "arg_groups/field_order.py"
sha = lambda x: hashlib.sha256(x).hexdigest()
donors = {v: next(C.glob("v"+v+"_*")) for v in ("68","168","171","182","185")}
frozen = {v: json.loads((d/"validation.json").read_text()) for v,d in donors.items() if v in ("168","171","182")}

def rebase_spec_field(diff):
    old = b'@@ -553,5 +553,6 @@ POSITIONAL_FIELD_ORDER = (\n     "aux_hidden_state_capture",\n     "return_aux_hidden_state_layer_ids",\n     "muse_sparse_prompt_exchange",\n'
    new = b'@@ -553,4 +553,5 @@ POSITIONAL_FIELD_ORDER = (\n     "aux_hidden_state_capture",\n     "return_aux_hidden_state_layer_ids",\n'
    assert diff.count(old)==1
    return diff.replace(old,new)

def check_cli(get, scratch):
    helper_path = donors["171"]/"verify.py"
    module_spec = importlib.util.spec_from_file_location("verify_v171",helper_path)
    helper = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
    helper.check_cli(get,scratch)
    return helper.msgspec.__version__

def main():
    service = json.loads((Path(sys.argv[2]).read_bytes() if len(sys.argv)>2 else subprocess.check_output(["curl","-fsSL",URL])))
    assert service["base"]["commit"]==BASE
    full,delta=[(HERE/f).read_bytes() for f in ("patch.diff","delta.diff")]
    flags={**frozen["182"]["launch_flags"],FLAG:""}
    assert len(flags)==5
    d171=(donors["171"]/"delta.diff").read_bytes();d185=(donors["185"]/"delta.diff").read_bytes()
    assert sha(d171)==frozen["171"]["delta_sha256"] and sha(d185)=="7b2df781f001fa78db9d6e437773cc3bdc49bfbb11d601bab3f7aa7016c73aba"
    with tempfile.TemporaryDirectory() as td:
        t=Path(td);e=dict(os.environ,GIT_INDEX_FILE=str(t/"index"))
        def git(*args,data=None):return subprocess.check_output(["git","-C",str(STAGE),*args],env=e,input=data)
        def apply(tree,patch,reverse=False):
            git("read-tree",tree);git("apply","--cached","--whitespace=error-all",*(["--reverse"] if reverse else []),"-",data=patch)
            return git("write-tree").decode().strip()
        trees={}
        for name,d in donors.items():
            p=(d/"patch.diff").read_bytes();tree=trees[name]=apply(BASE,p)
            if name in frozen:assert tree==frozen[name]["tree"] and sha(p)==frozen[name]["patch_sha256"]
            if name=="185":assert tree==V185_TREE and sha(p)==V185_DIGEST
        assert apply(trees["168"],d171)==trees["171"] and apply(trees["171"],d185)==trees["185"]
        rebased=rebase_spec_field(d171)
        intermediate=apply(trees["182"],rebased)
        tree=apply(intermediate,d185)
        assert apply(tree,d185,True)==intermediate and apply(intermediate,rebased,True)==trees["182"]
        assert tree==apply(BASE,full)==apply(trees["182"],delta)
        assert apply(tree,delta,True)==trees["182"] and apply(tree,full,True)==git("rev-parse",BASE+"^{tree}").decode().strip()
        changes=git("diff","--name-only",trees["182"],tree).decode().splitlines()
        paths=git("diff","--name-only",BASE,tree).decode().splitlines();rules=service["rules"]
        assert len(changes)==9 and len(paths)==25
        for p in paths:
            assert p.startswith(rules["allowed_prefix"]) and not p.startswith(tuple(rules["forbidden_prefixes"]))
            assert p.endswith(tuple(rules["allowed_suffixes"]))
            data=git("show",tree+":"+p)
            assert (STAGE/p).read_bytes()==data
            mode=git("ls-tree",BASE,p).split()
            assert git("ls-tree",tree,p).startswith((mode[0] if mode else b"100644")+b" blob ")
            if p.endswith(".py"):compile(data,p,"exec")
        assert not any(line.startswith((b"rename ",b"Binary files ",b"GIT binary patch")) for line in full.splitlines())
        added=[line[1:] for line in full.decode().splitlines() if line.startswith("+") and not line.startswith("+++ ")]
        assert len(full)<=rules["max_patch_bytes"] and max(map(len,added))<=rules["max_line_chars"]
        assert not any(word in line for line in added for word in rules["forbidden_words"])
        banned=[s for group in service["launch_flags"]["denied"].values() for s in group]
        assert not any(fnmatch.fnmatchcase(k,pat) for k in flags for pat in banned)
        def contents(p):return git("show",tree+":"+p)
        def fields(tree):
            return ast.literal_eval(next(x.value for x in ast.parse(git("show",tree+":"+ORDER)).body if isinstance(x,ast.Assign)))
        assert fields(tree)==(*fields(trees["182"]),FLAG[2:].replace("-","_"))
        version=check_cli(contents,t)
        reference=json.loads((C.parent/"runs/04b01f958aa5487c.json").read_text())
        assert reference["mode"]=="check" and reference["result"]["passed"] is True
        rr=reference["result"];assert rr["serving_flags"]==frozen["182"]["launch_flags"] and rr["rungs"][0]["tok_s_per_gpu"]==17512.9
        assert rr["patch"]["sha256"]==sha((donors["68"]/"patch.diff").read_bytes())
        suites={p:sha((HERE/(p+".json")).read_bytes()) for p in ("worker","model","transport","producer","prompt_verify","paged","dispatch")}
        return dict(candidate="V187",base=BASE,parent="V182",parent_sha256=frozen["182"]["patch_sha256"],
                    parent_tree=trees["182"],endpoint_donor_sha256=frozen["171"]["patch_sha256"],projection_donor_sha256=V185_DIGEST,
                    projection_donor_tree=V185_TREE,intermediate_tree=intermediate,tree=tree,bytes=len(full),patch_sha256=sha(full),
                    delta_bytes=len(delta),delta_sha256=sha(delta),paths=paths,changed_from_parent=changes,launch_flags=flags,
                    checked_flag_patterns=len(banned),max_added_line_chars=max(map(len,added)),parent_positional_fields=len(fields(trees["182"])),
                    cli_msgspec=version,strict_apply_reverse_full_delta=True,strict_named_donor_reconstruction=True,
                    donor_delta_rebase="One hunk context line omits V168's separate sparse-expert flag. All V171 added/removed source lines and V185 delta bytes are original.",
                    hardware_control=dict(job=reference["id"],users=16,candidate="V68",tok_s_per_gpu=rr["rungs"][0]["tok_s_per_gpu"],
                                          gpu_memory_over_stock_gb=rr["gpu_memory"]["over_stock_gb"]),
                    suites=suites,official_rules=URL,official_export="okc serve diff .",physical_H200=False,submitted=False)
if __name__=="__main__":print(json.dumps(main(),indent=2))
