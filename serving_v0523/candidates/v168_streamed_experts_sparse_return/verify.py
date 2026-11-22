"""Rebuild V168 from the archived exports in both component orders using isolated Git indices."""
import ast
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENTS = {
    "v163_sparse_prompt_exchange":"d4840308ea25166a1a3db0b4358beb6d59b9de97faaa43ef8a3795241bdc33a1",
    "v164_tma_experts_sparse_prompt":"a8b11f1460005a0290dfef86e97a8f043c016ac8b4540f1f423cd7cdd6a6ba04",
    "v165_streamed_expert_stages":"687ee3877a2fd0b97a76cb8ee6df260306d2acf57b91ebe16cfed6ebf95e0348",
    "v167_sparse_expert_return":"de58851c3cef9bfdf1c432b566f9294b41bbd4a81cad62e182022de11a22ba5d",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    official_url = "https://openkernelchallenge.dev/api/v1/serving"
    live = json.loads(subprocess.check_output(["curl","-fsSL",official_url]))
    assert live["base"]["commit"] == BASE
    rules = live["rules"]
    flags = json.loads((HERE.parent/"v167_sparse_expert_return/validation.json").read_text())["launch_flags"]
    assert flags == json.loads((HERE.parent/"v164_tma_experts_sparse_prompt/validation.json").read_text())["launch_flags"]
    denied = [p for group in live["launch_flags"]["denied"].values() for p in group]
    assert not any(fnmatch.fnmatchcase(f,p) for f in flags for p in denied)
    full,delta,adjust = [(HERE/filename).read_bytes() for filename in ("patch.diff","delta.diff","window_budget.diff")]
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ,GIT_INDEX_FILE=tmp+"/index")
        def git(*args,data=None):
            return subprocess.check_output(["git","-C",STAGE,*args],input=data,env=env)
        def apply(tree, patch, reverse=False):
            git("read-tree",tree)
            git("apply","--cached","--whitespace=error-all",*(["--reverse"] if reverse else []),"-",data=patch)
            return git("write-tree").decode().strip()
        parents={}
        for folder,digest in PARENTS.items():
            patch=(HERE.parent/folder/"patch.diff").read_bytes()
            assert sha(patch)==digest
            parents[folder.split("_")[0]]=apply(BASE,patch)
        assert parents["v164"]=="c45818ae2d5edecf97d279c55f473e8c0cd2f169"
        assert parents["v165"]=="e6079797e56ec23957c80454710e84d7e7319fdb"
        assert parents["v167"]=="20f9a009fea6646ab2099493e18a3f58f82b3b22"
        decode=(HERE.parent/"v165_streamed_expert_stages/delta.diff").read_bytes()
        prompt=(HERE.parent/"v167_sparse_expert_return/delta.diff").read_bytes()
        assert apply(parents["v164"],decode)==parents["v165"]
        assert apply(parents["v163"],prompt)==parents["v167"]
        raw=apply(parents["v165"],prompt)
        assert raw==apply(apply(parents["v164"],prompt),decode)
        tree=apply(BASE,full)
        assert tree==apply(raw,adjust)==apply(parents["v164"],delta)
        assert apply(tree,full,True)==git("rev-parse",BASE+"^{tree}").decode().strip()
        assert apply(tree,delta,True)==parents["v164"] and apply(tree,adjust,True)==raw
        assert subprocess.check_output(["git","-C",STAGE,"write-tree"],text=True).strip()==tree
        def paths(start,end):
            return git("diff","--name-only",start,end).decode().splitlines()
        changed=paths(parents["v164"],tree)
        sources=paths(BASE,tree)
        blobs={p:git("show",tree+":"+p) for p in sources}
        for p,data in blobs.items():
            assert p.startswith(rules["allowed_prefix"]) and not any(p.startswith(d) for d in rules["forbidden_prefixes"])
            mode=git("ls-tree",tree,p).split()[0]
            previous_mode=git("ls-tree",BASE,p).split()
            assert p.endswith(tuple(rules["allowed_suffixes"])) and mode in (b"100644",b"100755")
            assert mode==(previous_mode[0] if previous_mode else b"100644")
            assert (STAGE/p).read_bytes()==data
            if p.endswith(".py"):
                compile(data,p,"exec")
        def method(tree,path,name):
            node=next(n for n in ast.walk(ast.parse(git("show",tree+":"+path))) if getattr(n,"name",None)==name)
            return ast.dump(node,include_attributes=False)
        hum=next(p for p in changed if p.endswith("muse_spark_humming_moe.py"))
        sparse=next(p for p in changed if p.endswith("sparse_prompt_exchange.py"))
        reducer=next(p for p in changed if p.endswith("muse_spark_sparse_return.py"))
        model=next(p for p in changed if p.endswith("models/muse_spark.py"))
        for f in ("prepare_humming_nvfp4","humming_expert_outputs","humming_grouped_expert_outputs"):
            assert method(tree,hum,f)==method(parents["v165"],hum,f)
        for f in ("_return_partials","_exchange_inputs","_group_width","_local_grouped_eligible"):
            assert method(tree,sparse,f)==method(parents["v167"],sparse,f)
        assert method(tree,model,"_forward_sp_replicated_experts")==method(parents["v167"],model,"_forward_sp_replicated_experts")
        assert blobs[reducer]==git("show",parents["v167"]+":"+reducer)
        cuda_paths=[p for p in sources if p.endswith(".cuh")]
        assert all(blobs[p]==git("show",parents["v165"]+":"+p) for p in cuda_paths)
        added=[s[1:] for s in full.decode().splitlines() if s.startswith("+") and not s.startswith("+++")]
        assert len(full)<=rules["max_patch_bytes"] and max(map(len,added))<=rules["max_line_chars"]
        assert not any(word in "\n".join(added) for word in rules["forbidden_words"])
        print(json.dumps(dict(candidate="V168",base=BASE,parent_sha256=PARENTS["v164_tma_experts_sparse_prompt"],
            patch_sha256=sha(full),bytes=len(full),delta_sha256=sha(delta),delta_bytes=len(delta),
            window_budget_sha256=sha(adjust),tree=tree,before_window_budget=raw,component_sha256=PARENTS,
            paths=sources,changed_from_parent=changed,cuda_paths_identical_to_v165=cuda_paths,
            reducer_identical_to_v167=sha(blobs[reducer]),launch_flags=flags,official_export="okc serve diff",
            strict_apply_both_orders_full_delta_reverse=True,max_added_line_chars=max(map(len,added)),
            fixed_flag_patterns=len(denied),official_rules=official_url,physical_H200=False,judge_submitted=False),indent=2))


if __name__=="__main__":
    main()
