"""Reconstruct both tested trees from the public pin and validate source, flags and evidence."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
TRACK=HERE.parents[1]
SG=TRACK/"sglang"
PARENT=HERE.parent/"v120b_native_muse_prompt"
PUB=HERE.parent/"v306_compact_prompt_waves/public_contract.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    before=json.loads((PARENT/"validation.json").read_bytes())
    rules=json.loads(PUB.read_bytes())
    pin=rules["base"]["commit"]
    judge=json.loads((TRACK/"runs/a73535cda6bf4cac.json").read_bytes())
    accepted=before["launch_flags"]
    assert before["base"]==pin and judge["result"]["patch"]["sha256"]==before["patch_sha256"]
    assert judge["serving_flags"]==accepted and judge["result"]["concurrency"]==[32]
    parent=(PARENT/"patch.diff").read_bytes()
    full=(HERE/"patch.diff").read_bytes()
    delta=(HERE/"delta.diff").read_bytes()
    assert sha(parent)==before["patch_sha256"]
    with tempfile.TemporaryDirectory() as temp:
        scratch=Path(temp);(scratch/"objects").mkdir()
        original=subprocess.check_output(["git","-C",SG,"rev-parse","--absolute-git-dir"],text=True).strip()
        env=dict(os.environ,GIT_INDEX_FILE=str(scratch/"index"),GIT_OBJECT_DIRECTORY=str(scratch/"objects"),
                 GIT_ALTERNATE_OBJECT_DIRECTORIES=str(Path(original)/"objects"))
        def git(*args,data=None):
            return subprocess.check_output(["git","-C",SG,*args],env=env,input=data)
        def apply(revision,patch,reverse=False):
            git("read-tree",revision)
            git("apply","--whitespace=error-all","--cached",*(["-R"] if reverse else []),"-",data=patch)
            return git("write-tree").decode().strip()
        prior=apply(pin,parent); tree=apply(pin,full); base=git("rev-parse",pin+"^{tree}").decode().strip()
        assert prior==before["full_tree"] and apply(prior,delta)==tree and apply(tree,delta,True)==prior
        assert apply(tree,full,True)==base and git("diff","-U3","--no-renames",pin,tree)==full
        assert git("diff","-U3","--no-renames",prior,tree)==delta
        changed=git("diff","--name-only",prior,tree).decode().splitlines()
        kernel="python/sglang/kernels/ops/memory/allocator.py"
        callers=["python/sglang/srt/mem_cache/allocator/"+n+".py" for n in ("paged","unified_sub_pool")]
        assert changed==[kernel,*callers]
        all_paths=git("diff","--name-only",pin,tree).decode().splitlines()
        limits=rules["rules"]
        for path in all_paths:
            assert path.startswith(limits["allowed_prefix"]) and not path.startswith(tuple(limits["forbidden_prefixes"]))
            assert path.endswith(tuple(limits["allowed_suffixes"]))
            reference=git("ls-tree",pin,path); new=git("ls-tree",tree,path).split()[0]
            assert new==(reference.split()[0] if reference else b"100644") and new in (b"100644",b"100755")
            content=git("show",tree+":"+path);assert content and b"\0" not in content
            ast.parse(content)
        for patch in (full,delta):
            lines=patch.decode().splitlines()
            assert not any(x.startswith(("rename ","old mode ","new mode ","GIT binary patch","Binary files")) for x in lines)
            adds=[x[1:] for x in lines if x.startswith("+") and not x.startswith("+++")]
            assert max(map(len,adds))<=limits["max_line_chars"] and len(patch)<=limits["max_patch_bytes"]
            assert all(word not in line for word in limits["forbidden_words"] for line in adds)
        assert not any(fnmatch.fnmatchcase(flag,p) for flag in accepted for patterns in rules["launch_flags"]["denied"].values() for p in patterns)
        source_sha=sha(git("show",tree+":"+kernel));numeric=json.loads((HERE/"interpreter.json").read_bytes());assembly=json.loads((HERE/"sm90.json").read_bytes())
        assert source_sha==assembly["source_sha256"]==numeric["source_sha256"]
        assert len(assembly["builds"])==33 and all(x["spill_bytes"]==0 for x in assembly["builds"])
        assert assembly["plain_parent_error"].startswith("Unsupported function referenced: <function next_power_of_2")
        assert assembly["plain_candidate"]["sm90a_cubin_sha256"]==next(row["sm90a_cubin_sha256"] for row in assembly["builds"] if row["page"]==160 and row["bs"]==32 and row["seq_ptr"]=="*i64")
        affected=sorted(p.parent.name for p in HERE.parent.glob("*/patch.diff") if b'tl.arange(0, triton.next_power_of_2(page_size))' in
                        b'\n'.join(row for row in p.read_bytes().splitlines() if row.startswith(b'+')))
        assert "v120b_native_muse_prompt" in affected and HERE.name not in affected
        return dict(candidate="V307",base=pin,parent="V120b",parent_tree=prior,tree=tree,parent_sha256=sha(parent),patch_sha256=sha(full),
            delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),changed_from_parent=changed,paths=all_paths,source_sha256=source_sha,
            flags=accepted,original_job=judge["id"],original_error=judge["result"]["crash"]["message"],strict_apply_and_reverse_full_delta=True,
            public_contract=True,original_mode_for_every_existing_file=True,exact_affected_archives=affected,physical_gpu=False,submitted=False)

if __name__=="__main__":
    print(json.dumps(main(),indent=2))
