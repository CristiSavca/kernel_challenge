"""Validate the official V176 export, reproducible composition and public write/launch restrictions."""
import ast
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from assemble import HERE, MUSE, ORDER, PIN, ROOT, artifact, build, sha


STAGE = Path(sys.argv[1]).resolve()
URL = "https://openkernelchallenge.dev/api/v1/serving"
BATCH = ROOT + "model_executor/forward_batch_info.py"


def method(source, owner, name):
    nodes = ast.parse(source).body
    if owner:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)


def field_names(data, name):
    owner = next(n for n in ast.parse(data).body if isinstance(n, ast.ClassDef) and n.name == name)
    return [n.target.id for n in owner.body if isinstance(n, ast.AnnAssign)]


def cli_order(data):
    return tuple(ast.literal_eval(next(n.value for n in ast.parse(data).body if isinstance(n, ast.Assign))))


def main():
    published = json.loads(subprocess.check_output(["curl", "-fsSL", URL]))
    assert published["base"]["commit"] == PIN
    rules = published["rules"]
    full = (HERE/"patch.diff").read_bytes()
    delta = (HERE/"delta.diff").read_bytes()
    manifests = {n: json.loads((artifact(n)/"validation.json").read_bytes()) for n in (168,169,171,172,173,174)}
    flags = {key: val for n in (168,169,171,172,174) for key,val in manifests[n]["launch_flags"].items()}
    assert len(flags) == 7 and all(flags[k] == v for m in manifests.values() for k,v in m["launch_flags"].items())
    denied = [p for patterns in published["launch_flags"]["denied"].values() for p in patterns]
    assert not any(fnmatch.fnmatchcase(f,p) for f in flags for p in denied)
    lines = [l[1:] for l in full.decode().splitlines() if l.startswith("+") and not l.startswith("+++ ")]
    assert len(full) <= rules["max_patch_bytes"] and max(map(len,lines)) <= rules["max_line_chars"]
    assert not any(word in line for word in rules["forbidden_words"] for line in lines)
    assert not any(line.startswith((b"old mode ", b"new mode ", b"Binary files ", b"GIT binary patch", b"rename ")) for line in full.splitlines())

    with tempfile.TemporaryDirectory() as tmp:
        idx = Path(tmp)/"index"
        tree, parents = build(STAGE, idx)
        env = {**os.environ, "GIT_INDEX_FILE":str(idx)}
        def git(*args, data=None):
            return subprocess.check_output(["git","-C",STAGE,*args],env=env,input=data)

        def apply(start, patch, reverse=False):
            git("read-tree", start)
            git("apply","--cached","--whitespace=error-all",*(["--reverse"] if reverse else []),"-",data=patch)
            return git("write-tree").decode().strip()

        assert apply(PIN,full) == apply(parents[168],delta) == tree
        assert apply(tree,full,True) == git("rev-parse",PIN+"^{tree}").decode().strip()
        assert apply(tree,delta,True) == parents[168]
        # The two independent revisions to the Humming adapter commute without manual edits.
        prefill=(artifact(169)/"delta.diff").read_bytes()
        medium=(artifact(173)/"delta.diff").read_bytes()
        assert apply(parents[169],medium) == apply(parents[173],prefill)
        def src(t,p):
            return git("show",t+":"+p)

        paths = git("diff","--name-only",PIN,tree).decode().splitlines()
        changes = git("diff","--name-only",parents[168],tree).decode().splitlines()
        for row in git("diff","--raw",PIN,tree).decode().splitlines():
            old,new = row.split()[:2]
            assert new in ("100644","100755") and (old==":"+new or (old==":000000" and new=="100644"))
        for p in paths:
            assert p.startswith(rules["allowed_prefix"]) and p.endswith(tuple(rules["allowed_suffixes"]))
            assert not any(p.startswith(x) for x in rules["forbidden_prefixes"])
            data = src(tree,p)
            assert (STAGE/p).read_bytes() == data
            if p.endswith(".py"):
                compile(data,p,"exec")
        cu="python/sglang/kernels/jit/csrc/muse_spark/"
        compiler=json.loads((artifact(174)/"sm90.json").read_text())
        for name,digest in compiler["build_inputs"]["new"].items():
            assert sha(src(tree,cu+name)) == digest
        assert src(tree,ROOT+"layers/attention/flashattention_backend.py") == src(parents[172],ROOT+"layers/attention/flashattention_backend.py")
        for suffix in ("layers/dflash_context_projection.py","layers/logits_processor.py","layers/layernorm_sp.py",
                       "model_executor/runner/prefill_cuda_graph_runner.py","speculative/dflash_worker_v2.py"):
            assert src(tree,ROOT+suffix) == src(parents[171],ROOT+suffix)
        for suffix in ("layers/muse_spark/prefill_expert_tiles.py","layers/muse_spark/muse_spark_fp4_moe.py",
                       "layers/muse_spark/sparse_prompt_exchange.py"):
            assert src(tree,ROOT+suffix) == src(parents[169],ROOT+suffix)
        adapter=ROOT+"layers/muse_spark/muse_spark_humming_moe.py"
        for n,names in ((169,("prepare_humming_nvfp4","compact_prefill_and_combine","_compact_prefill_windows","humming_grouped_expert_outputs")),
                        (174,("_tma_route_pair_module","humming_expert_outputs"))):
            for f in names:
                assert ast.dump(method(src(tree,adapter),None,f)) == ast.dump(method(src(parents[n],adapter),None,f))
        # The model's scheduler scope and the complete prompt-output tail both survived the merge.
        def body(t):
            return method(src(t,MUSE),"MuseSparkTextForCausalLM","forward").body
        assert ast.dump(body(tree)[0])==ast.dump(body(parents[172])[0])
        assert [ast.dump(n) for n in body(tree)[1:]]==[ast.dump(n) for n in body(parents[171])[1:]]
        old=cli_order(src(parents[168],ORDER)); new=cli_order(src(tree,ORDER))
        assert new==(*old,"speculative_dflash_sp_prefill_context","speculative_dflash_share_verify_schedule")
        before=field_names(src(parents[168],BATCH),"ForwardBatch")
        after=field_names(src(tree,BATCH),"ForwardBatch")
        assert after==[*before,"_fa3_verify_schedules"]
    print(json.dumps(dict(candidate="V176",base=PIN,parent="V168",parent_tree=parents[168],parent_sha256=manifests[168]["patch_sha256"],
        tree=tree,patch_sha256=sha(full),bytes=len(full),delta_sha256=sha(delta),delta_bytes=len(delta),
        component_sha256={n:m["patch_sha256"] for n,m in manifests.items()},paths=paths,changed_from_parent=changes,
        launch_flags=flags,official_export="okc serve diff",official_rules=URL,checked_flag_patterns=len(denied),
        max_added_line_chars=max(map(len,lines)),strict_apply_reverse_full_delta=True,
        commutative_expert_adapter_edits=True,parent_positional_server_fields=len(old),
        parent_dataclass_annotations_preserved=len(before),physical_H200=False,submitted=False),indent=2))


if __name__=="__main__":
    main()
