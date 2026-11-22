"""Build the V290 tree from each admissible component order, reverse it, and check the live serving contract."""
import ast
from collections import Counter
import fnmatch
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
canonical = here.parents[1]/"sglang"
contract = json.loads(Path(sys.argv[1]).read_bytes()) if len(sys.argv)>1 else json.loads(
    subprocess.check_output(["curl","-fsSL","https://openkernelchallenge.dev/api/v1/serving"]))
pin = contract["base"]["commit"]
rules = contract["rules"]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file(n,name):
    return next(here.parent.glob(f"v{n}_*"))/name

def receipt(n):
    return json.loads(file(n,"validation.json").read_bytes())

with tempfile.TemporaryDirectory(prefix="okc-v290-audit-") as scratch:
    path=Path(scratch)
    objects=path/"objects"
    objects.mkdir()
    env={**os.environ,"GIT_INDEX_FILE":str(path/"index"),"GIT_OBJECT_DIRECTORY":str(objects),
         "GIT_ALTERNATE_OBJECT_DIRECTORIES":str(canonical/".git/objects")}
    def git(*args,data=None):
        return subprocess.run(["git","-C",canonical,*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              env=env,check=True).stdout
    def apply(ref,patch,reverse=False,merge=False):
        git("read-tree",ref)
        git("apply","--cached","--unidiff-zero","--whitespace=error-all",
            *(["--reverse"] if reverse else []),*(["--3way"] if merge else []),"-",data=patch)
        return git("write-tree").decode().strip()
    native=git("rev-parse",pin+"^{tree}").decode().strip()
    trees={}
    refs=(229,230,231,232,233,279,284,286,289)
    for n in refs:
        data=file(n,"patch.diff").read_bytes()
        m=receipt(n)
        assert sha(data)==m["patch_sha256"]
        trees[n]=apply(pin,data)
        assert trees[n]==m["tree"]
    deltas={n:file(n,"delta.diff").read_bytes() for n in (230,231,232,233,286,289)}
    for n,parent in ((230,229),(231,229),(232,229),(233,229),(286,279),(289,284)):
        assert apply(trees[parent],deltas[n])==trees[n]
    full=(here/"patch.diff").read_bytes()
    delta=(here/"delta.diff").read_bytes()
    current=apply(pin,full)
    parent=trees[289]
    assert apply(parent,delta)==current
    assert apply(current,delta,reverse=True)==parent
    assert apply(current,full,reverse=True)==native
    assert git("diff","--no-renames","-U1",pin,current)==full
    assert git("diff","--no-renames","-U1",parent,current)==delta
    ranked=git("diff","--no-renames","-U1",trees[229],parent)
    components={230:deltas[230],231:deltas[231],232:deltas[232],286:deltas[286],289:ranked}
    # 286 changes the transport file introduced by 230. All other component
    # orders must agree. Share identical prefix subtrees to avoid redundant IO.
    prefix_cache={():trees[229]}
    final_orders=0
    for sequence in permutations(components):
        if sequence.index(230)>sequence.index(286):
            continue
        for i,n in enumerate(sequence):
            prefix=sequence[:i+1]
            if prefix not in prefix_cache:
                prefix_cache[prefix]=apply(prefix_cache[sequence[:i]],components[n],merge=True)
        assert prefix_cache[sequence]==current,sequence
        final_orders+=1
    assert final_orders==60
    assert apply(apply(parent,deltas[233],merge=True),deltas[286],merge=True)==current
    paths=git("diff","--name-only",pin,current).decode().splitlines()
    changed=git("diff","--name-only",parent,current).decode().splitlines()
    def src(p):
        return git("show",current+":"+p)
    assert set(changed)==set(receipt(233)["changed_from_parent"])
    added=[l[1:] for l in full.decode().splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert len(full)<=rules["max_patch_bytes"] and max(map(len,added))<=rules["max_line_chars"]
    assert all(word not in full.decode() for word in rules["forbidden_words"])
    assert not re.search(rb"^(old mode|new mode|rename |GIT binary patch|Binary files|deleted file mode)",full,re.M)
    for p in paths:
        assert p.startswith(rules["allowed_prefix"]) and not any(p.startswith(d) for d in rules["forbidden_prefixes"])
        assert Path(p).suffix in rules["allowed_suffixes"]
        current_entry=git("ls-tree",current,p).split()
        before_entry=git("ls-tree",pin,p).split()
        assert current_entry[1]==b"blob"
        assert current_entry[0]==(before_entry[0] if before_entry else b"100644")
        data=src(p)
        assert data
        if p.endswith(".py"): compile(data,p,"exec")
        if p.endswith(".json"): json.loads(data)
    flags=receipt(233)["launch_flags"]
    assert receipt(289)["launch_flags"].items()<=flags.items()
    blocked=re.findall(r"--[\w*.-]+",contract["launch_flags"]["rules"])
    assert all(not any(fnmatch.fnmatch(k,p) for p in blocked) for k in flags)
    # Key production implementation closures stay byte-identical beneath the
    # three shared outer model dispatch files.
    expected={

        233:["kernels/ops/moe/muse_spark_peer_pack.py","kernels/ops/attention/muse_spark_qk_norm_rope_store.py",
             "srt/speculative/dflash_mixed_graph_admission.py","srt/speculative/dflash_mixed_forward.py",
             "srt/managers/scheduler_components/mixed_dflash_admission.py","srt/layers/attention/direct_prompt_kv.py"],
        289:["kernels/jit/csrc/muse_spark/ranked_pool_norm.cuh","srt/layers/communication/muse_spark_gemm_ag.py",
             "srt/model_executor/runner/decode_cuda_graph_runner.py","kernels/ops/layernorm/muse_spark_norm_residual_jit.py",
             "srt/model_executor/forward_batch_info.py"]
    }
    equal={}
    for n,matched in expected.items():
        equal[n]={}
        for sub in matched:
            p="python/sglang/"+sub
            assert src(p)==git("show",trees[n]+":"+p)
            equal[n][p]=sha(src(p))
    peer_path="python/sglang/srt/layers/muse_spark/peer_prompt_pack.py"
    def transport_defs(data):
        return {f.name:ast.dump(f,include_attributes=False) for f in ast.parse(data).body
                if isinstance(f,(ast.FunctionDef,ast.ClassDef)) and f.name in
                ("PeerPromptInputs","peer_storage_rows","prepare_peer_prompt_inputs")}
    donor_api=transport_defs(git("show",trees[286]+":"+peer_path))
    assert len(donor_api)==3 and transport_defs(src(peer_path))==donor_api
    bound={}
    for label,field in (("peer_grouped","sources"),("peer_ordering","sources"),("ranked_vote","sources"),("ranked_model","hashes"),("independent_transitions","source_sha256")):
        evidence=json.loads((here/(label+".json")).read_bytes())
        bound[label]={}
        for p,digest in evidence[field].items():
            if not p.startswith("python/sglang/"):
                p="python/sglang/srt/"+p
            assert sha(src(p))==digest,(label,p)
            bound[label][p]=digest
    assert json.loads((here/"peer_transport.json").read_bytes())["header_sha256"]==sha(src(peer_path))
    query=json.loads((here/"prompt_query.json").read_bytes())
    assert query["candidate_tree"]==current and query["patch_sha256"]==sha(full)
    assert query["passed"] and query["query_bytes"]==67248128
    for p,digest in query["source_sha256"]["candidate"].items():
        assert sha(src(p))==digest
    payload=json.loads((here/"routed_payload.json").read_bytes())
    for p,digest in payload["sources"].items():
        assert sha((here.parents[2]/p).read_bytes())==digest
    joint=[p for p in changed if git("diff","--name-only",trees[229],parent,"--",p).strip()]
    assert sorted(Path(p).name for p in joint)==["muse_spark.py","muse_spark_fp4_moe.py","muse_spark_humming_moe.py"]
    print(json.dumps(dict(candidate="V290",parent="V289",base=pin,parent_tree=parent,tree=current,
        patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),max_added_line=max(map(len,added)),
        parent_sha256=receipt(289)["patch_sha256"],paths=paths,changed_from_parent=changed,launch_flags=flags,
        component_trees=trees,component_orderings=final_orders,unique_prefix_trees=len(set(prefix_cache.values())),
        combined_dispatch_sources={p:sha(src(p)) for p in joint},identical_component_sources=equal,
        V286_matching_transport_methods=sorted(donor_api), bound_runtime_replays=bound,
        physical_h200=False),indent=2))
