"""Check the V297 source, standalone public patch, reversible delta and offline evidence."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
name, prior = "V297", "V295"
parent = next(here.parent.glob(prior.lower()+"_*"))
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent/"validation.json").read_bytes())
changed_expected = {"python/sglang/kernels/jit/csrc/muse_spark/" + x + ".cuh"
                    for x in ("decode_gate_up","decode_gate_up_launch","grouped_decode_mma_pair","live_decode_reduce")}
changed_expected |= {"python/sglang/srt/layers/muse_spark/" + x + ".py"
                     for x in ("decode_gate_up","decode_mma_pair","live_decode_reduce")}
def sha(data):
    return hashlib.sha256(data).hexdigest()

with tempfile.TemporaryDirectory() as scratch:
    obj = Path(scratch) / "objects"
    obj.mkdir()
    source = subprocess.check_output(["git", "-C", stage, "rev-parse", "--absolute-git-dir"], text=True).strip()
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch)/"index"), GIT_OBJECT_DIRECTORY=str(obj),
               GIT_ALTERNATE_OBJECT_DIRECTORIES=source+"/objects")
    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", stage, *args], env=env, input=data)

    def apply(tree, patch, reverse=False):
        git("read-tree", tree)
        git("apply", "--cached", "--whitespace=error-all", *(["-R"] if reverse else []), "-", data=patch)
        return git("write-tree").decode().strip()

    old = (parent/"patch.diff").read_bytes()
    full = (here/"patch.diff").read_bytes()
    delta = (here/"delta.diff").read_bytes()
    native = git("rev-parse", base+"^{tree}").decode().strip()
    previous, tree = apply(base, old), apply(base, full)
    assert before["base"] == base and sha(old) == before["patch_sha256"] and previous == before["tree"]
    assert apply(previous,delta) == tree and apply(tree,delta,True) == previous
    assert apply(tree,full,True) == native and git("diff","--no-renames","-U1",base,tree) == full
    changed = git("diff","--name-only",previous,tree).decode().splitlines()
    assert set(changed) == changed_expected
    paths = git("diff","--name-only",base,tree).decode().splitlines()
    for path in paths:
        assert path.startswith(limits["allowed_prefix"]) and not path.startswith(tuple(limits["forbidden_prefixes"]))
        assert path.endswith(tuple(limits["allowed_suffixes"]))
        entry = git("ls-tree",tree,path)
        old_entry = git("ls-tree",base,path)
        mode = old_entry.split()[0] if old_entry else b"100644"
        assert mode in (b"100644", b"100755") and entry.startswith(mode+b" blob ")
        data = git("show",tree+":"+path)
        assert data and b"\0" not in data
        if path.endswith(".py"):
            ast.parse(data)
    added = [line[1:] for line in full.decode().splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert len(full) <= limits["max_patch_bytes"] and max(map(len,added)) <= limits["max_line_chars"]
    assert not any(w in line for line in added for w in limits["forbidden_words"])
    assert not any(line.startswith(("rename ", "old mode ", "new mode ", "GIT binary patch", "Binary files"))
                   for line in full.decode().splitlines())
    flags = before["launch_flags"]
    denied = [pattern for patterns in contract["launch_flags"]["denied"].values() for pattern in patterns]
    assert not any(fnmatch.fnmatchcase(flag, pat) for flag in flags for pat in denied)
    sources = {p: sha(git("show",tree+":"+p)) for p in changed}
    def read(n):
        return json.loads((here/(n+".json")).read_bytes())
    width = read("w13_sm90a")
    assert len(width) == 5 and all(d["passes_production_instruction_check"] for d in width.values())
    assert width["w13_m16_n512"]["loops"]["16"]["drains"] == 16
    assert width["w13_m24_n512"]["loops"]["24"]["drains"] == 16
    for receipt,key in (("independent_w13_maps","source_sha256"),("independent_w13_writer","inputs")):
        for p,digest in read(receipt)[key].items():
            if p.endswith(".cuh"):
                path = next(x for x in changed if x.endswith("/"+p))
                assert sources[path] == digest
            elif (here/p).is_file():
                assert sha((here/p).read_bytes()) == digest
    ffi = read("independent_ffi")
    assert sources["python/sglang/kernels/jit/csrc/muse_spark/decode_gate_up_launch.cuh"] == ffi["launcher_sha256"]
    assert sources["python/sglang/srt/layers/muse_spark/decode_gate_up.py"] == ffi["python_sha256"]
    w2 = read("w2_sass")
    assert w2["tree"] == tree and len(w2["cases"]) == 7
    assert all(image["no_stack_or_spills"] and image["no_recurrent_local_accesses"]
               for image in w2["cases"].values())
    dispatch, decode = read("independent_dispatch"), read("ordinary_decode")
    assert dispatch["parent_tree"] == previous and dispatch["calls"] == 56
    assert decode["trees"]["candidate"] == tree and decode["compared_calls"] == 34
    assert read("w2_partials")["FP32_bit_comparisons"] == 3416064
    flags_check, server = read("flags"), read("server_config")
    assert flags_check["tree"] == server["tree"] == read("prompt_graph")["tree"] == tree
    assert flags_check["launch_flags"] == flags and flags_check["parsed_flag_orderings"] == 40320
    assert len(server["cases"]) == 17 and server["patch_sha256"] == sha(full)
    assert read("prompt_routes")["totals"]["observed_calls"] == 276
    lifetimes = read("independent_lifetimes")
    assert set(lifetimes) == set(width)
    for image, safety in lifetimes.items():
        assert safety["sha256"]["sass"] == width[image]["sass_sha256"]
        assert safety["sha256"]["cubin"] == width[image]["cubin_sha256"]
    spill = read("independent_spill_guard")
    assert spill["source_sha256"] == sources[next(x for x in changed if x.endswith("/live_decode_reduce.py"))]
    assert sum(not x["accepted"] for x in spill["samples"]) >= 3
    for source_map in (dispatch["source_sha256"], decode["hashes"], read("prompt_routes")["sources"],
                       read("prompt_graph")["source_sha256"], w2["selected_source_sha256"], server["source_sha256"]):
        for path,digest in source_map.items():
            if not path.startswith("python/"):
                path = "python/sglang/srt/" + path
            assert sha(git("show",tree+":"+path)) == digest, path
    physical = sorted(here.glob("*.json"))
    evidence={p.name:sha(p.read_bytes()) for p in physical if p.stem not in {"validation","cleanup"}}
    print(json.dumps(dict(candidate=name,parent=prior,base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,evidence_sha256=evidence,launch_flags=flags,
        max_added_line_chars=max(map(len,added)),byte_budget=limits["max_patch_bytes"],
        strict_apply_reverse_full_delta=True,official_export="okc serve diff, Git diff context 1",official_rules=rules_url,
        physical_H200=False,submitted=False), indent=2))
