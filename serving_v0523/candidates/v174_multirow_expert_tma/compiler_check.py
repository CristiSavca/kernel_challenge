"""Compare matched native Hopper wrappers for V173 and V174."""
import json
from pathlib import Path
import runpy
import sys

PARENT = Path(__file__).resolve().parent.parent / "v173_medium_expert_tma/compiler_check.py"
tools = runpy.run_path(str(PARENT))
analyze, sha, old_label = (tools[k] for k in ("analyze", "sha", "label"))


def label(name):
    if "multirow_projection" in name:
        return "multirow_" + ("W13" if "ILb1" in name else "W2")
    return old_label(name)


def inspect(folder):
    report = dict(nvcc="12.9.86", hardware=False, build_inputs={}, builds={}, retained={})
    for role in ("prior", "new"):
        tree = folder / role
        headers = tree / "python/sglang/kernels/jit/csrc/muse_spark"
        report["build_inputs"][role] = {f.name: sha(f.read_bytes()) for f in sorted(headers.glob("*.cuh"))}
        builds = report["builds"][role] = {}
        for mod in ("tma", "medium"):
            for arch in ("sm_90a", "sm_90"):
                key = mod + "." + arch
                files = {e: (tree/"out"/(key + "." + e)).read_bytes() for e in ("o","log","sass")}
                kernels = {}
                for body in files["sass"].decode().split("Function : ")[1:]:
                    name = body.splitlines()[0]
                    kernels[label(name)] = analyze(name, body, files['log'].decode())
                assert all(not (k["spill_store_bytes"] or k["spill_load_bytes"] or k["stack_bytes"])
                           for k in kernels.values())
                builds[key] = dict(kernels=kernels,file_sha256={e: sha(b) for e,b in files.items()},
                                   wrapper_sha256=sha((tree/"out"/(mod+".cu")).read_bytes()))
    for key, old in report["builds"]["prior"].items():
        new = report["builds"]["new"][key]["kernels"]
        old = old['kernels']
        common = old.keys() & new.keys()
        equal = {k: old[k]['machine_sha256'] == new[k]['machine_sha256'] for k in sorted(common)}
        assert all(equal.values())
        report["retained"][key] = dict(machine_identical=equal,
                                      removed_specializations=sorted(old.keys()-new.keys()),
                                      added_specializations=sorted(new.keys()-old.keys()))
        if key.startswith('tma'):
            for gate,groups in (("W13",(1,2)),("W2",(1,2,3,4))):
                loops = new["multirow_"+gate]["inner_mma_loops"]
                units = 16 if gate=="W13" else 8
                for n in groups:
                    # Every actual 128-K work loop issues the same dequantization
                    # count per packed bank, regardless of simultaneous row groups.
                    assert any(x['interesting_ops']['HMMA']==units*n and
                               x['interesting_ops']['HMUL2']==8*units for x in loops)
    return report


if __name__ == "__main__":
    print(json.dumps(inspect(Path(sys.argv[1])), indent=2))
