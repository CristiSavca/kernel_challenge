"""Narrow diagnostics/decision preservation checks, without torch or CUDA."""
import ast
import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v31_build", HERE / "build.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def literal(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def source_identity():
    p = (HERE / "parent.py").read_text(); c = (HERE / "submission.py").read_text()
    assert b.sha(p) == b.PARENT_SHA
    built, edits = b.build(); assert built == c
    for e in reversed(edits):
        assert c.count(e["new"]) == 1
        c = c.replace(e["new"], e["old"])
    assert c == p
    c = (HERE / "submission.py").read_text()
    pcu, ccu = literal(p, "bfp_SK2_SRC"), literal(c, "bfp_SK2_SRC")
    assert ccu.replace(b.HOST_HELPER + "\n", "").replace(b.HOST_CALL, "") == pcu
    assert ccu.index("      return;\n    }\n" + b.HOST_CALL) > 0
    assert "v31_reported.exchange(true, std::memory_order_relaxed)" in ccu
    # Timing expressions/call ordering are restored exactly, including outer failures and choices.
    assert function(c, "_tune").count("_graph_time(") == function(p, "_tune").count("_graph_time(")
    for name in ["bfp_sk2_compress", "bfp_sk2_prepare", "bfp__static_choice", "bfp__candidates", "bfp__run_sk2", "kernel_fn"]:
        assert function(p, name) == function(c, name), name
    return {"parent_sha256": b.sha(p), "candidate_sha256": b.sha(c),
            "full_restoration": True, "all_device_source_identical": True,
            "resource_report_after_caponly_return": True}


class Scalar:
    def __init__(self, x): self.x = x
    def __truediv__(self, other): return Scalar(self.x / other.x)
    def item(self): return self.x


class Tensor:
    shape = (4, 8)
    device = "mock-cuda"
    def __init__(self, error=0): self.error = error
    def to(self, *_): return self
    def double(self): return self
    @property
    def T(self): return self
    def __getitem__(self, _): return self
    def __matmul__(self, _): return Tensor(1)
    def __sub__(self, _): return Tensor(self.error)
    def norm(self): return Scalar(self.error)


def selftest_decisions():
    outputs = []
    for filename in ["parent.py", "submission.py"]:
        s = (HERE / filename).read_text()
        calls = []
        def run(ch, x, layer, m, nn, k, state):
            calls.append((layer, m, ch))
            if layer == "launch": raise RuntimeError("synthetic resource failure")
            return Tensor({"o": 0.001, "bad": 0.02, "nan": float("nan")}.get(layer, 0.001))
        env = {"torch": SimpleNamespace(
                   Generator=lambda **_: SimpleNamespace(manual_seed=lambda _: None),
                   randn=lambda *_, **kw: Tensor(), bfloat16="bf16",
                   cuda=SimpleNamespace(synchronize=lambda: None)),
               "bfp_SELFTEST_M": (1,), "bfp__run": run,
               "bfp__candidates": lambda *args: [("sk2",)],
               "bfp__route_key": lambda layer,m,kind: (layer,kind,"fixture",8)}
        if filename == "submission.py": exec(b.PY_HELPER, env)
        exec(function(s, "bfp__selftest"), env)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            env["bfp__selftest"]({k:Tensor() for k in ["o", "bad", "launch", "nan"]}, state)
        outputs.append((state["disabled"], calls, output.getvalue()))
    assert outputs[0][:2] == outputs[1][:2]
    rows = [json.loads(x[5:]) for x in outputs[1][2].splitlines()]
    by = {r["layer"]: r for r in rows if r["kind"] == "selftest"}
    assert by["o"]["passed"] and not by["bad"]["passed"]
    assert by["launch"]["exception"] == "RuntimeError: synthetic resource failure"
    assert by["launch"]["error"] is None and by["nan"]["error"] is None
    assert len(next(r["keys"] for r in rows if r["kind"] == "disabled")) == 3
    return {"same_calls_and_disabled_set": True, "directed_modes": ["pass", "numerical_fail", "exception", "nan"],
            "observed_rows": rows}


def python_budget():
    env={}; exec(b.PY_HELPER, env)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        for i in range(500): env["_v31_diag_emit"]("selftest", index=i, exception="x"*240)
    log=out.getvalue()
    assert len(log.encode()) <= 4096 and log.count("V31P_STOP") == 1
    with contextlib.redirect_stdout(io.StringIO()) as out2:
        for i in range(24): env["_v31_diag_emit"]("outer", m=i, layer="qkv", us=[1,2,3,4], pick="bfp")
    assert out2.getvalue().count('"kind":"outer"') == 24 and "STOP" not in out2.getvalue()
    return {"selftest_bytes": len(log.encode()), "one_selftest_stop": True,
            "outer_24_rows_survive_exhausted_selftest_budget": True}


def outer_decisions():
    cases = [("qkv", [3.,2.,1.]), ("qkv", [RuntimeError("bfp rejected"),4.,5.]),
             ("pr", [3.,4.,5.,1.]), ("pr", [3.,4.,5.,RuntimeError("fused rejected")])]
    for name, times in cases:
        observed=[]
        for filename in ["parent.py", "submission.py"]:
            src=function((HERE/filename).read_text(),"_tune")
            start=src.index('            if st.get("bfp") is not None:')
            end=src.index('\n            if log:',start)
            code=textwrap.dedent(src[start:end])
            calls=[]
            def timing(fn, flush, reps):
                v=times[len(calls)];calls.append(reps)
                if isinstance(v,Exception):raise v
                return v
            env={"st":{"bfp":object(),"mm_pick":{}},"T":16,"name":name,"w":Tensor(),
                 "xd":object(),"K":8,"flush":None,"_graph_time":timing,
                 "bfp_fused_pooler_router":lambda *args:None,"v31_outer_errors":{}}
            if filename=="submission.py":exec(b.PY_HELPER,env)
            with contextlib.redirect_stdout(io.StringIO()) as out:exec(code,env)
            observed.append((env["st"]["mm_pick"],calls,out.getvalue()))
        assert observed[0][:2]==observed[1][:2]
        record=json.loads(observed[1][2][5:])
        assert record["kind"]=="outer" and len(record["us"])==4
        if any(isinstance(v,Exception) for v in times):assert "ex" in record
    return {"actual_selection_block_cases":len(cases),"same_timer_calls_and_pick":True,
            "bfp_and_fused_exception_records":True}


def host_mock():
    header = r'''
#include <cstddef>
using cudaError_t=int;
struct cudaFuncAttributes { int numRegs=0; size_t localSizeBytes=0,sharedSizeBytes=0; int maxDynamicSharedSizeBytes=0; };
int queries=0,mode=0;
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*a,const void*){
  ++queries;if(mode)return 7;a->numRegs=91;a->localSizeBytes=16;a->sharedSizeBytes=1024;a->maxDynamicSharedSizeBytes=231424;return 0;}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*o,const void*,int,size_t){
  ++queries;*o=mode?-1:1;return mode?9:0;}
'''
    main = r'''
int main(){v31_resource("t=fixture-success",nullptr,160,180464);mode=1;
v31_resource("t=fixture-errors",nullptr,160,180464);mode=0;
for(int i=0;i<500;i++)v31_resource("t=budget-fixture",nullptr,160,180464);
  const int q=queries;v31_resource("t=after-stop",nullptr,160,180464);
if(q!=queries||!v31_resource_stopped||v31_resource_bytes>4096)return 2;return 0;}
'''
    with tempfile.TemporaryDirectory(prefix="v31-host-") as td:
        p=Path(td)/"a.cpp"; e=Path(td)/"a"; p.write_text(header+b.HOST_HELPER+main)
        subprocess.run(["clang++", "-std=c++17", str(p), "-o", str(e)], check=True, capture_output=True)
        result=subprocess.run([str(e)],capture_output=True,text=True,check=True)
    assert "attr=91,16,1024,231424 occ=1 api=0,0" in result.stderr
    assert "occ=-1 api=7,9" in result.stderr
    assert result.stderr.count("V31R_STOP") == 1
    assert len(result.stderr.encode())<=4096
    return {"bytes":len(result.stderr.encode()), "successful_and_failed_api_observations":True,
            "no_query_after_cap":True,"no_admission_changes":True}


def run():
    return {"status":"CPU_PASS", "identity":source_identity(), "selftest":selftest_decisions(),
            "outer_selection":outer_decisions(), "python_budget":python_budget(), "host_mock":host_mock(),
            "limits":["No CUDA compiler/device used", "Host resource observations are first-launch module evidence, not achieved occupancy",
                      "Partial/truncated diagnostics do not prove all pointers or failed keys were observed",
                      "This diagnostic is not a performance promotion attempt"]}


if __name__ == "__main__":
    result=run()
    (HERE/"cpu_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
