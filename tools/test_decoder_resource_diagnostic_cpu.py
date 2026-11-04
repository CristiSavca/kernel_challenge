#!/usr/bin/env python3
"""Host-only conformance/lifecycle/log-bound proof; no CUDA execution or timing."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

import build_decoder_resource_diagnostic as builder

ROOT = builder.ROOT


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sk2_host_test():
    preamble = r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <string>
struct cudaFuncAttributes {
  int numRegs=0; size_t localSizeBytes=0, sharedSizeBytes=0;
  int maxDynamicSharedSizeBytes=0, maxThreadsPerBlock=0;
};
enum { cudaDevAttrMultiProcessorCount = 1 };
static int queries=0, error=0;
int cudaGetDevice(int* p) { ++queries; *p=0; return error; }
int cudaDeviceGetAttribute(int* p, int, int) { ++queries; *p=132; return error; }
template<class K> int cudaFuncGetAttributes(cudaFuncAttributes* p, K) {
  ++queries; p->numRegs=144; p->sharedSizeBytes=1024; p->localSizeBytes=32;
  p->maxDynamicSharedSizeBytes=231424; p->maxThreadsPerBlock=1024; return error;
}
template<class K> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* p,K,int,size_t) {
  ++queries; *p=1; return error;
}
'''
    exact_call = r'''
void exact_call(int M, int pmax, int mode, int pdl) {
  constexpr int MT=1,TOKR=8,NT=3,NWR=1,WSK=4,UPW=2,NSTAGE=3,NACC=1,CL=1;
  constexpr bool COMP=true,HL=false,PK=true,FUSE=false,BND=false;
  constexpr int NW=NWR*WSK,NESC=3,R=NWR*NT*8;
  using OutT=uint16_t;
  const int N=2304,K=8192,kern=0;
  const size_t smem=200000+(size_t)pmax*8;
  struct {int mode;} ta{mode};
  struct X {int get_device() const{return 0;}} x;
''' + builder.SK2_CALL + "\n}\n"
    main = r'''
int main() {
  using namespace sk2_resource_diag;
  report(0,0,"disabled",288,200000); assert(queries==0 && count==0);
  sk2_set_diag(1,1);
  report(0,0,"first",288,200000); assert(queries==4 && count==1);
  report(0,0,"first",288,200000); assert(queries==4 && count==1);
  sk2_set_diag(0,2); report(0,0,"off",288,200000); assert(queries==4 && count==1);
  sk2_set_diag(1,2); report(0,0,"first",288,200000); assert(queries==4 && count==1);
  exact_call(1,12,0,2); exact_call(1,12,0,2);
  exact_call(8,12,0,2); exact_call(1,16,0,2); exact_call(1,12,1,2); exact_call(1,12,0,0);
  assert(queries==24 && count==6);
  error=7; report(0,0,"api_error",288,200000); assert(queries==26 && count==7);
  error=0;
  for(int i=0;i<1000;++i) {
    std::string key="unique_"+std::to_string(i)+std::string(180,'x');
    report(0,0,key.c_str(),288,200000);
  }
  assert(exhausted && count<=64 && used<=8192);
  const int frozen_queries=queries, frozen_count=count; const size_t frozen_bytes=used;
  sk2_set_diag(0,3); sk2_set_diag(1,3);
  report(0,0,"after_overflow",288,200000);
  assert(queries==frozen_queries && count==frozen_count && used==frozen_bytes);
  std::printf("{\"bytes\":%zu,\"records\":%d,\"queries\":%d}\n",used,count,queries);
}
'''
    with tempfile.TemporaryDirectory(prefix="okc-resource-host-") as scratch:
        source = Path(scratch) / "test.cpp"
        binary = Path(scratch) / "test"
        source.write_text(preamble + builder.SK2_HOST + exact_call + main)
        compile_result = subprocess.run(["clang++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], capture_output=True, text=True)
        assert compile_result.returncode == 0, compile_result.stderr
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert len(result.stderr.encode()) <= 8192
    assert result.stderr.count("SK2R_SCHEMA") == 1
    assert result.stderr.count("SK2R_STOP") == 1
    assert sum(line.startswith("SK2R g=") and " t=" in line for line in result.stderr.splitlines()) == 5
    error_line = next(line for line in result.stderr.splitlines() if "api_error" in line)
    assert "a=-1,-1,-1,-1,-1 occ=-1,-1" in error_line, error_line
    assert "err=7,-1,7,-1" in error_line, error_line
    assert "dev=0,-1" in error_line, error_line
    report = json.loads(result.stdout)
    assert report["bytes"] == len(result.stderr.encode())
    report["kind"] = "actual_SK2_host_helper_compiled_with_mock_CUDA"
    return report


def conformance():
    import decoder_moe_resource_diag_fragment as moe
    actual = builder.CANDIDATE.read_text()
    rebuilt = builder.build(write=False)
    assert actual == rebuilt, "candidate differs from frozen deterministic builder"
    restored = actual
    # Lifecycle/end-hook edits occur after the MoE fragment; reverse those first.
    for old, new in reversed(builder.OWN_REPLACEMENTS[-2:]):
        assert restored.count(new) == 1
        restored = restored.replace(new, old, 1)
    restored = moe.reverse_moe_fragment(restored)
    for old, new in reversed(builder.OWN_REPLACEMENTS[:-2]):
        assert restored.count(new) == 1, new[:100]
        restored = restored.replace(new, old, 1)
    assert digest(restored) == builder.PARENT_SHA
    assert restored == builder.PARENT.read_text()
    # Query placement must be after the cap-only return and before original launch.
    tree = ast.parse(actual)
    def value(name):
        return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
    sk2 = value("bfp_SK2_SRC")
    parent_tree = ast.parse(builder.PARENT.read_text())
    source_hashes = {}
    for node in parent_tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if not (name.endswith("CUDA_SRC") or name == "bfp_SK2_SRC"):
            continue
        parent_value = ast.literal_eval(node.value)
        candidate_value = value(name)
        if name == "MOE_CUDA_SRC":
            candidate_value = candidate_value.replace(moe.MOE_HOST + "\n", "", 1).replace(moe.MOE_CALL + "\n", "", 1)
        if name == "bfp_SK2_SRC":
            candidate_value = candidate_value.replace(builder.SK2_HOST + "\n", "", 1).replace(builder.SK2_CALL, "", 1)
        assert candidate_value == parent_value, name + " device or original host text changed"
        source_hashes[name] = digest(parent_value)
    start = sk2.index("static void launch_sk2")
    cap = sk2.index("if (cap != nullptr)", start)
    hook = sk2.index("// RESOURCE_DIAG_SK2_CALL_BEGIN", cap)
    launch = sk2.index("C10_CUDA_CHECK(cudaLaunchKernelEx", hook)
    assert cap < sk2.index("return;", cap) < hook < launch
    # All originally emitted dispatch and all embedded device text restore exactly.
    # Generate SK2 with pure AST-extracted code; do not import/execute the candidate.
    def generated(text, sm90):
        parsed = ast.parse(text)
        names = {"bfp_SK2_CONFIGS", "bfp_SK2_PICK", "bfp_SK2_K", "bfp_SK2_SRC",
                 "SK2B", "SK2B_UN_TOKR", "SK2B1", "SK2B_O_TOKR", "_bfp__sk2_src_unfused",
                 "bfp__sk2_geom", "bfp__sk2_src"}
        nodes = []
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                nodes.append(node)
        env = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<source-generators>", "exec"), env)
        return env["bfp__sk2_src"](sm90)
    generated_hashes = {}
    for sm90 in (False, True):
        parent_gen = generated(builder.PARENT.read_text(), sm90)
        restored_gen = generated(restored, sm90)
        assert parent_gen == restored_gen
        candidate_gen = generated(actual, sm90)
        if sm90:
            candidate_gen = candidate_gen.replace(builder.SK2_HOST + "\n", "", 1).replace(builder.SK2_CALL, "", 1)
        else:
            candidate_gen = candidate_gen.replace("void sk2_set_diag(int64_t, int64_t) {}\n", "", 1)
        assert candidate_gen == parent_gen, "generated SK2 device/dispatch/launch source changed"
        generated_hashes[str(sm90)] = digest(parent_gen)
    # Assert added helpers do not use synchronization/timing/harness/file APIs.
    added = builder.SK2_HOST + builder.SK2_CALL
    for forbidden in ("cudaEvent", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaGetLastError", "cudaLaunch", "cudaMemcpy", "fopen", "getenv"):
        assert forbidden not in added, forbidden
    return {"candidate_sha256": digest(actual), "parent_sha256": digest(restored),
            "full_parent_byte_restoration": True, "generated_SK2_parent_hashes": generated_hashes,
            "normalized_embedded_CUDA_parent_hashes": source_hashes,
            "moe_fragment_sha256": digest(Path(moe.__file__).read_text())}


def lifecycle_test():
    import decoder_moe_resource_diag_fragment as moe
    parent = ast.parse(builder.PARENT.read_text())
    names = {"MOE_TINY_DEFINES", "MOE_TINY2_DEFINES", "MOE_MID_DEFINES", "_MA_SFX_IF_PRMT", "DQ7"}
    env = {n.targets[0].id: ast.literal_eval(n.value) for n in parent.body
           if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
           and n.targets[0].id in names}

    class Extension:
        def __init__(self): self.calls = []
        def sk2_set_diag(self, *args): self.calls.append(("sk2", *args))
        def okc_set_diag(self, *args): self.calls.append(("moe", *args))

    sk2 = Extension()
    env.update(bfp__ext=sk2, _MA_EXTS={})
    exec(compile(moe.PYTHON_LIFECYCLE, "<actual-lifecycle>", "exec"), env)
    output = io.StringIO()
    with contextlib.redirect_stderr(output):
        env["_resource_diag_begin"]()
        assert not env["_RESOURCE_DIAG_ARMED"]
        assert sk2.calls[-1] == ("sk2", 0, 1)
        keys = sorted(env["_resource_diag_allowed_moe_keys"]())
        assert len(keys) == 6
        modules = [Extension() for _ in keys]
        for key, ext in zip(keys, modules):
            env["_MA_EXTS"][key] = ext
            env["_resource_diag_bind_moe"](ext, key)
            assert ext.calls[-1] == ("moe", 0, 1, 320)
        env["_resource_diag_end"]()
        assert sk2.calls[-1] == ("sk2", 1, 1)
        assert all(ext.calls[-1] == ("moe", 1, 1, 320) for ext in modules)
        lazy_sk2 = Extension()
        env["_resource_diag_bind_sk2"](lazy_sk2)
        assert lazy_sk2.calls[-1] == ("sk2", 1, 1)
        # Unsupported/lazy modules must stay disabled and must not grow the registry.
        for index in range(100):
            ext = Extension()
            key = ("-DOKC_TINY=1", "-DOKC_WS=1", "-DUNSUPPORTED=" + str(index))
            env["_resource_diag_bind_moe"](ext, key)
            assert ext.calls[-1][-1] == 0
        main = Extension()
        env["_resource_diag_bind_moe"](main, ("-DOKC_WGS=1",))
        assert main.calls[-1][-1] == 0
        extra = Extension()
        env["_resource_diag_bind_moe"](extra, keys[0])
        assert extra.calls[-1][-1] == 0
        assert len(env["_RESOURCE_DIAG_MOE_MODULES"]) == 6
        assert env["_RESOURCE_DIAG_MOE_ALLOCATED"] == 1920
        env["_resource_diag_begin"]()
        assert sk2.calls[-1] == ("sk2", 0, 2)
        assert all(ext.calls[-1] == ("moe", 0, 2, 320) for ext in modules)
        assert not env["_RESOURCE_DIAG_ARMED"]  # a failed prepare has no end/arm call
        env["_resource_diag_end"]()
        assert all(ext.calls[-1] == ("moe", 1, 2, 320) for ext in modules)
    assert output.getvalue().count("MOER_STOP") == 1
    assert len(output.getvalue().encode()) <= 128
    return {"actual_python_lifecycle": True, "positive_budget_modules": 6,
            "allocated_moe_bytes": 1920, "global_notice_bytes": len(output.getvalue().encode()),
            "reprepare_disable_and_rearm": True, "new_lazy_module_control": True,
            "unsupported_module_registry_growth": 0}


def main():
    from test_decoder_moe_resource_diag_cpu import run as moe_host_test
    report = {"scope": "CPU host-source/logging only; no CUDA compilation, numeric check, or performance claim",
              "sk2_host": sk2_host_test(), "conformance": conformance(), "lifecycle": lifecycle_test(),
              "moe_host": moe_host_test(),
              "diagnostic_byte_cap": 10240}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
