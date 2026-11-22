"""Offline registration, compatibility, and resource checks for the combined serving tree."""
import argparse
import ast
import dataclasses
from collections import Counter
import hashlib
import inspect
import itertools
import json
import subprocess
import sys
import types
from pathlib import Path

import msgspec

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
SRT = "python/sglang/srt/"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
NEW_FLAGS = ("muse_sparse_prompt_exchange", "speculative_dflash_sp_prefill_context",
             "speculative_dflash_share_verify_schedule")
FLAG_FIELDS = {"parallel":("Parallel", ("enable_layernorm_sp", NEW_FLAGS[0])),
               "spec":("Spec",("speculative_moe_runner_backend",*NEW_FLAGS[1:])),
               "schedule":("Schedule",("swa_full_tokens_ratio",)), "exec_":("ExecMoe",("moe_runner_backend",))}
EXPECTED = dict(enable_layernorm_sp=True, moe_runner_backend="humming", speculative_moe_runner_backend="marlin",
                swa_full_tokens_ratio=0.25)


def git(*args):
    return subprocess.check_output(["git","-C",str(STAGE),*args])


def source(tree,path):
    return git("show",tree+":"+SRT+path).decode()


def module(name,source_code):
    mod=types.ModuleType(name);mod.__file__=name+".py";sys.modules[name]=mod
    exec(compile(source_code,mod.__file__,"exec",dont_inherit=True),mod.__dict__)
    return mod


def declarations(raw,name):
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name==name)
    return [n for n in cls.body if isinstance(n,ast.AnnAssign)]


def public_signature(tree,path,cls):
    # Keep all original field declarations, defaults, decorators and ordering.
    # Deferred annotations avoid loading unrelated accelerator-only modules.
    fields=declarations(source(tree,path),cls)
    body="\n".join("    "+ast.unparse(n) for n in fields)
    mod=module("native_"+cls,"from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass "+cls+":\n"+body)
    native=getattr(mod,cls)
    values=[object() for _ in fields]
    sample=native(*values)
    assert all(getattr(sample,n.target.id) is v for n,v in zip(fields,values,strict=True))
    return tuple(inspect.signature(native).parameters)


def parser(tree):
    group="sglang.srt.arg_groups."
    def load(name):
        return module(group+name,source(tree,"arg_groups/"+name.replace(".","/")+".py"))
    # Placeholder for the hardware runtime. No parser should query a device.
    module("sglang.srt.runtime_context","def get_exec(): raise AssertionError('unexpected runtime initialization')")
    common_code=source(tree,"utils/common.py")
    human=next(n for n in ast.parse(common_code).body if getattr(n,"name",None)=="human_readable_int")
    module("sglang.srt.utils.common","import re,argparse\nfrom decimal import Decimal\n"+ast.unparse(human))
    module("sglang.srt.model_executor.cuda_graph_config",source(tree,"model_executor/cuda_graph_config.py"))
    util=load("arg_utils");load("choices")
    prior=load("field_order").POSITIONAL_FIELD_ORDER
    gather=module(group+"fields",source(tree,"arg_groups/fields/__init__.py")).collect_input_fields
    p=argparse.ArgumentParser()
    schemas=[]
    for filename,(name,wanted) in FLAG_FIELDS.items():
        cls=getattr(load("fields."+filename),name);schemas.append(cls)
        members={f.name for f in msgspec.structs.fields(cls)}
        util.add_cli_args_from_dataclass(p,cls,fields=[f for f in wanted if f in members])
    typed,defaults,ns=gather(schemas)
    return prior,p,tuple(typed),defaults,ns


def platform_matrix(tree):
    # Invoke the actual hook/validator with hardware and configuration accessors supplied.
    sp=source(tree,"layers/layernorm_sp.py")
    needed=("MUSE_SP_ARCHITECTURES","SP_SUPPORTED_ARCHITECTURES","is_muse_spark_architecture")
    nodes=[n for n in ast.parse(sp).body if (getattr(n,"name",None) in needed or
            isinstance(n,ast.Assign) and getattr(n.targets[0],"id",None) in needed)]
    module("sglang.srt.layers.layernorm_sp","from typing import Optional\n"+"\n".join(ast.unparse(n) for n in nodes))
    raw=source(tree,"arg_groups/layernorm_sp_hook.py")
    nodes=[n for n in ast.parse(raw).body if getattr(n,"name",None) in ("handle_layernorm_sp","validate_layernorm_sp")]
    hooks=module("actual_sp_hooks","from __future__ import annotations\n"+"\n".join(ast.unparse(n) for n in nodes))
    def fake(**kwargs):return types.SimpleNamespace(**kwargs)
    hooks.resolving_view=lambda cfg:cfg
    hooks.model_config_of=lambda cfg:fake(hf_config=fake(architectures=["MuseSparkForConditionalGeneration"]))
    hooks.attn_dp_enabled_of=lambda cfg:cfg.attn_dp
    module("sglang.srt.environ","").envs=fake(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=fake(get=lambda:False))
    base=dict(enable_layernorm_sp=True, moe_runner_backend="humming", tp_size=8,ep_size=8,pp_size=1,
              speculative_algorithm="DFLASH",quantization="modelopt_fp4",moe_a2a_backend="none",attn_dp=False,
              attn_cp_size=1,dcp_size=1,enable_prefill_cp=False,enable_attn_tp_input_scattered=False,
              enable_mixed_chunk=False,enable_deterministic_inference=False)
    common=sys.modules["sglang.srt.utils.common"]
    results=[]
    for arch in ("cpu","rocm","sm80","sm90","sm100"):
        cuda=arch.startswith("sm")
        common.is_cuda=lambda:cuda
        common.get_device_capability=lambda:{"sm80":(8,0),"sm90":(9,0),"sm100":(10,0)}[arch]
        for edits in ({},dict(tp_size=4,ep_size=4),dict(tp_size=2,ep_size=2),dict(ep_size=4),dict(pp_size=2),
                      dict(attn_dp=True),dict(attn_cp_size=2),dict(dcp_size=2),dict(enable_prefill_cp=True),
                      dict(enable_mixed_chunk=True),dict(enable_attn_tp_input_scattered=True),
                      dict(enable_deterministic_inference=True),dict(quantization="other"),dict(moe_a2a_backend="other"),
                      dict(speculative_algorithm="EAGLE"),dict(enable_layernorm_sp=False)):
            try:hooks.handle_layernorm_sp(fake(**(base|edits)))
            except ValueError:ok=False
            else:ok=True
            expect=(arch=="sm90" and edits in ({},dict(tp_size=4,ep_size=4))) or edits==dict(enable_layernorm_sp=False)
            assert ok==expect,(arch,edits,ok)
            results.append(ok)
    return dict(case_count=len(results),accepted=sum(results),rejected=len(results)-sum(results))


def donor_changes(tree):
    manifests={n:json.loads(next(HERE.parent.glob(f"v{n}_*/validation.json")).read_text()) for n in (168,169,171,172,174)}
    def changes(to):
        edits=Counter()
        for line in git("diff","--minimal","--unified=0",manifests[168]["tree"],to).decode().splitlines():
            if line.startswith("diff --git"):
                path=line.split(" b/")[-1]
            elif line.startswith(("+","-")) and not line.startswith(("+++","---")):
                edits[(path,line)]+=1
        return edits
    expected=sum((changes(manifests[n]["tree"]) for n in (169,171,172,174)),Counter())
    combined=changes(tree)
    assert not expected-combined
    assert combined-expected==Counter({(SRT+"model_executor/forward_batch_info.py","+"):1})
    return dict(donor_operations=sum(expected.values()),extra_operations=sum((combined-expected).values()),
                extra="empty line around the appended ForwardBatch field",
                donors={f"V{n}":m["patch_sha256"] for n,m in manifests.items()})

def pinned_calls():
    refs=git("grep","-l","ForwardBatch(",BASE,"--","*.py").decode().splitlines()
    counts=[]
    for ref in refs:
        for f in ast.walk(ast.parse(git("show",ref))):
            if isinstance(f,ast.Call) and getattr(f.func,"id",getattr(f.func,"attr",""))=="ForwardBatch":
                counts.append(len(f.args))
    assert counts and not any(counts)
    return len(counts)


def main():
    tree=git("write-tree").decode().strip()
    old_order,_,old_record,_,_=parser(BASE)
    order,p,record,defaults,ns=parser(tree)
    assert order==old_order+NEW_FLAGS
    assert record==old_record+NEW_FLAGS
    assert all(defaults[name] is False for name in NEW_FLAGS)
    assert [ns[name] for name in NEW_FLAGS]==["parallel","spec","spec"]
    argv=["--enable-layernorm-sp","--moe-runner-backend","humming","--speculative-moe-runner-backend","marlin",
          "--swa-full-tokens-ratio","0.25"]
    cases=0
    for n in range(4):
        for flags in itertools.permutations(NEW_FLAGS,n):
            toggles=["--"+f.replace("_","-") for f in flags]
            for args in (argv+toggles,toggles+argv):
                d=vars(p.parse_args(args))
                assert d==EXPECTED|{name:name in flags for name in NEW_FLAGS}
                cases+=1
    assert all(not getattr(p.parse_args([]),name) for name in NEW_FLAGS)
    cls_views={}
    for path,cls,new in (("model_executor/forward_batch_info.py","ForwardBatch","_fa3_verify_schedules"),
                         ("layers/logits_processor.py","LogitsProcessorOutput","dflash_prefill_context_done")):
        old=public_signature(BASE,path,cls);now=public_signature(tree,path,cls)
        assert now==(*old,new)
        cls_views[cls]=dict(original_positional_parameters=len(old),tail=now[-2:])
    vendor=json.loads(next(HERE.parent.glob("v169_*/sm90a.json")).read_text())
    generated=json.loads(next(HERE.parent.glob("v169_*/generated.json")).read_text())
    v174=json.loads(next(HERE.parent.glob("v174_*/sm90.json")).read_text())
    for name,digest in v174["build_inputs"]["new"].items():
        blob=git("show",tree+":python/sglang/kernels/jit/csrc/muse_spark/"+name)
        assert hashlib.sha256(blob).hexdigest()==digest
    info=v174["builds"]["new"]["tma.sm_90a"]["kernels"]
    selected={k:{key:info[k][key] for key in ("shared_bytes","registers","spill_store_bytes","spill_load_bytes")}
              for k in ("multirow_W13","multirow_W2")}
    extra=[(k,v) for k,v in vendor["kernels"].items() if not k.endswith("_176")]
    return dict(tree=tree,patch_sha256=hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest(),msgspec=msgspec.__version__,
        flagged_cli_permutations=cases,original_server_positionals=len(old_order),
        relevant_native_namespace_fields_tested=len(record),default_false=list(NEW_FLAGS),dataclass_signatures=cls_views,
        platform_matrix=platform_matrix(tree),medium_hopper=selected,
        optional_Humming_AOT_cubin_bytes=sum(v["cubin_bytes"] for k,v in extra),
        prompt_shared_bytes_by_tile={k:v["shared_bytes"] for k,v in generated.items()},
        donor_line_comparison=donor_changes(tree),pinned_direct_ForwardBatch_calls_all_keywords=pinned_calls(),
        scored_v64_gpu_memory=json.loads((HERE.parents[1]/"runs/abd217baf0ed49fd.json").read_text())["result"]["gpu_memory"],
        medium_routed_tensor_bytes={str(users):{"bf16_activations_and_outputs":users*8*8*(4096+4096)*2,
                     "local_int32_expert_plan":32*(users*8*8+1)*4} for users in (16,32)},
        hardware_test=False)


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
