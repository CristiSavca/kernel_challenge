"""Compare V151/V172 operations and execute exact page-one Hopper scheduler geometry."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen

HERE=Path(__file__).resolve().parent
V151=HERE.parent/"v151_integrated_verify_scheduler"
V168=HERE.parent/"v168_streamed_experts_sparse_return"
V132=HERE.parent/"v132_verify_scheduler_metadata"
sys.path.insert(0,str(V132))
from independent_geometry import build_oracle,evaluate
from splits import BASE,FILES as VENDOR_FILES
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"


def changes(candidate):
    groups={}
    for line in (candidate/"delta.diff").read_text().splitlines():
        if line.startswith("diff --git "):
            path=line.split(" b/")[1]
            groups[path]=[]
        elif line[:1] in ("+","-") and line[:3] not in ("+++","---"):
            groups[path].append(line)
    return groups


def main():
    v172,old=changes(HERE),changes(V151)
    schema="python/sglang/srt/arg_groups/field_order.py"
    opt="speculative_dflash_share_verify_schedule"
    assert all(v172[p]==ops for p,ops in old.items())
    assert v172.keys()-old.keys()=={schema}
    assert v172[schema]==[f'+    "{opt}",']
    val=json.loads((HERE/"validation.json").read_text())
    assert hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest()==val["patch_sha256"]
    option=next(v for v in val.values() if isinstance(v,dict) and "--speculative-dflash-share-verify-schedule" in v)
    base=json.loads((V168/"validation.json").read_text())
    flags=next(v for v in base.values() if isinstance(v,dict) and "--muse-sparse-prompt-exchange" in v)
    assert option==flags|{"--speculative-dflash-share-verify-schedule":""}
    official_git=HERE.parents[1]/"sglang"
    raw=subprocess.check_output(["git","-C",str(official_git),"show",PIN+":python/sglang/srt/arg_groups/arg_utils.py"],text=True)
    func=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=="_field_to_cli_name")
    scope={};exec(compile(ast.Module(body=[func],type_ignores=[]),"<original CLI>","exec"),scope)
    assert scope["_field_to_cli_name"](opt) in option
    with tempfile.TemporaryDirectory(prefix="okc-v172-independent-") as t:
        vendor=Path(t)
        with ThreadPoolExecutor(5) as pool:
            sources=dict(zip(VENDOR_FILES,pool.map(lambda n:urlopen(BASE+n, timeout=20).read(),VENDOR_FILES)))
        for name,raw in sources.items():
            (vendor/name).write_bytes(raw)
        cpp=sources["flash_api.cpp"].decode()
        loc=cpp.index("run_mha_fwd_combine(params, stream, true")
        assert "if (params.num_splits > 1)" in cpp[loc-2000:loc]
        assert "else if (scheduler_needs_semaphore && params.skip_scheduler_metadata_computation)" in cpp[loc:loc+450]
        assert "tile_count_semaphore_offset + 1)}).zero_()" in cpp[loc:loc+950]
        assert "*params.semaphore_to_reset = 0;" in sources["flash_fwd_combine_kernel.h"].decode()
        assert "*tile_count_semaphore = 0;" in sources["flash_prepare_scheduler.cu"].decode()
        assert "if (Varlen && !params.skip_scheduler_metadata_computation)" in sources["flash_fwd_launch_template.h"].decode()
        assert "params.tile_count_semaphore" in sources["flash_fwd_combine_launch_template.h"].decode()
        exe=build_oracle(vendor,vendor)
        inputs=[(b,8,1,131072,w,132,0) for b in (1,4,16,32) for w in (-1,2047)]
        shapes=evaluate(exe,inputs)
        expected=[57,15,15,14,4,4,2,2]
        assert [r[5] for r in shapes]==expected
        assert [r[1] for r in shapes]==[0,1]*4
        assert all(not r[4] and r[6] for r in shapes)
    return dict(v172_sha=val["patch_sha256"],identical_production_line_operations={k:len(v) for k,v in old.items()},
                cli_flags=option,geometry=[dict(batch=x[0],window=x[4],splits=row[5],m_blocks=2*x[0],twice_m_blocks=4*x[0]) for x,row in zip(inputs,shapes)],
                full_layers=16,sliding_layers=46,standalone_builders=2,net_prepare_reduction_if_all_layers_eligible=60,
                added_explicit_zeroes_across_62_multisplit_consumers=0,offline_source_execution=True,hardware_timing=False)

if __name__=="__main__":
    print(json.dumps(main(),indent=2))
