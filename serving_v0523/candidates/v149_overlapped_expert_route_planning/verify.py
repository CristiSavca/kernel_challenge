"""Reconstruct the exported V149 source tree and count the per-rank work."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from fnmatch import fnmatchcase
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENT="b5a37c2923f8d1ab24c1c5cf7920a21e2c086774"


def sha(b):return hashlib.sha256(b).hexdigest()


def verify(stage):
    patch=(HERE/"patch.diff").read_bytes()
    delta=(HERE/"delta.diff").read_bytes()
    parent=(HERE.parent/"v64_humming_grouped_sequence_parallel/patch.diff").read_bytes()
    score=json.loads((HERE.parent.parent/"runs/abd217baf0ed49fd.json").read_text())
    assert sha(parent)==score["result"]["patch"]["sha256"]
    flags=score["serving_flags"]
    fixed=(HERE.parent.parent/"flags").read_text().split("# Fixed,")[1].split("\n",1)[1]
    patterns=[v.strip() for v in fixed.replace("#","").replace("\n","").split(",") if v.strip()]
    assert not any(fnmatchcase(flag,pattern) for flag in flags for pattern in patterns)
    with tempfile.TemporaryDirectory() as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/"index"))
        def git(*args,content=None):
            return subprocess.check_output(["git","-C",stage,*args],env=env,input=content)
        def apply(base,payload,reverse=False):
            git("read-tree",base)
            git("apply","--cached","--whitespace=error",*(["-R"] if reverse else []),"-",content=payload)
            return git("write-tree").decode().strip()
        baseline=git("rev-parse",BASE+"^{tree}").decode().strip()
        target=apply(BASE,patch)
        assert apply(target,patch,True)==baseline and apply(BASE,parent)==PARENT
        assert apply(PARENT,delta)==target and apply(target,delta,True)==PARENT
        changed=git("diff","--name-only",PARENT,target).decode().splitlines()
        paths=git("diff","--name-only",BASE,target).decode().splitlines()
        sources={}
        for name in paths:
            assert name.startswith("python/sglang/") and "/model_loader/" not in name and name.endswith((".py",".json"))
            data=git("show",target+":"+name)
            assert data==(stage/name).read_bytes()
            sources[name]=sha(data)
            if name.endswith(".py"):compile(data,name,"exec")
        source=git("show",target+":python/sglang/srt/models/muse_spark.py").decode()
        assert source.count("route_stream = torch.cuda.Stream()")==1
        assert "moe_layer.sp_route_stream = route_stream" in source
        assert "backend.is_humming() and is_cuda() and get_device_capability()[0] == 9" in source
    added=max(len(line)-1 for line in patch.splitlines() if line.startswith(b"+") and not line.startswith(b"+++"))
    assert len(paths)==9 and len(changed)==3 and added<=400 and len(patch)<400000
    geometry=[]
    for tokens in (4096,8192,16384):
        slots=tokens*8
        pooled=tokens*4096*2
        send=pooled*7//8
        geometry.append(dict(tokens=tokens,eligible=slots>=256*256,pooled_BF16_bytes=pooled,
            pooled_gather_remote_bytes_per_rank=send,expert_return_BF16_bytes=pooled,
            score_and_id_gather_remote_bytes_per_rank=slots*8*7//8,
            planned_local_ids_sorted_experts_int32_plus_int64_slots_and_offsets_bytes=slots*16+33*4,
            illustrative_gather_ms_at_100_200_400_GB_s=[round(send/(rate*1e9)*1e3,4) for rate in (100,200,400)]))
    return dict(candidate="V149",base=BASE,parent="V64",parent_tree=PARENT,parent_sha256=sha(parent),
                patch_sha256=sha(patch),delta_sha256=sha(delta),tree=target,bytes=len(patch),delta_bytes=len(delta),
                paths=paths,changed_from_parent=changed,source_sha256=sources,launch_flags=flags,fixed_flag_patterns=len(patterns),
                maximum_added_line_bytes=added,python_compiled=sum(name.endswith(".py") for name in paths),strict_forward_reverse=True,
                official_export="okc serve diff .",physical_H200=False,judge_submitted=False,
                scored_parent_memory_GB_over_stock=score["result"]["gpu_memory"]["over_stock_gb"],
                scored_parent_frontier=score["result"]["frontier"]["score"],geometry=geometry)


if __name__=="__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve()),indent=2))
