"""Replay the public Python launch entry and check every packed-row tile boundary."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = "python/sglang/kernels/ops/attention/paged_verify_attention.py"
PROFILE = ("experiments/candidates/muse_attention_decode/profiles/"
           "87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c.c21d391dbf604bb5.json")


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def record(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return record


def entry(source):
    nodes = ast.parse(source).body
    fn = next(x for x in nodes if getattr(x, "name", "") == "paged_verify_attention")
    stage, merge = Launch(), Launch()
    triton = SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda n, d: -(-n // d))
    scope = dict(torch=torch, triton=triton, math=math, _partition_attention=stage,
                 _merge_partitions=merge, _direct_tile_args=lambda *args: {})
    exec(compile(ast.Module([fn], type_ignores=[]), "<production-launch>", "exec"), scope)
    return scope[fn.name], stage, merge


def launch_check(parent, child):
    before, after = entry(parent), entry(child)
    stats = {"public_sizes": {}, "valid_dispatches": 0, "invalid_dispatches": 0,
             "coverage_proofs": 0, "q_gqa_pairs": 0}
    head_dim, sms = 64, 132
    # Meta tensors exercise Torch shape, dtype, contiguous and allocation semantics.
    def tensor(shape, dtype):
        return torch.empty(shape, device="meta", dtype=dtype)
    for width in range(2, 17):
        for gqa in range(1, 33):
            if not 16 <= width * gqa <= 64:
                continue
            stats["q_gqa_pairs"] += 1
            for b in range(1, 65):
                h = 2
                k = tensor((4096, 1, h, head_dim), torch.float8_e4m3fn)
                q = tensor((b * width, h * gqa, head_dim), k.dtype)
                lens = tensor((b,), torch.int32)
                cu = tensor((b + 1,), torch.int32)
                pages = tensor((b, 4096), torch.int32)
                for window in (-1, 2047):
                    for function, partial, merge in (before, after):
                        returned = function(q, k, k, pages, lens, cu, max_q=width,
                            window_left=window, softmax_scale=0.125, k_descale=None,
                            v_descale=None, out=None, num_sms=sms)
                        assert returned.shape == q.shape and returned.dtype == torch.bfloat16
                    (pgrid, pargs, pkw), (cgrid, cargs, ckw) = before[1].calls.pop(), after[1].calls.pop()
                    assert pgrid == cgrid and pkw == ckw
                    assert pargs[7].shape == cargs[7].shape and pargs[8].shape == cargs[8].shape
                    (before_grid, before_args, before_kw) = before[2].calls.pop()
                    (grid, args, kw) = after[2].calls.pop()
                    rows = args[-1]
                    assert before_args[6:] == args[6:-1] and before_kw == kw
                    assert grid[:2] == before_grid[:2] == (b, h)
                    assert before_grid[2] == width * gqa and grid[2] == -(-(width*gqa)//rows)
                    assert rows in (1, 2, 4) and (rows == 1 or math.prod(grid) >= sms)
                    if window < 0:
                        for live_q in range(width + 1):
                            actual = []
                            for tile in range(grid[2]):
                                first = tile * rows
                                if first >= live_q * gqa:
                                    continue
                                count = min(rows, live_q * gqa - first) if gqa % rows else rows
                                actual.extend(range(first, first + count))
                            assert actual == list(range(live_q * gqa))
                            stats["coverage_proofs"] += 1
                    if width == gqa == 8 and b in (1, 4, 16, 32):
                        old, new = math.prod(before_grid), math.prod(grid)
                        stats["public_sizes"][b] = dict(rows=rows, old_ctas_per_layer=old,
                            new_ctas_per_layer=new, old_ctas_62_layers=old*62,
                            new_ctas_62_layers=new*62, planning_i32_words_saved_full_grid=(old-new)*3*b*62)
                    stats["valid_dispatches"] += 1
    # Rejection must occur before any scratch allocation or kernel call.
    for width,b,gqa,qdtype in ((1,4,8,k.dtype),(17,4,2,k.dtype),(8,65,8,k.dtype),
                              (8,2,9,k.dtype),(8,8,1,k.dtype),(8,4,8,torch.bfloat16)):
        for fn, stage, merge in (before,after):
            q=tensor((b*width,2*gqa,64),qdtype)
            assert fn(q,k,k,tensor((b,4096),torch.int32),tensor((b,),torch.int32),
                      tensor((b+1,),torch.int32),max_q=width,window_left=-1,softmax_scale=0.125,
                      k_descale=None,v_descale=None,out=None,num_sms=sms) is None
            assert not stage.calls and not merge.calls
        stats["invalid_dispatches"]+=1
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    args=parser.parse_args()
    manifest=json.loads((HERE/"validation.json").read_text())
    # Reapply both published patches into a private git index if the stage no
    # longer carries a cached parent. This read does not touch its working tree.
    import tempfile,os
    local=[]
    for directory in (HERE.parent/"v235_peer_mixed_sliding_prefill",HERE):
        with tempfile.TemporaryDirectory() as tmp:
            env={**os.environ,"GIT_INDEX_FILE":str(Path(tmp)/"index")}
            def git(*cmd, data=None):
                return subprocess.check_output(["git","-C",str(args.stage),*cmd],input=data,env=env)
            git("read-tree",manifest["base"])
            git("apply","--cached","-",data=(directory/"patch.diff").read_bytes())
            local.append(git("show",":"+SOURCE).decode())
    result=launch_check(*local)
    assert local[1]==(args.stage/SOURCE).read_text()
    path=HERE.parents[2]/PROFILE
    archive=json.loads(path.read_bytes())
    result["standalone_h200_comparison"] = [
        dict(batch=e["case"]["batch"],layer=e["case"]["layer"],load=e["case"]["load"],
             total_us=float(re.search(r"([\d.]+) µs",e["report"].splitlines()[-1]).group(1)))
        for e in archive["profiles"] if e["case"]["load"].startswith("verify")]
    result["profile_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    result["full_patch_sha256"]=manifest["patch_sha256"]
    result["source_sha256"]=[hashlib.sha256(x.encode()).hexdigest() for x in local]
    result["scope"]="CPU meta tensor launch replay and integer ownership proof. Logical lane loads are not measured HBM traffic."
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
