"""Test the actual dispatch predicate and a two-rank asymmetric JIT failure."""

import ast
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def function(path, name, env):
    node = next(n for n in ast.walk(ast.parse(path.read_text())) if getattr(n, "name", None) == name)
    node.decorator_list = []
    class LocalImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return ast.Pass()
    node = LocalImports().visit(node)
    mod = ast.Module([ast.ImportFrom("__future__",[ast.alias("annotations")],0),node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(path),"exec"),env)
    return env[name]


def rank_test(rank, root, rendezvous):
    dist.init_process_group("gloo",init_method=f"file://{rendezvous}",rank=rank,world_size=2)
    group=SimpleNamespace(world_size=2,cpu_group=dist.group.WORLD)
    outcomes=[]
    for case in ("good","jit-one","arch-one","off-one","strict-one","app-deterministic"):
        def switch(value): return SimpleNamespace(get=lambda:value)
        def compile_module():
            if case=="jit-one" and rank==1: raise RuntimeError("synthetic local JIT failure")
        runtime=SimpleNamespace(version=SimpleNamespace(cuda="12.9"),distributed=dist,
            tensor=torch.tensor,int32=torch.int32,
            cuda=SimpleNamespace(get_device_capability=lambda _: (8,0) if rank==1 and case=="arch-one" else (9,0)))
        env=dict(torch=runtime,logging=logging,subprocess=subprocess,_expert_scaled_silu_module=compile_module,
            get_exec=lambda:SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=case=="app-deterministic")),
            envs=SimpleNamespace(SGLANG_ENABLE_MUSE_SPARK_HUMMING_GRAPH_PREFILL=switch(case!="off-one" or rank==0),
                                  SGLANG_ENABLE_DETERMINISTIC_INFERENCE=switch(case=="strict-one" and rank==0)))
        f=function(root/"srt/layers/muse_spark/muse_spark_humming_moe.py","_graph_prefill_ready",env)
        assert f(None,group) == (case=="good")
        outcomes.append(case)
    dist.destroy_process_group()


class Dispatch(Exception): pass


def selected(name):
    def mark(*args,**kw): raise Dispatch(name)
    return mark


def main(s):
    root=s/"python/sglang"
    module=root/"srt/layers/muse_spark/muse_spark_fp4_moe.py"
    rows=[(8192,True,True,False,False,"wave"),(8192,True,False,False,False,"wave"),
          (8192,False,False,False,False,"eager"),(8192,False,True,False,False,"indexed"),
          (8192,True,False,True,False,"eager"),(8192,True,False,False,True,"indexed"),
          (8191,True,False,False,False,"indexed")]
    for t,enabled,capturing,verify,compile_mode,expected in rows:
        conv=SimpleNamespace(compiler=SimpleNamespace(is_compiling=lambda:compile_mode),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda:capturing),bfloat16=torch.bfloat16,
            empty_like=torch.empty_like,contiguous_format=torch.contiguous_format)
        en=dict(torch=conv,local_humming_expert_ids=lambda layer,ids:ids,humming_decode_eligible=lambda *_:False,
            humming_expert_outputs=selected("indexed"),wave_capacity=lambda *args:128,
            grouped_prefill_waves=selected("wave"),is_in_breakable_cuda_graph=lambda:False,
            _HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256)
        shared=object()
        layer=SimpleNamespace(_post_expert_norm=shared,_post_expert_norm_weight=object(),use_humming=True,
            humming_graph_prefill=enabled,global_num_experts=256,num_local_experts=32,
            intermediate_size_per_partition=4096,_grouped_humming_and_combine=selected("eager"))
        f=function(module,"_forward_w4a16",en)
        try:
            f(layer,torch.empty((t,16),dtype=torch.bfloat16),None,torch.empty((t,8),dtype=torch.int32),shared,grouped_decode=verify)
        except Dispatch as out:
            assert out.args==(expected,), (out.args,expected)
        else:
            raise AssertionError("expected a dispatch")
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(rank_test,args=(root,str(Path(tmp)/"gloo")),nprocs=2)
    print(json.dumps(dict(dispatch_cases=len(rows),gloo_ranks=2,distributed_consensus_cases=6),indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))
