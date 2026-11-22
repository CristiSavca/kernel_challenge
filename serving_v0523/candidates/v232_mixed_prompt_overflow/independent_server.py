"""Replay V232's existing CLI registration, graph policy and actual DFLASH width inference."""
import argparse
import ast
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from numbers import Integral
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
ROOT=STAGE/"python/sglang/srt"
TARGET=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else HERE
MANIFEST=json.loads((TARGET/"validation.json").read_bytes())
assert hashlib.sha256((TARGET/"patch.diff").read_bytes()).hexdigest()==MANIFEST["patch_sha256"]
FILES={}


def read(path):
    material=subprocess.check_output(["git","-C",str(STAGE),"show",MANIFEST["tree"]+":python/sglang/srt/"+path])
    if (ROOT/path).is_file():assert (ROOT/path).read_bytes()==material,path
    FILES[path]=hashlib.sha256(material).hexdigest()
    return material.decode()


def load_defs(path,names,env,strip_imports=False):
    tree=ast.parse(read(path))
    nodes=[n for n in tree.body if getattr(n,"name","") in names]
    assert len(nodes)==len(names),(names,{getattr(n,"name","") for n in nodes})
    if strip_imports:
        class Strip(ast.NodeTransformer):
            def visit_ImportFrom(self,node): return ast.copy_location(ast.Pass(),node)
        for n in nodes: Strip().visit(n)
    exec(compile("from __future__ import annotations\n"+ast.unparse(ast.Module(nodes,[])),path,"exec"),env)


def run():
    # Genuine msgspec-native CLI declarations and argparse action builder.
    helper=HERE.parent/"v176_integrated_prompt_decode/independent_registration.py"
    sp=importlib.util.spec_from_file_location("v232_cli",helper)
    cli=importlib.util.module_from_spec(sp);sp.loader.exec_module(cli)
    cli.source=lambda _,path:read(path)
    cli.FLAG_FIELDS=dict(cli.FLAG_FIELDS)
    cli.FLAG_FIELDS["schedule"]=("Schedule",("swa_full_tokens_ratio","max_prefill_tokens","chunked_prefill_size","max_running_requests"))
    cli.FLAG_FIELDS["spec"]=("Spec",("speculative_dflash_mixed_prefill","speculative_num_draft_tokens","speculative_dflash_block_size"))
    _,parser,_,_,_=cli.parser("worktree")
    U=sys.modules["sglang.srt.arg_groups.arg_utils"]
    G=sys.modules["sglang.srt.arg_groups.fields.exec_"].ExecGraph
    fields=("cuda_graph_max_bs_prefill","cuda_graph_bs_prefill","cuda_graph_config","cuda_graph_backend_prefill",
        "cuda_graph_backend_decode","cuda_graph_max_bs_decode","cuda_graph_bs_decode","disable_cuda_graph",
        "disable_prefill_cuda_graph","disable_decode_cuda_graph","cuda_graph_max_seq_len_prefill","cuda_graph_prefill_max_context")
    U.add_cli_args_from_dataclass(parser,G,fields=list(fields))
    cli.module("sglang.srt.utils","def is_cuda(): return True")
    graphmod=sys.modules["sglang.srt.model_executor.cuda_graph_config"]
    scope=dict(Any=Any,Dict=Dict,List=List,Optional=Optional,Integral=Integral,Mapping=Mapping,dataclass=dataclass,math=math,json=json,
        logger=logging.getLogger(),current_platform=NS(is_out_of_tree=lambda:False),attn_dp_enabled_of=lambda cfg:False,
        is_record=lambda record:False,get_context=lambda:NS(server_args=None),_resolve_dflash_draft_attention_backend=lambda args:None,
        get_config=lambda *args,**kwargs:NS(dflash_config={"block_size":8},num_hidden_layers=5),
        with_phase=graphmod.with_phase,Phase=graphmod.Phase,Backend=graphmod.Backend,CudaGraphConfig=graphmod.CudaGraphConfig,
        default_cuda_graph_config=graphmod.default_cuda_graph_config)
    scope['logger'].setLevel(logging.ERROR)
    load_defs("arg_groups/model_override_base.py",{"ResolvingConfig","ResolvedView","resolving_view"},scope)
    load_defs("arg_groups/overrides.py",{"declare_resolution","max_prefill_buffer_tokens"},scope)
    exec(compile(read("arg_groups/prefill_buffer_ceiling.py"),"prefill_buffer_ceiling.py","exec"),scope)
    load_defs("arg_groups/cuda_graph_hook.py",{"parse_cuda_graph_config","generate_prefill_cuda_graph_batch_sizes",
        "generate_decode_cuda_graph_batch_sizes"},scope)
    # Recursively recover the actual checkpoint parser and typed config from the released source.
    parsefile="speculative/dflash_utils.py"; t=ast.parse(read(parsefile)); defs={getattr(n,"name",""):n for n in t.body}
    keep={"parse_dflash_draft_config"}
    while True:
        extra={n.id for key in keep for n in ast.walk(defs[key]) if isinstance(n,ast.Name) and n.id in defs and n.id}-keep
        if not extra:break
        keep|=extra
    for node in t.body:
        if isinstance(node,ast.Assign) and any(getattr(x,"id","")=="DEFAULT_DFLASH_MASK_TOKEN" for x in node.targets):
            scope["DEFAULT_DFLASH_MASK_TOKEN"]=ast.literal_eval(node.value)
    load_defs(parsefile,keep,scope)
    load_defs("arg_groups/speculative_hook.py",{"_handle_dflash"},scope,strip_imports=True)
    runtime=dict(scope)
    runtime["get_server_args"]=lambda:current[0]
    runtime["get_schedule"]=lambda:scope["resolving_view"](current[0])
    runtime["get_spec"]=runtime["get_schedule"]
    runtime["get_exec"]=lambda:NS(graph=NS(cuda_graph_config=scope["resolving_view"](current[0]).cuda_graph_config))
    runtime["get_parallel"]=lambda:NS(pp_size=1)
    load_defs("runtime_context.py",{"max_prefill_buffer_tokens"},runtime)
    current=[None]
    topo={"alignment":1}
    provider={}
    scope["register_prefill_buffer_ceiling"](lambda record, limit: min(limit,provider.get("cap",limit)))
    runtime["get_cuda_graph_batch_size_alignment"]=lambda:topo["alignment"]
    load_defs("utils/common.py",{"get_cuda_graph_max_batch_size","ceil_align","ceil_div"},runtime)
    load_defs("model_executor/runner/prefill_cuda_graph_runner.py",{"get_prefill_num_tokens_to_capture","_mixed_dflash_capture_sizes"},runtime,True)
    rows=[]
    for name,argv,expected in [
        ("standard",[],[]),("max_prefill_only",["--max-prefill-tokens","16640"],[]),
        ("graph_max_only",["--cuda-graph-max-bs-prefill","16640"],[]),
        ("intact_chunk_both",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640"],[8,16,32,64,128,256]),
        ("partial_capacity",["--max-prefill-tokens","16448","--cuda-graph-max-bs-prefill","16640"],[8,16,32,64]),
        ("limit_decode",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640","--cuda-graph-max-bs-decode","5"],[8,16,32,40]),
        ("explicit_lists_lock",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640","--cuda-graph-bs-prefill","4096","16384"],[]),
        ("full_prefill",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640","--cuda-graph-backend-prefill","full"],[]),
        ("disabled_decode",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640","--disable-decode-cuda-graph"],[]),
        ("explicit_json_priority",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640","--cuda-graph-config",
            '{"prefill":{"max_bs":16400}}'],[8,16]),
        ("low_provider",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640"],[]),
        ("aligned_overrun",["--max-prefill-tokens","16402","--cuda-graph-max-bs-prefill","16402"],[16]),
        ("draft_runner",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640"],[]),
        ("other_algorithm",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640"],[]),
        ("feature_off",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640"],[]),
        ("raw_defaults",[],[]),
        ("width4",["--max-prefill-tokens","16640","--cuda-graph-max-bs-prefill","16640","--speculative-dflash-block-size","4"],[4,8,16,32,64,128]),
    ]:
        topo["alignment"]=16 if name=="aligned_overrun" else 1 if name=="width4" else 8
        provider.clear()
        if name=="low_provider":provider["cap"]=16384
        mixed=[] if name=="feature_off" else ["--speculative-dflash-mixed-prefill"]
        limit=[] if name=="raw_defaults" else ["--max-running-requests","32"]
        parsed=vars(parser.parse_args([*mixed,*limit,"--chunked-prefill-size","16384",*argv]))
        parsed.setdefault("disable_cuda_graph",False)
        args=NS(**parsed)
        args.device,args.pp_size,args.attn_cp_size="cuda",1,1
        args.speculative_algorithm,args.disable_cuda_graph_padding="DFLASH",False
        args.speculative_draft_model_path,args.speculative_draft_model_revision="public-width-eight-fixture",None
        args.json_model_override_args,args.trust_remote_code="{}",False
        args.enable_dynamic_chunking=False
        args.speculative_eagle_topk=args.speculative_num_steps=args.speculative_draft_window_size=None
        current[0]=args
        scope["parse_cuda_graph_config"](args)
        cfg=scope["resolving_view"](args)
        graph=copy.deepcopy(cfg.cuda_graph_config)
        assert cfg.chunked_prefill_size==16384
        # Native memory hook defaults non-MLA prefill max_bs to the chunk and generates
        # both bs lists before the DFLASH resolver. Public decode buckets end at 32.
        if graph.prefill.max_bs is None: graph.prefill.max_bs=cfg.chunked_prefill_size
        if graph.prefill.bs is None:graph.prefill.bs=scope["generate_prefill_cuda_graph_batch_sizes"](graph.prefill.max_bs)
        if graph.decode.bs is None:graph.decode.bs=scope["generate_decode_cuda_graph_batch_sizes"](args,graph.decode.max_bs or 32)
        scope["declare_resolution"](args,"memory_fixture",cuda_graph_config=graph)
        before=tuple(graph.prefill.bs)
        scope["_handle_dflash"](args)
        graph=cfg.cuda_graph_config
        assert tuple(graph.prefill.bs)==before, "DFLASH modified existing user-declared grid"
        runner=NS(is_draft_worker=name=="draft_runner",spec_algorithm=NS(is_dflash=lambda:name!="other_algorithm"))
        native=runtime["get_prefill_num_tokens_to_capture"](graph.prefill.bs)
        added,physical=runtime["_mixed_dflash_capture_sizes"](runner,native)
        new=sorted(set(added)-set(native))
        new=[x-16384 for x in new]
        assert new==expected,(name,new,expected,graph.decode.bs)
        first=scope["max_prefill_buffer_tokens"](args)
        second=runtime["max_prefill_buffer_tokens"]()
        assert first==second,(first,second)
        rows.append(dict(name=name,argv=argv,extras=new,configured_graph_max=graph.prefill.max_bs,
            max_prefill_tokens=cfg.max_prefill_tokens,prepublish_and_runtime_ceiling=first,
            target_mixed_capacity=physical, alignment=topo["alignment"],
            max_running_requests=cfg.max_running_requests,block_size=cfg.speculative_num_draft_tokens,locked_bs=(graphmod.Phase.PREFILL,"bs") in args._cuda_graph_config_locked))
    return rows,cli.msgspec.__version__


if __name__=="__main__":
    cases,version=run()
    print(json.dumps(dict(candidate=MANIFEST["candidate"],patch_sha256=MANIFEST["patch_sha256"],tree=MANIFEST["tree"],
        msgspec=version,cases=cases,source_sha256=FILES,
        checkpoint_fetch="Synthetic checkpoint metadata with released DFLASH width 8; actual pinned config parser and width resolver.",
        physical_h200=False),indent=2))
