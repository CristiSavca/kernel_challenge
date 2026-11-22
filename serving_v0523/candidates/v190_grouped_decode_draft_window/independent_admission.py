"""Run V190's actual draft startup guard with the released checkpoint config and incompatible runtime configurations."""
import ast
from collections.abc import Sequence
from contextlib import ExitStack
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as N
from unittest.mock import patch

import torch

STAGE = Path(sys.argv[1]).resolve()
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
CONFIG = "https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/resolve/46c90f412549c472941a68382d91a99e5e6372a0/config.json"
config_bytes = subprocess.check_output(["curl","-fsSL",CONFIG])
assert sha256(config_bytes).hexdigest()=="2f88d111418b4018678b8a80ed0bdf83c14b0e395cd6cc163153bce4dd8ba0ae"
raw = json.loads(config_bytes)
cfg = N(**raw)
cfg.get_text_config = lambda:cfg
env = dict(torch=torch, Enum=Enum, Sequence=Sequence)
hashes = {}


def extract(relative, names, owner=None):
    path = "python/sglang/" + relative
    local = STAGE/path
    data = local.read_bytes() if local.exists() else subprocess.check_output(["git","-C",str(STAGE),"show",PIN+":"+path])
    hashes[path] = sha256(data).hexdigest()
    nodes = ast.parse(data).body
    if owner:
        nodes = next(n for n in nodes if getattr(n,"name",None)==owner).body
    selected = [n for n in nodes if getattr(n,"name",None) in names]
    assert {n.name for n in selected}==set(names)
    exec("from __future__ import annotations\n"+ast.unparse(ast.Module(selected,[])),env)


extract("srt/layers/radix_attention.py",["AttentionType"])
extract("srt/speculative/dflash_utils.py",["get_dflash_layer_types","_get_text_config","_cfg_get","get_dflash_attention_sliding_window_size"])
extract("srt/models/dflash.py",["_get_dflash_attention_type","_get_dflash_layer_attention_params"])
extract("srt/layers/attention/flashattention_backend.py",["_draft_graph_window"],"FlashAttentionBackend")
model_mod = ModuleType("sglang.srt.models.dflash")
model_mod.DFlash2DraftModel = type("DFlash2DraftModel",(),{})
model = model_mod.DFlash2DraftModel()
model.block_size = raw["dflash_config"]["block_size"]
model.layers = []
for layer_id in range(cfg.num_hidden_layers):
    left, direction = env["_get_dflash_layer_attention_params"](cfg,layer_id)
    attn = N(head_dim=cfg.head_dim,v_head_dim=cfg.head_dim,is_cross_attention=False,attn_type=direction,sliding_window_size=left)
    model.layers.append(N(self_attn=N(attn=attn)))
assert len(model.layers)==5 and all(layer.self_attn.attn.sliding_window_size==2047 for layer in model.layers)
platform = N(is_sm90=True)
spec = N(speculative_draft_window_size=None)
opt = N(enable_deterministic_inference=False)
parallel = N(attn_dp_enabled=False,attn_dcp_size=1,enable_prefill_cp=False)
deterministic = N(get=lambda:False)
env.update(get_spec=lambda:spec, get_exec=lambda:N(deterministic=opt), get_parallel=lambda:parallel,
           get_platform=lambda:platform, envs=N(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=deterministic))
translate=N(is_translating=False)
backend=N(is_draft_runner=True,fa_impl_ver=4,page_size=1,kv_cache_dtype=torch.bfloat16,kv_index_translator=translate,
          use_sliding_window_kv_pool=False,local_attn_builder=None,_kv_shard_pool=None,is_encoder_decoder=False,
          is_prefill_aware_swa=False,use_mla=False,topk=1,attn_cp_size=1,speculative_num_draft_tokens=model.block_size)
runner=N(model=model)
guard = env["_draft_graph_window"]
negative = [(backend,k,v) for k,v in dict(is_draft_runner=False,fa_impl_ver=3,page_size=16,kv_cache_dtype=torch.float8_e4m3fn,
            use_sliding_window_kv_pool=True,local_attn_builder=object(),_kv_shard_pool=object(),is_encoder_decoder=True,
            is_prefill_aware_swa=True,use_mla=True,topk=2,attn_cp_size=2,speculative_num_draft_tokens=model.block_size+1).items()]
negative += [(platform,"is_sm90",False),(translate,"is_translating",True),(parallel,"attn_dp_enabled",True),
             (parallel,"attn_dcp_size",2),(parallel,"enable_prefill_cp",True),(opt,"enable_deterministic_inference",True),
             (spec,"speculative_draft_window_size",2048),(deterministic,"get",lambda:True),(runner,"model",object()),
             (model,"layers",[])]
for layer in model.layers:
    for name,value in dict(head_dim=64,v_head_dim=64,is_cross_attention=True,attn_type=env["AttentionType"].DECODER,
                           sliding_window_size=-1).items():
        negative.append((layer.self_attn.attn,name,value))
with patch.dict(sys.modules, {model_mod.__name__:model_mod}):
    assert guard(backend,runner)==2047
    for owner,name,value in negative:
        with patch.object(owner,name,value):
            assert guard(backend,runner)==-1, name
    with ExitStack() as mixed:
        for i,layer in enumerate(model.layers):
            mixed.enter_context(patch.object(layer.self_attn.attn,"sliding_window_size",128*(i+1)-1))
        assert guard(backend,runner)==639
print(json.dumps(dict(config=CONFIG,config_sha256=sha256(config_bytes).hexdigest(),source_sha256=hashes,
    published_model_window=2047,published_block=model.block_size,adverse_cases=len(negative),mixed_window_case=639,
    scope="Native layer parameter extraction and V190 backend guard run with a symbolic DFlash2 class/runner and CPU config. Physical checkpoint weights, CUDA and graph launch are outside this fixture."),indent=2))
