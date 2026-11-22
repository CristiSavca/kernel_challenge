"""Replay the native V68 planner/gather/activation/finalizer and the real V189 optional W2 branch."""
import contextlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType
from collections import Counter

import torch

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
bank=(stage/"python/sglang/srt/layers/muse_spark/decode_mma_pair.py")
ops=ModuleType("humming.ops");humming=ModuleType("humming");humming.ops=ops
sys.modules["humming"]=humming;sys.modules["humming.ops"]=ops
device=ModuleType("sglang.srt.layers.muse_spark.decode_mma_pair")
exec(compile(bank.read_bytes(),str(bank),"exec"),device.__dict__)
sys.modules[device.__name__]=device
previous=here.parent/"v66_humming_grouped_decode/cpu_check.py"
fixture=previous.read_text()
# V187 introduces independent prompt collectives in _forward_tp. This fixture
# tests the V68 grouped decoder and its finalizer. V187's own joined model-path
# suite is rerun separately for its prompt-only dispatch additions.
fixture=fixture[:fixture.index("model_path = source")]
fixture=fixture.replace("assert json.loads(tuning_config) == {'use_stream_k': False, 'use_tma_c': False}",
    "assert {k:v for k,v in json.loads(tuning_config).items() if k != 'block_shape'} == {'use_stream_k': False, 'use_tma_c': False}")
counts=Counter();summaries={}
anchor="        values, row_map, local_ids = scope['humming_grouped_decode_outputs']"
assert fixture.count(anchor)==1
fixture=fixture.replace(anchor,"        selected(layer, tokens, topk)\n"+anchor)
cfg=json.loads((here/"config.json").read_bytes())
tunes=[(v["bounds"][1],json.dumps(dict(block_shape=v["block"],use_stream_k=False,use_tma_c=False))) for v in cfg["configs"].values()]
for mode in ("enabled","missing_middle","disabled"):
    env={"__file__":str(previous)}
    by_weight={}
    def selected(layer,tokens,k):
        layer.humming_decode_tunings["w2"]=tunes
        layer.w2_weight=layer.w2
        layer.w2_weight_scale=object()
        by_weight[id(layer.w2_weight)]=layer
        layer.humming_decode_down={t:t for _,t in tunes
             if mode!="disabled" and not(mode=="missing_middle" and json.loads(t)["block_shape"][0]==16)}
    def launch(*,configs,inputs,outputs,weight,weight_scale,expert_layout,locks):
        layer=by_weight[id(weight)]
        assert weight_scale is layer.w2_weight_scale and locks is None
        n=json.loads(configs)["block_shape"][0]
        counts[(mode,n)]+=1
        env["Humming"].forward_layer(layer,inputs=inputs,outputs=outputs,sublayer_name="w2",expert_layout=expert_layout,
            compute_config=env["scope"]["_GROUPED_COMPUTE"],tuning_config=configs)
    ops.launch_kernel=launch
    env["selected"]=selected
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(fixture,str(previous),"exec"),env)
    summaries[mode]=dict(route_planner=env["routing_cases"],end_to_end=env["kernel_cases"],checked_output_values=env["values_compared"])
assert all(counts[("enabled",w)] for w in (8,16,24)) and not counts[("missing_middle",16)] and not any(k[0]=="disabled" for k in counts)
print(json.dumps(dict(cases=summaries,new_down_calls={f"{m}_M{w}":v for (m,w),v in counts.items()},
    original_route=True,native_operator_boundary="CPU BF16 fixture",physical_gpu=False),indent=2))
