"""Replay both optional projections on the production V190 planner, gather, and sparse finalizer."""
import contextlib
import io
import json
from collections import Counter
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1])
prev=here.parent/"v66_humming_grouped_decode/cpu_check.py"
module_dir=stage/"python/sglang/srt/layers/muse_spark"
counts=Counter()
summary={}
def load(short):
    module=ModuleType("sglang.srt.layers.muse_spark."+short)
    file=module_dir/(short+".py")
    exec(compile(file.read_bytes(),str(file),"exec"),module.__dict__)
    sys.modules[module.__name__]=module
    return module

utils=ModuleType("sglang.kernels.jit.utils");utils.load_jit=None
sys.modules[utils.__name__]=utils
gate=load("decode_gate_up"); down=load("decode_mma_pair")
vendor=ModuleType("humming");ops=ModuleType("humming.ops");vendor.ops=ops
sys.modules[vendor.__name__]=vendor
fixture=prev.read_text().split("model_path = source")[0]
fixture=fixture.replace("assert json.loads(tuning_config) == {'use_stream_k': False, 'use_tma_c': False}",
    "assert {k:v for k,v in json.loads(tuning_config).items() if k!='block_shape'} == {'use_stream_k': False, 'use_tma_c': False}")
anchor="        values, row_map, local_ids = scope['humming_grouped_decode_outputs']"
assert fixture.count(anchor)==1
fixture=fixture.replace(anchor,"        configure(layer)\n"+anchor)
fixture=fixture.replace("    def numel(self):", "    def data_ptr(self):\n        return self.ctypes.data\n\n    def numel(self):")
conf=json.loads((here/"config.json").read_text())["configs"]
tunings=[(s["bounds"][1],json.dumps(dict(block_shape=s["block"],use_stream_k=False,use_tma_c=False))) for s in conf.values()]
for mode in ("all","missing_w13_M16","map_declines","moved_bank","no_w13","no_w2"):
    env={"__file__":str(prev)}
    def configure(layer):
        layer.humming_decode_tunings={"w13":tunings,"w2":tunings}
        layer.w13_weight=layer.w13
        layer.w13_weight_scale=layer.w13_global_scale
        class Launcher:
            def run(self,x,active,keys,offsets,scales,maps,handle,shared,width,sms):
                counts[(mode,"attempt",width)]+=1
                assert x.size and not np.shares_memory(x,active)
                if mode=="map_declines":
                    return False
                # Materialize only in this CPU boundary to separate the WGMMA input
                # reducer, checked independently in C++, from this dispatch fixture.
                t=next(t for _,t in tunings if json.loads(t)["block_shape"][0]==width)
                buf=x.new_empty((len(x),2*active.shape[1]))
                env["Humming"].forward_layer(layer,inputs=x,outputs=buf,expert_layout=offsets,
                    compute_config=env["scope"]["_GROUPED_COMPUTE"],tuning_config=t,sublayer_name="w13")
                env["silu_scaled"](buf,active,keys,offsets,scales,1<<(len(x)-1).bit_length())
                layer.input_addr=x.data_ptr()
                counts[(mode,"gate",width)]+=1
                return True
        choices={t:(None,77,88000,json.loads(t)["block_shape"][0]) for _,t in tunings
                 if not(mode=="missing_w13_M16" and json.loads(t)["block_shape"][0]==16)}
        addresses=tuple((b.data_ptr()+int(mode=="moved_bank"),b.shape) for b in (layer.w13_weight,layer.w13_weight_scale))
        layer.humming_decode_gate_up=None if mode=="no_w13" else (Launcher(),None,addresses,choices,132)
        layer.humming_decode_down={} if mode=="no_w2" else {t:(layer,t) for _,t in tunings}
        layer.w2_weight=layer.w2;layer.w2_weight_scale=layer.w2_global_scale
    def gpu_run(*,configs,inputs,outputs,weight,weight_scale,expert_layout,locks):
        layer,t=configs
        counts[(mode,"down",json.loads(t)["block_shape"][0])]+=1
        assert locks is None and weight is layer.w2_weight and weight_scale is layer.w2_weight_scale
        if hasattr(layer,"input_addr"):
            assert outputs.data_ptr()==layer.input_addr and not np.shares_memory(inputs,outputs)
        env["Humming"].forward_layer(layer,inputs=inputs,outputs=outputs,expert_layout=expert_layout,
            compute_config=env["scope"]["_GROUPED_COMPUTE"],tuning_config=t,sublayer_name="w2")
    env["configure"]=configure;ops.launch_kernel=gpu_run
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(fixture,str(prev),"exec"),env)
    summary[mode]=dict(route_plans=env["routing_cases"],joined_forward=env["kernel_cases"],bf16_compared=env["values_compared"])
for w in (8,16,24):
    assert counts["all","gate",w] and counts["all","down",w]
    assert counts["no_w2","gate",w] and not counts["no_w2","down",w]
    for mode in ("map_declines","moved_bank","no_w13"):
        assert not counts[mode,"gate",w] and counts[mode,"down",w]
assert not counts["missing_w13_M16","gate",16] and counts["map_declines","attempt",16]
print(json.dumps(dict(physical_gpu=False,scenarios=summary,calls={"/".join(map(str,k)):v for k,v in counts.items()}),indent=2))
