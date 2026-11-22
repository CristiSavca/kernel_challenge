"""Execute the submitted wave planner, loop, row packer and finalizer with CPU projection oracles."""
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

here = Path(__file__).resolve().parent
previous = runpy.run_path(str(next(here.parent.glob("v115_*/replay_check.py"))))
base = previous["BASE"]
torch.set_num_threads(1)


class Graph(previous["Graph"]):
    def end_capture_to_conditional_node(self):
        self.recording = False
        assert self.body[-1][0] == self.update and len(self.body) in (6, 7)
        self.again = bool(self.active)
        self.iterations = 0
        while self.again:
            self.iterations += 1
            assert self.iterations < 64
            for fn, args, kwargs in self.body:
                fn(*args, **kwargs)


def disjoint(a, b):
    assert a.is_contiguous() and b.is_contiguous()
    return a.data_ptr() + a.nbytes <= b.data_ptr() or b.data_ptr() + b.nbytes <= a.data_ptr()


def check(stage):
    source = stage / "python/sglang/srt/layers/muse_spark/humming_prefill_waves.py"
    ops, _, _ = base["make_ops"](stage)
    pid, (loader,) = base["program"](source, ("_load_wave",))
    graph = Graph()
    ops["_load_wave"] = graph.kernel(base["Launch"](loader, pid, (2,3,4,5,6)))
    ops["_pack_wave"] = graph.kernel(ops["_pack_wave"])
    ops["fused_post_expert_norm_weighted_sum"] = graph.call(ops["fused_post_expert_norm_weighted_sum"])
    launches, modes, result = [], Counter(), Counter()
    storage = {}
    def grouped(layer, inputs, outputs, expert_layout, compute_config, tuning_config, sublayer_name):
        assert disjoint(inputs, outputs)
        assert compute_config=="grouped" and sublayer_name==tuning_config
        bounds = expert_layout.tolist()
        assert bounds == sorted(bounds) and bounds[-1] == len(inputs)
        outputs.fill_(float("nan"))
        for e,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
            if lo<hi:
                outputs[lo:hi]=inputs[lo:hi].float() @ getattr(layer,sublayer_name)[e].T
        launches.append((sublayer_name,len(inputs)-bounds[0],len(inputs)))
        storage[sublayer_name]=inputs.untyped_storage().nbytes()
    ops["HummingMethod"]=SimpleNamespace(forward_layer=graph.call(grouped))
    def scaled(raw, act, keys, offsets, scales, span):
        assert disjoint(raw,act)
        lo = int(offsets[0])
        experts = keys[lo:]//span-1
        gate,up=(raw[lo:].float()*scales[experts,None].float()).bfloat16().float().chunk(2,dim=1)
        act[lo:]=torch.nn.functional.silu(gate)*up
    ops["_expert_scaled_silu_module"]=lambda:SimpleNamespace(run=graph.call(scaled))
    def bind(layer,packed,act):
        assert disjoint(packed,act)
        if mode=="decline":
            return None
        def project(keys,offsets):
            edges=offsets.tolist()
            assert edges==sorted(edges) and edges[-1]==len(packed)
            for e,(lo,hi) in enumerate(zip(edges,edges[1:])):
                if lo==hi:
                    continue
                assert (keys[lo:hi]==e).all(), "wrong suffix or encoded expert ID"
                raw=(packed[lo:hi].float()@layer.w13[e].T).bfloat16()
                gate,up=(raw.float()*layer.w13_global_scale[e].float()).bfloat16().float().chunk(2,dim=1)
                act[lo:hi]=torch.nn.functional.silu(gate)*up
            launches.append(("fused",len(packed)-edges[0],len(packed)))
        return graph.call(project)
    mod=ModuleType("sglang.srt.layers.muse_spark.decode_gate_up")
    mod.bind_prompt_wave=bind
    rng=torch.Generator().manual_seed(306)
    rand=lambda shape:torch.randint(-4,5,shape,generator=rng).float()/8
    with patch.dict(sys.modules,{mod.__name__:mod}):
        for t,k,e,cap,h,i in [(19,4,3,6,16,16),(37,8,7,23,8,16),(23,3,4,11,8,16),(9,2,4,5,16,8),(1,2,2,2,8,8)]:
            hidden=rand((t,h)).bfloat16()
            scores=rand((t,k))
            layer=SimpleNamespace(num_local_experts=e,intermediate_size_per_partition=i,
                w13=rand((e,2*i,h)),w2=rand((e,h,i)),w13_global_scale=torch.linspace(.5,1.25,e).bfloat16(),
                w2_global_scale=torch.linspace(.625,1.5,e).bfloat16(),_post_expert_norm_weight=rand((h,)).bfloat16(),
                _post_expert_norm_eps=1e-5,humming_grouped_tunings={"w13":"w13","w2":"w2"})
            for layout in ("random","remote","all_local","tail","duplicates","boundary"):
                ids=torch.randint(-e,e*2,(t,k),dtype=torch.int32,generator=rng)
                if layout=="remote": ids.fill_(-1)
                if layout=="all_local": ids.remainder_(e)
                if layout=="tail": ids[:t//2].fill_(-1)
                if layout=="duplicates": ids.fill_(0)
                if layout=="boundary": ids.fill_(-1);ids.flatten()[:cap]=e-1
                valid=int(((ids>=0)&(ids<e)).sum())
                for mode,capture,condition,first in itertools.product(("fused","decline","unavailable"),(True,False),(True,False),(True,False)):
                    layer._post_expert_norm_weight_first=first
                    layer.humming_prompt_gate_up=mode!="unavailable" or None
                    layer.humming_conditional_waves=condition
                    expected=base["reference"](layer,hidden,scores,ids)
                    launches.clear();graph.iterations=None
                    with patch.object(torch.cuda,"is_current_stream_capturing",return_value=capture),patch.object(torch.cuda,"CUDAGraph",graph):
                        output=ops["grouped_prefill_waves"](layer,hidden,scores,ids,cap)
                    assert torch.equal(expected.view(torch.int16),output.view(torch.int16))
                    assert all(p[0]==("fused" if mode=="fused" else "w13") for p in launches[::2])
                    assert all(p[0]=="w2" for p in launches[1::2])
                    assert sum(p[1] for p in launches[::2])==sum(p[1] for p in launches[1::2])==valid
                    n=min(t*k,cap+k-1);budget=n*(max(h,i)+max(h,2*i))
                    assert storage["w2"]==budget*2
                    capacity=cap
                    if mode=="fused":
                        rows=min(t*k,budget//(h+i));capacity=rows if rows==t*k else rows-k+1
                    _,_,_,bounds=ops["plan_waves"](ids,e,capacity)
                    total=len(bounds)-1
                    actual=int((bounds[:-1]<t).sum()) if capture and condition and total>=4 else total
                    assert len(launches)==2*actual
                    loops=actual-int(mode=="fused") if capture and condition and total>=4 else None
                    assert graph.iterations==loops
                    result["cases"]+=1;result["bf16_words"]+=output.numel();result["valid_routes"]+=valid
                    modes[f"{mode}/while={loops}"]+=1
    return dict(counts=result,mode_counts=modes,scope="Real planner, partitioning, scratch views, packer and finalizer; modeled CUDA graph, matrix projections and epilogue on host Torch.",
                wave_sha256=sha256(source.read_bytes()).hexdigest(),physical_gpu=False)


if __name__=="__main__":
    print(json.dumps(check(Path(sys.argv[1])),indent=2))
