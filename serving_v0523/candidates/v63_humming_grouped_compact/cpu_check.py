#!/usr/bin/env python3
"""Run the V63 CPU adapter differential against an exact patched source tree."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def bf16(array):
    values = np.asarray(array, dtype=np.float32)
    bits = values.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


class Tensor(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def new_empty(self, shape):
        return np.full(shape, np.nan, dtype=self.dtype).view(Tensor)

    def sort(self, stable=False):
        order = np.argsort(self, kind='stable' if stable else None)
        return self[order], order.view(Tensor)

    def long(self):
        return self.astype(np.int64)

    def view(self, *shape):
        return super().view(*shape) if len(shape) == 1 and isinstance(shape[0], type) else self.reshape(*shape)

    def index_select(self, dim, indices):
        return np.take(self, indices, axis=dim).view(Tensor)

    def index_copy_(self, dim, indices, values):
        assert dim == 0
        self[indices] = values
        return self

    def mul_(self, value):
        self[...] = bf16(np.multiply(self, value))
        return self

    def unsqueeze(self, dim):
        return np.expand_dims(self, dim).view(Tensor)

    def numel(self):
        return self.size

    def zero_(self):
        self.fill(0)
        return self


def as_tensor(a):
    return np.asarray(a).view(Tensor)


def index_select(x, dim, indices, *, out):
    assert not np.shares_memory(x, out)
    out[...] = np.take(x, indices, axis=dim)
    return out


def silu_and_mul(gate_up, output):
    assert not np.shares_memory(gate_up, output)
    gate, up = np.split(np.asarray(gate_up), 2, axis=1)
    output[...] = bf16(gate / (1 + np.exp(-gate)) * up)
    return output


def module(name, **values):
    m = ModuleType(name)
    vars(m).update(values)
    sys.modules[name] = m
    return m


module('sglang.kernels.ops.activation.activation', silu_and_mul=silu_and_mul)


class HummingMethod:
    @staticmethod
    def forward_layer(layer, *, inputs, outputs, expert_layout, compute_config, tuning_config, sublayer_name):
        assert json.loads(compute_config) == {'gemm_type': 'grouped_contiguous', 'use_f16_accum': False}
        assert json.loads(tuning_config) == {'use_stream_k': False, 'use_tma_c': False}
        assert expert_layout.size == layer.num_local_experts + 1
        assert expert_layout[0] == 0 and expert_layout[-1] == inputs.shape[0]
        assert outputs.flags.c_contiguous and inputs.flags.c_contiguous
        assert not np.shares_memory(inputs, outputs)
        assert np.all(np.isfinite(inputs))
        layer.launches.append((sublayer_name, inputs, outputs))
        weights = getattr(layer, sublayer_name)
        for expert, (lo, hi) in enumerate(zip(expert_layout[:-1], expert_layout[1:])):
            outputs[lo:hi] = bf16(inputs[lo:hi] @ weights[expert].T)


module('humming.layer', HummingMethod=HummingMethod)
torch = SimpleNamespace(
    arange=lambda *a, dtype, device: np.arange(*a, dtype=dtype).view(Tensor),
    searchsorted=lambda a, v, out_int32: np.searchsorted(a, v).astype(np.int32).view(Tensor),
    index_select=index_select,
    div=lambda x,y,rounding_mode: np.floor_divide(x,y).view(Tensor),
    full_like=lambda x,value: np.full_like(x,value).view(Tensor),
    empty_like=lambda x, memory_format: x.new_empty(x.shape),
    contiguous_format=None,
    bfloat16=np.dtype(np.float32),
    compiler=SimpleNamespace(is_compiling=lambda: False),
    cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
)


def extract(path, names, env):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for klass in (n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'MuseSpark1xFp4Experts'):
        nodes.extend(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in names)
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list.clear()
    fut = ast.parse('from __future__ import annotations').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=fut+nodes,type_ignores=[])),path,'exec'),env)


if len(sys.argv) != 2:
    raise SystemExit("usage: cpu_check.py PATCHED_SGLANG_ROOT")

source = Path(sys.argv[1]) / "python/sglang/srt/layers/muse_spark"


def reference_expert(layer, x, row, expert):
    gate = bf16(x[row:row+1] @ layer.w13[expert].T)
    gate = as_tensor(bf16(gate * layer.w13_global_scale[expert]))
    activated = x.new_empty((1,layer.intermediate_size_per_partition))
    silu_and_mul(gate,activated)
    return bf16(bf16(activated @ layer.w2[expert].T) * layer.w2_global_scale[expert])[0]


def indexed(layer, hidden, local_ids):
    layer.fallback += 1
    out = hidden.new_empty((*local_ids.shape, hidden.shape[1]))
    for row,ids in enumerate(local_ids):
        for slot,e in enumerate(ids):
            if e >= 0:
                out[row,slot] = reference_expert(layer,hidden,row,e)
    return out,local_ids


def finalizer(values,scores,*,norm_weight,eps,weight_first,slot_expert_ids=None,slot_row_indices=None,out=None):
    if out is None:
        out = values.new_empty((scores.shape[0],values.shape[-1]))
    map_ = slot_expert_ids if slot_row_indices is None else slot_row_indices
    assert map_.shape == scores.shape
    for row in range(scores.shape[0]):
        acc = np.zeros(values.shape[-1], np.float32)
        for k in range(scores.shape[1]):
            i = map_[row,k]
            if i >= 0:
                val=values[row,k] if slot_row_indices is None else values[i]
                val = val * norm_weight
                val = val * np.float32(1/np.sqrt(np.mean(val*val)+eps))
                acc = acc + val*scores[row,k]
        out[row] = bf16(acc)
    return out


humming_env={'torch':torch,'_GROUPED_COMPUTE':'{"gemm_type": "grouped_contiguous", "use_f16_accum": false}','humming_expert_outputs':indexed}
extract(str(source / 'muse_spark_humming_moe.py'),{'humming_grouped_expert_outputs'},humming_env)
grouped=humming_env['humming_grouped_expert_outputs']
module('sglang.srt.layers.muse_spark.muse_spark_humming_moe',
       humming_grouped_expert_outputs=grouped,humming_expert_outputs=indexed,
       local_humming_expert_ids=lambda layer,ids:ids)
wrapper_env={'torch':torch,'fused_post_expert_norm_weighted_sum':finalizer,
             '_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT':256,
             'is_in_breakable_cuda_graph':lambda:False}
extract(str(source / 'muse_spark_fp4_moe.py'),{'_grouped_humming_and_combine_impl','_forward_w4a16'},wrapper_env)


rng=np.random.default_rng(26)
counts={'grouped':0,'indexed':0,'empty':0}
for width,intermediate in ((4,4),(8,4),(4,8),(10,3)):
    for n in (1,2,7,17,33):
        for topk in (1,3,8):
            for distribution in ('empty','balanced','skew','all'):
                e=5
                x=as_tensor(bf16(rng.integers(-12,13,(n,width),dtype=np.int32)/16))
                scores=as_tensor((rng.integers(1,9,(n,topk),dtype=np.int32)/16).astype(np.float32))
                p = rng.integers(-1, e, (n, topk), dtype=np.int32)
                if distribution=='empty': p.fill(-1)
                if distribution=='all': p[:]=np.abs(p)
                if distribution=='balanced': p[rng.random(p.shape)<0.6]=-1
                if distribution=='skew': p[p>=0]=0
                ids=as_tensor(p)
                layer=SimpleNamespace(
                    num_local_experts=e,intermediate_size_per_partition=intermediate,
                    w13=as_tensor(bf16(rng.integers(-8,9,(e,2*intermediate,width),dtype=np.int32)/16)),
                    w2=as_tensor(bf16(rng.integers(-8,9,(e,width,intermediate),dtype=np.int32)/16)),
                    w13_global_scale=as_tensor(bf16(rng.integers(1,9,e,dtype=np.int32)/16)),
                    w2_global_scale=as_tensor(bf16(rng.integers(1,9,e,dtype=np.int32)/16)),
                    humming_grouped_tunings={k:'{"use_stream_k": false, "use_tma_c": false}' for k in ('w13','w2')},
                    _post_expert_norm_weight=as_tensor(np.ones(width,np.float32)),
                    _post_expert_norm_eps=1e-8,_post_expert_norm_weight_first=True,
                    launches=[],fallback=0,
                )
                reference=finalizer(indexed(layer,x,ids)[0], scores, norm_weight=layer._post_expert_norm_weight,
                                    eps=layer._post_expert_norm_eps, weight_first=True, slot_expert_ids=ids)
                layer.fallback=0
                out=x.new_empty(x.shape)
                returned=wrapper_env['_grouped_humming_and_combine_impl'](layer,x,ids,scores,out)
                np.testing.assert_array_equal(out,reference)
                assert returned is None
                if layer.launches:
                    assert len(layer.launches)==2 and layer.fallback==0
                    _,input1,output1=layer.launches[0]; _,input2,output2=layer.launches[1]
                    assert np.shares_memory(input1,input2) and np.shares_memory(output1,output2)
                    assert input1.shape[0]==int(np.count_nonzero(p>=0))
                    counts['grouped']+=1
                elif layer.fallback:
                    assert layer.fallback==1
                    counts['indexed']+=1
                else:
                    assert (p<0).all()
                    counts['empty']+=1
print('Actual adapter and finalize wrapper cases',counts)


anchor=object()
class GateLayer:
    use_humming=True
    global_num_experts=256
    _post_expert_norm=anchor
    _post_expert_norm_weight=as_tensor(np.ones(1,np.float32))
    _post_expert_norm_eps=1e-8
    _post_expert_norm_weight_first=True
    def _grouped_humming_and_combine(self,hidden,ids,scores,out):
        self.routed='grouped'; out.zero_()

gate_module=sys.modules['sglang.srt.layers.muse_spark.muse_spark_humming_moe']

def indexed_gate(layer,x,ids):
    layer.routed='indexed'
    return x.new_empty((*ids.shape,1)).zero_(),ids

gate_module.humming_expert_outputs=indexed_gate
wrapper_env['fused_post_expert_norm_weighted_sum']=lambda values,scores,**kwargs: values[:,0,:]
for tokens in (8191,8192,16384):
    for active in (False,True):
        for breakable in (False,True):
            for compiling in (False,True):
                for is_bf16 in (False,True):
                    torch.cuda.is_current_stream_capturing=lambda:active
                    torch.compiler.is_compiling=lambda:compiling
                    wrapper_env['is_in_breakable_cuda_graph']=lambda:breakable
                    g=GateLayer()
                    x=as_tensor(np.zeros((tokens,1),np.float32 if is_bf16 else np.float64))
                    ids=as_tensor(np.zeros((tokens,8),np.int32))
                    wrapper_env['_forward_w4a16'](g,x,ids,ids,anchor)
                    assert g.routed==('grouped' if is_bf16 and tokens>=8192 and not compiling and (not active or breakable) else 'indexed')
print('Actual entry gate cases 48')


scheduler_checks=0
for block in (8,32,128,176):
    for grid in (1,17,132):
        for _ in range(20):
            lengths=rng.integers(0,520,32)
            lengths[rng.random(32)<0.5]=0
            first=np.cumsum(np.r_[0,lengths])
            expected=[(e,int(first[e]+m),int(first[e+1])) for e,n in enumerate(lengths) for m in range(0,n,block)]
            nblocks=len(expected)
            for cta in range(grid):
                current_expert=old=within=0
                for mib in range(cta,nblocks,grid):
                    within += (mib-old)*block
                    while within>=lengths[current_expert]:
                        within-=int(-(-lengths[current_expert]//block)*block)
                        current_expert+=1
                    old=mib
                    observed=(current_expert,int(first[current_expert]+within),int(first[current_expert+1]))
                    assert observed==expected[mib]
                    scheduler_checks+=1
print('Pinned grouped scheduler tail cases',scheduler_checks)
