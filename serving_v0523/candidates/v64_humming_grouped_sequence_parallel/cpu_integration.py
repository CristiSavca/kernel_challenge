"""CPU routing integration only. Usage: python cpu_integration.py PATCHED_SGLANG_ROOT."""
import ast
import contextlib
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


assert len(sys.argv) == 2, 'Pass the patched SGLang source root'
source_root = Path(sys.argv[1]).resolve()
base=str(source_root / 'python/sglang/srt/layers/muse_spark') + '/'

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
extract(base+'muse_spark_humming_moe.py',{'humming_grouped_expert_outputs'},humming_env)
grouped=humming_env['humming_grouped_expert_outputs']
module('sglang.srt.layers.muse_spark.muse_spark_humming_moe',
       humming_grouped_expert_outputs=grouped,humming_expert_outputs=indexed,
       local_humming_expert_ids=lambda layer,ids:ids)
wrapper_env={'torch':torch,'fused_post_expert_norm_weighted_sum':finalizer,
             '_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT':256,
             'is_in_breakable_cuda_graph':lambda:False}
extract(base+'muse_spark_fp4_moe.py',{'_grouped_humming_and_combine_impl','_forward_w4a16'},wrapper_env)



from types import MethodType
sp_source=ast.parse((source_root / 'python/sglang/srt/models/muse_spark.py').read_text())
cls=next(n for n in sp_source.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xMoE')
funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_forward_sp_experts','_forward_sp_replicated_experts')]
sp_env={}
exec(compile(ast.fix_missing_locations(ast.Module(body=ast.parse('from __future__ import annotations').body+funcs,type_ignores=[])),'sp','exec'),sp_env)
rng=np.random.default_rng(6416); cases=0; norm_calls=0
for world in (4,8):
 for n in (7,16,33):
  for width,inter in ((4,4),(4,8),(10,3)):
   for distribution in ('balanced','rank_zero','last_rank'):
    topk=8;e=3;total_e=world*e;pad=(-n)%world;local_rows=(n+pad)//world
    x=as_tensor(bf16(rng.integers(-8,9,(n,width))/16))
    normalized=as_tensor(bf16(x*0.75))
    ids=as_tensor(rng.integers(0,total_e,(n,topk),dtype=np.int32))
    if distribution=='rank_zero':ids[:]=ids%e
    if distribution=='last_rank':ids[:]=ids%e+(world-1)*e
    scores=as_tensor((rng.integers(1,9,(n,topk))/16).astype(np.float32))
    w13=as_tensor(bf16(rng.integers(-8,9,(total_e,2*inter,width))/16))
    w2=as_tensor(bf16(rng.integers(-8,9,(total_e,width,inter))/16))
    gs13=as_tensor(bf16(rng.integers(1,9,total_e)/16));gs2=as_tensor(bf16(rng.integers(1,9,total_e)/16))
    nw=as_tensor(bf16(rng.integers(1,9,width)/16))
    all_ref=[]; actual=[];returned=[];calls=[]
    for rank in range(world):
     offset=rank*e
     layer=SimpleNamespace(num_local_experts=e,intermediate_size_per_partition=inter,
       w13=w13[offset:offset+e],w2=w2[offset:offset+e],w13_global_scale=gs13[offset:offset+e],w2_global_scale=gs2[offset:offset+e],
       humming_grouped_tunings={k:'{"use_stream_k": false, "use_tma_c": false}' for k in ('w13','w2')},
       _post_expert_norm_weight=nw,_post_expert_norm_eps=1e-8,_post_expert_norm_weight_first=True,launches=[],fallback=0)
     local_ids=as_tensor(np.where((ids>=offset)&(ids<offset+e),ids-offset,-1).astype(np.int32))
     direct=indexed(layer,normalized,local_ids)[0]
     all_ref.append(finalizer(direct,scores,norm_weight=nw,eps=1e-8,weight_first=True,slot_expert_ids=local_ids))
     class Experts:
      post_expert_norm=object()
      def __call__(self,inputs,received_scores,received_ids,norm):
       assert norm is self.post_expert_norm
       np.testing.assert_array_equal(inputs,normalized)
       np.testing.assert_array_equal(received_scores,scores)
       np.testing.assert_array_equal(received_ids,ids)
       out=inputs.new_empty(inputs.shape)
       wrapper_env['_grouped_humming_and_combine_impl'](layer,inputs,local_ids,received_scores,out)
       return out
     counter={'gather':0,'norm':0}
     global_arrays=(normalized,scores,ids)
     def gather(local,num_tokens):
      k=counter['gather'];counter['gather']+=1;assert num_tokens==n
      padded=np.pad(global_arrays[k],((0,pad),(0,0)))
      np.testing.assert_array_equal(local,padded[rank*local_rows:(rank+1)*local_rows])
      return global_arrays[k]
     def reduce_rows(out,group):
      assert group.rank==rank
      actual.append(out.copy());z=out.new_empty((local_rows,width));returned.append(z);return z
     def norm(local):counter['norm']+=1;return as_tensor(bf16(local*0.75))
     sp_env.update(layernorm_sp=SimpleNamespace(sp_num_tokens=lambda:n,sp_exit_gather=gather,reduce_scatter_token_rows=reduce_rows),get_parallel=lambda:SimpleNamespace(tp_group=SimpleNamespace(rank=rank)))
     moe=SimpleNamespace(pre_expert_norm=norm,sp_replicate_expert_input=True,experts=Experts())
     moe._forward_sp_replicated_experts=MethodType(sp_env['_forward_sp_replicated_experts'],moe)
     local=lambda arr:as_tensor(np.pad(arr,((0,pad),(0,0)))[rank*local_rows:(rank+1)*local_rows])
     result=sp_env['_forward_sp_experts'](moe,local(x),local(scores),local(ids),None)
     assert result is returned[-1] and counter=={'gather':3,'norm':1}
     norm_calls+=counter['norm']
    for a,b in zip(actual,all_ref):np.testing.assert_array_equal(a,b)
    reduced=bf16(np.stack(actual).sum(axis=0,dtype=np.float32));expected=bf16(np.stack(all_ref).sum(axis=0,dtype=np.float32))
    padded=np.pad(reduced,((0,pad),(0,0)))
    for rank,out in enumerate(returned):out[:]=padded[rank*local_rows:(rank+1)*local_rows]
    np.testing.assert_array_equal(np.concatenate(returned)[:n],expected)
    cases+=1
print(json.dumps({'state':'CPU_SP_GROUPED_PASS','integration_cases':cases,'actual_pre_norm_calls':norm_calls,'details':['actual SP gather entry and replicated expert methods','actual compact grouped expert adapter and caller output bridge','TP4 and TP8, original order, padded tails, balanced and rank-empty/skew routes','each rank calls pre-expert norm exactly once','modeled BF16 GEMM and finalizer; no CUDA execution']},indent=2))
