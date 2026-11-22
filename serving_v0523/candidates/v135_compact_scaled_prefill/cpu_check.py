"""Run the two production adapter paths and compact CUDA body on local CPU tensors.

Usage: python cpu_check.py PATCHED_SGLANG
The checkout must have the parent tree in its object store.
"""
import ast
import ctypes
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import torch

stage = Path(sys.argv[1]).resolve()
parent = "b5a37c2923f8d1ab24c1c5cf7920a21e2c086774"
group = "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
moe = group.replace("humming", "fp4")
header = stage / "python/sglang/kernels/jit/csrc/muse_spark/expert_prefill_scaled_silu.cuh"
activation = stage / "python/sglang/kernels/jit/csrc/elementwise/activation.cuh"
torch.set_num_threads(2)
torch.manual_seed(135)


def load(source, names, env):
    funcs = []
    for x in ast.walk(ast.parse(source)):
        if isinstance(x, ast.FunctionDef) and x.name in names:
            x.decorator_list.clear()
            x.body = [item for item in x.body if not isinstance(item, ast.ImportFrom)]
            x.returns = None
            for arg in x.args.args:
                arg.annotation = None
            funcs.append(x)
    assert len(funcs) == len(names)
    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None
    for f in funcs:
        RemoveImports().visit(f)
    exec(compile(ast.Module(funcs, type_ignores=[]), "<production>", "exec"), env)


cpp_prefix = r"""
#include <bit>
#include <cmath>
#include <algorithm>
#include <cstdint>
#define SGL_DEVICE inline
#define __global__
#define __launch_bounds__(x)
#define __grid_constant__
struct {uint32_t x;} threadIdx, blockIdx, gridDim, blockDim;
namespace sglang {
namespace host { template<typename T> constexpr bool dependent_false_v = false; }
namespace device {
struct bf16_t {
  uint16_t u;
  bf16_t() = default;
  explicit bf16_t(float x) {
    uint32_t bits=std::bit_cast<uint32_t>(x);
    u=(bits+32767+((bits>>16)&1))>>16;
  }
  explicit operator float() const {return std::bit_cast<float>((uint32_t)u<<16);}
};
constexpr uint32_t kMaxVecBytes=16;
template<class T, unsigned N> struct AlignedVector {T x[N]; T& operator[](int i){return x[i];}
  const T& operator[](int i) const {return x[i];}};
template<class T,class V> T cast(V x){return static_cast<T>(x);}
template<class T> T load_as(const void* p,int64_t i){return static_cast<const T*>(p)[i];}
template<class T> void store_as(void* p,T x,int64_t i){static_cast<T*>(p)[i]=x;}
}
"""
cpp_end = r"""
extern "C" void run(const void* input,void* output,const int32_t* ids,const void* scales,
                    uint32_t rows,uint32_t width,uint32_t max_blocks){
  using namespace muse_expert_prefill_scaled_silu;
  Params p{(const bf16_t*)input,(bf16_t*)output,ids,(const bf16_t*)scales,rows,width/kVecSize};
  blockDim.x=kThreads;gridDim.x=std::min(rows,max_blocks);
  for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)
    for(threadIdx.x=0;threadIdx.x<std::min(kThreads,p.row_vecs);threadIdx.x++)
      run_kernel(p);
}
}
"""
src = header.read_text()
act = activation.read_text()
cpp = (cpp_prefix + act[act.index("enum class ActivationKind"):act.index("struct ActivationParams")]
       + src[src.index("namespace muse_expert_prefill"):src.index("struct MuseExpertPrefill")] + cpp_end)


def make_emulator(temp):
    code, lib = temp / "test.cc", temp / "test.so"
    code.write_text(cpp)
    subprocess.run(["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-shared", "-fPIC", str(code), "-o", str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    launch = dll.run
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3
    launch.restype = None

    def kernel(values, outputs, ids, scales):
        assert all(t.is_contiguous() for t in (values, outputs, ids, scales))
        assert values.dtype == outputs.dtype == scales.dtype == torch.bfloat16
        assert ids.dtype == torch.int32 and bool(((ids >= 0) & (ids < len(scales))).all())
        assert torch.equal(ids, torch.sort(ids).values)
        launch(*(t.data_ptr() for t in (values, outputs, ids, scales)), len(ids), outputs.shape[1], 17)
    return kernel


def finalize(values, scores, *, norm_weight, eps, weight_first, slot_expert_ids, slot_row_indices, out, expert_output_scales=None):
    assert slot_row_indices is not None and values.ndim == 2
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    result = torch.zeros_like(out, dtype=torch.float)
    for k in range(scores.shape[1]):
        rows = slot_row_indices[:, k]
        live = rows >= 0
        if expert_output_scales is not None:
            assert torch.equal(live, slot_expert_ids[:, k] >= 0)
        x = values[rows[live].long()].float()
        if expert_output_scales is not None:
            x = (x * expert_output_scales[slot_expert_ids[live, k].long()].float()[:, None]).bfloat16().float()
        if weight_first:
            x = x * norm_weight.float()
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
        if not weight_first:
            x = x * norm_weight.float()
        result[live] += x * scores[live, k, None]
    out.copy_(result)


class Humming:
    @staticmethod
    def forward_layer(layer, *, inputs, outputs, expert_layout, compute_config, tuning_config, sublayer_name):
        assert compute_config == '{"gemm_type": "grouped_contiguous", "use_f16_accum": false}'
        assert tuning_config == dict(use_stream_k=False, use_tma_c=False)
        assert inputs.is_contiguous() and outputs.is_contiguous()
        assert inputs.untyped_storage().data_ptr() != outputs.untyped_storage().data_ptr()
        assert expert_layout.dtype == torch.int32 and expert_layout[0] == 0 and expert_layout[-1] == len(inputs)
        layer.calls.append((sublayer_name, len(inputs), expert_layout.tolist()))
        for e, (lo, hi) in enumerate(zip(expert_layout.tolist(), expert_layout[1:].tolist())):
            # Dyadic weights and inputs avoid host BLAS accumulation variations across row counts.
            outputs[lo:hi] = inputs[lo:hi].float() @ layer.weights[sublayer_name][e]


def stock_silu(gates, out):
    g, u = gates.float().chunk(2, 1)
    out.copy_(g / (1 + torch.exp(-g)) * u)


def setup(updated, fused, norm):
    def read(path):
        return (stage / path).read_text() if updated else subprocess.check_output(
            ["git", "show", f"{parent}:{path}"], cwd=stage).decode()
    env = dict(torch=torch, HummingMethod=Humming, _GROUPED_COMPUTE='{"gemm_type": "grouped_contiguous", "use_f16_accum": false}',
               fused_post_expert_norm_weighted_sum=finalize, silu_and_mul=stock_silu,
               _HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256, is_in_breakable_cuda_graph=lambda:False,
               local_humming_expert_ids=lambda layer, ids: ids.int().contiguous())
    names = {"humming_grouped_expert_outputs"}
    if updated:
        names.update({"_compact_prefill_windows", "compact_prefill_and_combine"})
    load(read(group), names, env)
    load(read(moe), {"_grouped_humming_and_combine_impl", "_forward_w4a16"}, env)
    layer = NS(num_local_experts=4, global_num_experts=32, hidden_size=16, intermediate_size_per_partition=16,
               _post_expert_norm=norm, _post_expert_norm_weight=norm, use_humming=True,
               _post_expert_norm_eps=1e-5, _post_expert_norm_weight_first=True, humming_prefill_silu=fused,
               calls=[], humming_grouped_tunings={k:dict(use_stream_k=False,use_tma_c=False) for k in ("w13", "w2")})
    layer._grouped_humming_and_combine = lambda *a: env["_grouped_humming_and_combine_impl"](layer,*a)
    return env, layer


def main():
    torch.cuda.is_current_stream_capturing = lambda: False
    with tempfile.TemporaryDirectory() as temp:
        kernel = make_emulator(Path(temp))
        norm = torch.randn(16).bfloat16()
        old, ref = setup(False, None, norm)
        new, layer = setup(True, NS(run=kernel), norm)
        weights = {k:torch.randint(-4,5,(4,16,d)).float()/8 for k,d in (("w13",32),("w2",16))}
        for model in (ref,layer):
            model.weights=weights
            model.w13_global_scale=torch.tensor([0.1171875,0.7578125,1.328125,2.265625],dtype=torch.bfloat16)
            model.w2_global_scale=torch.tensor([1.203125,0.04248046875,3.0625,0.91796875],dtype=torch.bfloat16)
        cases = 0
        values = 0
        counts = {}
        max_windows = 0
        for n in (1,127,128,129,1024,1025,4093,16384):
            for route_mode in ("remote","balanced","skew","corner"):
                k=8; hidden=torch.randint(-32,33,(n,16)).bfloat16()/16
                inds=torch.randint(0,4,(n,k),dtype=torch.int32)
                if route_mode=="remote":
                    inds.fill_(-1)
                if route_mode=="balanced":
                    inds.masked_fill_(torch.rand(n,k)<0.875,-1)
                if route_mode=="corner":
                    inds[:n//3]=-1; inds[-1]=torch.tensor([2,-1,0,2,1,0,-1,3])
                probs=torch.rand(n,k); probs/=probs.sum(1,keepdim=True)
                for weight_first in (True,False):
                    for model in (ref,layer):
                        model.calls.clear(); model._post_expert_norm_weight_first=weight_first
                    if n>=1024:
                        actual = new["_forward_w4a16"](layer,hidden,probs,inds,norm)
                        expected = old["_forward_w4a16"](ref,hidden,probs,inds,norm)
                    else:
                        actual,expected = torch.empty_like(hidden),torch.empty_like(hidden)
                        layer._grouped_humming_and_combine(hidden,inds,probs,actual)
                        ref._grouped_humming_and_combine(hidden,inds,probs,expected)
                    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)), (n,route_mode,weight_first,(actual!=expected).sum())
                    t=list(new["_compact_prefill_windows"](inds,4,32))
                    assert sum(w[-1] for w in t)==torch.count_nonzero(inds>=0).item()
                    assert [q[1] for q in layer.calls if q[0]=="w2"]==[w[-1] for w in t if w[-1]]
                    assert all(lo<hi and (i==0 or t[i-1][1]==lo) for i,(lo,hi,r) in enumerate(t))
                    assert t[0][0]==0 and t[-1][1]==n
                    max_windows=max(max_windows,len(t)); cases+=1; values+=actual.numel()
                    if n==16384: counts[route_mode]=[w[-1] for w in t]
        # Disabling at model setup goes through the unchanged grouped branch.
        layer.humming_prefill_silu=None
        fallback=new["_forward_w4a16"](layer,hidden,probs,inds,norm)
        assert torch.equal(fallback,expected)
        # Exercise the vector loop's second iteration and padded, all-remote masks.
        for width in (8,256,4096,8192):
            ids=torch.arange(68,dtype=torch.int32)//17
            raw=(torch.randn(68,2*width)*3).bfloat16()
            out=torch.empty((68,width),dtype=torch.bfloat16)
            kernel(raw,out,ids,layer.w13_global_scale)
            stock=torch.empty_like(out);stock_silu(raw*layer.w13_global_scale[ids.long(),None],stock)
            assert torch.equal(out.view(torch.int16),stock.view(torch.int16))
        print(json.dumps(dict(torch=torch.__version__,device="CPU",cases=cases,compared_output_elements=values,
              largest_wave_count=max_windows,routes_by_window_16384=counts,halfword_agreement=True,
              actual_forward_entry_cases=32,fused_silu_extra_widths=[8,256,4096,8192],
              cuda_header_sha256=hashlib.sha256(src.encode()).hexdigest(),
              limits="Host tensor Humming surrogate; CPU reference expf. GPU numerical and timing checks outstanding."),indent=2))


if __name__ == "__main__":
    main()
