"""Execute exact workspace selection and compare the unchanged CUDA reduction."""
import ast
import enum
import functools
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple, Optional

import numpy as np


class Tensor(np.ndarray):
    device = "cpu"
    def data_ptr(self): return self.__array_interface__["data"][0]
    def numel(self): return self.size
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (type, np.dtype)):
            if shape[0] is Tensor: return super().view(Tensor)
            return np.asarray(self).view(shape[0]).view(Tensor)
        return self.reshape(*shape)


real_torch = NS(bfloat16=np.dtype("uint16"), float32=np.dtype("float32"),
                uint8=np.dtype("uint8"), device=lambda x:x,
                empty=lambda n,dtype:np.empty(n,dtype=dtype).view(Tensor))

root = Path(sys.argv[1])
base_repo = Path(__file__).resolve().parents[2] / 'sglang'
prefix = root / 'python/sglang'
comm_file = prefix / 'srt/distributed/device_communicators/custom_all_reduce_v2.py'
helper_file = prefix / 'srt/layers/communication/muse_spark_workspace_reduce.py'


def extract(source, names, scope, cls=None):
    tree = ast.parse(source)
    body = tree.body if cls is None else next(x.body for x in tree.body if getattr(x, 'name', '') == cls)
    body = [x for x in body if getattr(x, 'name', '') in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body + body, type_ignores=[]), '<source>', 'exec'), scope)


def baseline(path):
    return subprocess.run(['git', '-C', str(base_repo), 'show', '297e54fd105e974cdf401e5b4d20253a22c16f90:' + path], check=True, capture_output=True, text=True).stdout


config_source = baseline('python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py')
cfg_scope = dict(NamedTuple=NamedTuple, Optional=Optional, cache=functools.cache, KB=1024, MB=1024**2)
extract(config_source, ('Range','Heuristic','AllReduceConfig','two_shot_push_slot_bytes','two_shot_push_max_bytes','_pack_heuristic','_sm90_configs'), cfg_scope)
choices = cfg_scope['_sm90_configs'](132)


class Algo(enum.Enum):
    ONE_SHOT_PUSH = 1
    TWO_SHOT_PUSH = 2
    ONE_SHOT_PULL = 3
    TWO_SHOT_PULL = 4


class Config(NamedTuple):
    algo: Algo
    use_graph: bool = False
    use_multicast: bool = False


class Env:
    def __init__(self, value): self.value = value
    def get(self): return self.value


flags = NS(SGLANG_ENABLE_MUSE_SPARK_WORKSPACE_REDUCE=Env(True), SGLANG_ENABLE_DETERMINISTIC_INFERENCE=Env(False))
state = NS(compiling=False, arch=(9,0), sp_active=False,
           overlap=NS(enable_two_batch_overlap=False, enable_single_batch_overlap=False))
parallel = NS(attn_dp_size=1, attn_dcp_size=1, attn_cp_size=1, pp_size=1, tp_size=8, attn_tp_size=8, moe_ep_size=8)
torch = NS(bfloat16=real_torch.bfloat16, compiler=NS(is_compiling=lambda:state.compiling),
           cuda=NS(get_device_capability=lambda _:state.arch))
scope = dict(torch=torch, AllReduceAlgo=Algo, AllReduceConfig=Config)
extract(comm_file.read_text(), ('_pick_config','multicast_workspace','all_reduce_workspace_'), scope, cls='CustomAllReduceV2')


class Comm:
    _pick_config = scope['_pick_config']
    multicast_workspace = scope['multicast_workspace']
    all_reduce_workspace_ = scope['all_reduce_workspace_']
    def _can_use_graph(self): return self.capture


buf = real_torch.empty(16*1024**2, dtype=real_torch.uint8)
comm = Comm()
comm.disabled = False
comm.override_algo = None
comm.capture = True
comm.rank = 0
comm.world_size = 8
comm.obj = object()
comm.max_pull_size = buf.numel()
comm._pull_workspaces = [buf] * 8
comm.config = choices[8].clip(max_pull_bytes=comm.max_pull_size, max_push_bytes=256*1024, world_size=8)
selector_cases = 0
for world, rows, width, capture in itertools.product((4,8), (0,1,8,31,32,33,63,64,65,127,128,129,256,512,1024,2049), (1,7,8,3072,4096,8192), (False,True)):
    comm.world_size = world
    comm.config = choices[world].clip(max_pull_bytes=comm.max_pull_size, max_push_bytes=256*1024, world_size=world)
    comm.capture = capture
    size = rows * width * 2
    expected = world == 8 and capture and 512*1024 <= size <= comm.max_pull_size and size%16 == 0
    view = comm.multicast_workspace(rows,width)
    assert (view is not None) == expected
    if view is not None:
        assert view.data_ptr() == buf.data_ptr()
        assert view.shape == (rows,width) and view.dtype == real_torch.bfloat16
    selector_cases += 1
comm.world_size = 8
comm.config = choices[8].clip(max_pull_bytes=comm.max_pull_size, max_push_bytes=256*1024, world_size=8)
comm.capture = True
for field, value in (('disabled',True),('override_algo',Algo.TWO_SHOT_PULL),('_pull_workspaces',None)):
    previous=getattr(comm,field);setattr(comm,field,value)
    assert comm.multicast_workspace(128,4096) is None
    setattr(comm,field,previous);selector_cases += 1


class LinearMethod: pass
linear = NS(quant_method=LinearMethod(), weight=NS(dtype=real_torch.bfloat16,requires_grad=False), bias=None, input_is_parallel=True, reduce_results=True)
inputs = NS(dtype=real_torch.bfloat16,is_cuda=True,ndim=2,device=real_torch.device('cpu'),shape=(128,8192))
group = NS(ca_comm=comm,_can_use_flashinfer_allreduce=lambda _:False,_resolve_outplace_all_reduce_method=lambda _:'ca')
helper_scope = dict(torch=torch,envs=flags,get_forward=lambda:state,get_parallel=lambda:parallel,get_exec=lambda:state,
                    CustomAllReduceV2=Comm,UnquantizedLinearMethod=LinearMethod,get_bf16_gemm_backend=lambda:NS(value='auto'))
extract(helper_file.read_text(), ('borrow_muse_reduce',), helper_scope)
borrow = helper_scope['borrow_muse_reduce']
assert borrow(group,inputs,8192,linear=linear)[1].data_ptr()==buf.data_ptr()
mutations = [(state,'compiling',True),(state,'sp_active',True),(state,'arch',(10,0)),
 (state.overlap,'enable_two_batch_overlap',True),(state.overlap,'enable_single_batch_overlap',True),
 (parallel,'attn_dp_size',2),(parallel,'attn_dcp_size',2),(parallel,'attn_cp_size',2),(parallel,'pp_size',2),
 (parallel,'tp_size',2),(parallel,'attn_tp_size',4),(parallel,'moe_ep_size',4),
 (inputs,'dtype',real_torch.float32),(inputs,'is_cuda',False),(inputs,'ndim',3),
 (linear,'bias',object()),(linear,'input_is_parallel',False),(linear,'reduce_results',False),
 (linear.weight,'requires_grad',True),(linear.weight,'dtype',real_torch.float32),
 (linear,'quant_method',object()),(group,'_can_use_flashinfer_allreduce',lambda _:True),
 (group,'_resolve_outplace_all_reduce_method',lambda _:'pynccl'),(group,'ca_comm',None),
 (flags.SGLANG_ENABLE_MUSE_SPARK_WORKSPACE_REDUCE,'value',False),(flags.SGLANG_ENABLE_DETERMINISTIC_INFERENCE,'value',True)]
for obj,name,value in mutations:
    previous=getattr(obj,name);setattr(obj,name,value)
    assert borrow(group,inputs,8192,linear=linear) is None,(name,value)
    setattr(obj,name,previous)
for backend in ('cutedsl','gemv'):
    helper_scope['get_bf16_gemm_backend']=lambda:NS(value=backend)
    assert borrow(group,inputs,8192,linear=linear) is None
helper_scope['get_bf16_gemm_backend']=lambda:NS(value='torch')
assert borrow(group,inputs,8192,linear=linear) is not None

view=comm.multicast_workspace(128,4096)
seen=[]
scope['get_all_reduce_module']=lambda dtype,world:NS(all_reduce_workspace_=lambda obj,x:seen.append((dtype,world,obj,x)))
assert comm.all_reduce_workspace_(view) is view
assert seen==[(real_torch.bfloat16,8,comm.obj,view)]

cuh_path='python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh'
old=baseline(cuh_path);new=(root/cuh_path).read_text()
assert old.split('struct AllReduceKernel {')[0]==new.split('struct AllReduceKernel {')[0]
body=new.split('static Tensor run_workspace_(',1)[1].split('\n  static Tensor\n  run(',1)[0]
assert 'in.data_ptr() == ws.workspaces[pull.rank]' in body
assert 'all_reduce_2shot_pull_kernel<MC, T, kWorldSize, kUsePDL>' in body
assert 'LaunchKernel(comm.get_pull_multicast_blocks(), 512u, stream)' in body
assert 'memcpy' not in body and 'empty_like' not in body
for kernel_guard in ('PDLWaitPrimary<kUsePDL>();','barrier.arrive_relaxed(/*n=*/0);','barrier.arrive_rel_acq(/*n=*/1);'):
    assert kernel_guard in new
print(json.dumps(dict(selector_cases=selector_cases,helper_rejection_cases=len(mutations)+2,
                     alias_and_inplace_cases=3,existing_device_reduction_byte_identical=True,
                     cuda_executed=False),indent=2))
