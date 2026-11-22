"""Exercise the staged FA3 selection and pool publication using genuine CPU tensors."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple

import torch


def definition(path, name):
    return next(x for x in ast.walk(ast.parse(path.read_text())) if getattr(x, "name", None) == name)


def load_function(path, name, env):
    body = definition(path, name)
    body.body = [s for s in body.body if not isinstance(s, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[body],type_ignores=[])
    exec(compile(module, str(path), "exec"),env)
    return env[name]


class VerifyFP8Store(NamedTuple):
    key: torch.Tensor
    value: torch.Tensor
    slots: torch.Tensor
    encodings: torch.Tensor
    capacity: torch.Tensor


class BaseFusedOp(torch.nn.Module):
    def _forward_backend_dynamic(self):
        return None

    def backend_eligible(self, backend, qkv):
        return True


class FusedQKNorm(BaseFusedOp):
    def __init__(self):
        super().__init__()
        self._num_q_heads, self._num_kv_heads, self._head_dim = 16, 2, 64
        self._forward_method = self._forward_backend_dynamic
        self.is_torch_compile = False
        self._dynamic_backend_candidates = ("triton",)

    def backend_eligible(self, backend, qkv, pos, cos, store):
        return store.key.shape == store.value.shape and store.key.shape[1] == 128


class RadixAttention(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.layer_id = i
        self.tp_k_head_num = self.tp_v_head_num = 2
        self.head_dim=self.v_head_dim=64
        self.is_cross_attention=False
        self.k_scale=self.v_scale=None


class MHATokenToKVPool:
    def __init__(self, n, page, vmm):
        self.size, self.page_size, self.post_capture_active = n,page,vmm
        self.dtype=torch.float8_e4m3fn
        self.store_dtype=torch.uint8
        self.kv_cache_layout="nhd"
        self.is_quantized_kv_cache=self.requires_physical_write_loc=False
        self.row_dim=self.v_row_dim=128
        self.k=torch.empty((n+page,2,64),dtype=self.dtype)
        self.v=torch.empty_like(self.k)
        self.device="cpu"
        self._post_capture_owner=NS(finalize=lambda x:None)
        initialize_capacity(self, vmm)
    def get_kv_buffer(self, i):
        return self.k,self.v


class SWAKVPool:
    def __init__(self, full,swa):
        self.full_kv_pool,self.swa_kv_pool=full,swa
        self.layers_mapping={0:(0,False),1:(0,True)}
    def get_kv_buffer(self,i):
        return (self.swa_kv_pool if i else self.full_kv_pool).get_kv_buffer(0)


class FlashAttentionBackend:
    def __init__(self,pool,slots,table):
        self.token_to_kv_pool=pool
        self.fa_skip_kv_cache=False
        self._muse_verify_encodings=table
        self.forward_metadata=NS(swa_out_cache_loc=slots)


def main(stage):
    global initialize_capacity
    root=stage/"python/sglang"
    backend=root/"srt/layers/attention/flashattention_backend.py"
    memory=root/"srt/mem_cache/memory_pool.py"
    members=definition(memory,"MHATokenToKVPool").body
    init=next(x for x in members if isinstance(x,ast.FunctionDef) and x.name=="__init__")
    targets=[s for s in init.body if isinstance(s,ast.Assign) and any(
        isinstance(t,ast.Attribute) and t.attr in ("_muse_verify_capacity","_muse_verify_slot_limit") for t in s.targets)]
    assert len(targets)==2
    init.name="initialize_capacity"
    init.args=ast.parse("def initialize_capacity(self,post_capture_active): pass").body[0].args
    init.body=targets
    scope=dict(torch=torch)
    exec(compile(ast.Module(body=[init],type_ignores=[]),str(memory),"exec"),scope)
    initialize_capacity=scope["initialize_capacity"]
    MHATokenToKVPool.enable_muse_verify_capacity=load_function(memory,"enable_muse_verify_capacity",scope)
    MHATokenToKVPool.finalize=load_function(memory,"_finalize_backing_tokens",scope)
    state=dict(cp=False,async_assert=False,prefill=False,forced=None,available=True)
    def write(batch,layer,swa_loc):
        return NS(loc=batch.out_cache_loc,swa_loc=swa_loc)
    env=dict(globals())
    env.update(KVWriteLoc=NS(for_layer=write),
               envs=NS(SGLANG_ENABLE_ASYNC_ASSERT=NS(get=lambda:state["async_assert"])),
               get_fused_op_backend=lambda:state["forced"],is_cp_active=lambda batch:state["cp"],
               is_in_prefill_graph_scope=lambda:state["prefill"],KernelBackend=NS(TRITON="triton"),
               can_use_store_cache=lambda *args:state["available"])
    choose=load_function(backend,"muse_verify_fp8_store",env)
    module=root/"kernels/ops/attention/muse_spark_qk_norm_rope_store.py"
    table=load_function(module,"bf16_e4m3_encodings",dict(torch=torch))("cpu")
    assert len(table)==65536
    good,bad=0,0
    for vmm in (False,True):
        full,swa=(MHATokenToKVPool(512,1,vmm),MHATokenToKVPool(369,1,vmm))
        for owner in (full,swa):
            owner.enable_muse_verify_capacity()
            ptr=owner._muse_verify_capacity.data_ptr()
            owner.enable_muse_verify_capacity()
            assert ptr==owner._muse_verify_capacity.data_ptr()
            assert owner._muse_verify_capacity.item()==(1 if vmm else owner.size+1)
        for rows in (8,32,256):
            q=torch.empty((rows,2304),dtype=torch.bfloat16)
            positions=torch.arange(rows)
            cos=torch.empty((rows,64))
            ls=torch.arange(rows,dtype=torch.int32)
            ls[0]=0
            ss=ls.flip(0).to(torch.int64)
            for owner_pool,layer_id in ((full,0),(SWAKVPool(full,swa),0),(SWAKVPool(full,swa),1)):
                be=FlashAttentionBackend(owner_pool,ss,table)
                norm=FusedQKNorm()
                layer=RadixAttention(layer_id)
                batch=NS(out_cache_loc=ls,dcp_kv_mask=None)
                use=lambda:choose(be,norm,layer,batch,q,positions,cos)
                args=use()
                assert args is not None
                owner=swa if layer_id else full
                expected=ss if layer_id else ls
                assert args.key.data_ptr()==owner.k.data_ptr()
                assert args.value.data_ptr()==owner.v.data_ptr()
                assert args.slots is expected and args.capacity is owner._muse_verify_capacity
                assert args.encodings is table
                good+=1
                for obj,attr,wrong in ((layer,"head_dim",512),(layer,"k_scale",1),(layer,"v_scale",2),
                        (layer,"is_cross_attention",True),(layer,"tp_k_head_num",8),(owner,"requires_physical_write_loc",True),
                        (owner,"kv_cache_layout","hnd"),(owner,"is_quantized_kv_cache",True),(owner,"store_dtype",torch.bfloat16),
                        (owner,"_muse_verify_capacity",None),(batch,"dcp_kv_mask",ls),(be,"fa_skip_kv_cache",True),
                        (be,"_muse_verify_encodings",None),(norm,"is_torch_compile",True),(norm,"_forward_method",None),
                        (norm,"_dynamic_backend_candidates",("helion","triton"))):
                    old=getattr(obj,attr)
                    setattr(obj,attr,wrong)
                    assert use() is None,attr
                    setattr(obj,attr,old)
                    bad+=1
                for name in ("cp","async_assert","prefill","forced","available"):
                    old=state[name];state[name]=not bool(old)
                    assert use() is None,name
                    state[name]=old;bad+=1
                for obj,method in ((owner,"set_kv_buffer"),(owner,"_store_kv_layer"),(norm,"_forward_triton"),(layer,"forward")):
                    setattr(obj,method,lambda:None)
                    assert use() is None
                    delattr(obj,method);bad+=1
                for obj in (norm,layer):
                    h=obj.register_forward_pre_hook(lambda m,inp:None)
                    assert use() is None
                    h.remove();bad+=1
                hook=torch.nn.modules.module.register_module_forward_hook(lambda *args:None)
                assert use() is None
                hook.remove();bad+=1
                for p,c in ((positions.float(),cos),(positions,cos.double()),(positions[:-1],cos)):
                    assert choose(be,norm,layer,batch,q,p,c) is None;bad+=1
                assert choose(be,norm,layer,batch,q,None,None) is not None;good+=1
        if vmm:
            for owner,capacity in ((full,365),(swa,91)):
                ptr=owner._muse_verify_capacity
                def failure(x):
                    raise ValueError("backing refused")
                owner._post_capture_owner.finalize=failure
                try: owner.finalize(capacity)
                except ValueError: pass
                else: raise AssertionError()
                assert ptr.item()==1
                owner._post_capture_owner.finalize=lambda x:None
                owner.finalize(capacity)
                assert ptr is owner._muse_verify_capacity and ptr.item()==capacity+owner.page_size
    return dict(torch=torch.__version__,selected_physical_paths=good,guarded_native_fallbacks=bad,
                full_swa_vmm_failure_and_publication=True,distinct_full_swa_bounds=[366,92],
                release_source_sha256=hashlib.sha256(backend.read_bytes()).hexdigest())


if __name__=="__main__":
    args=argparse.ArgumentParser();args.add_argument("stage",type=Path)
    print(json.dumps(main(args.parse_args().stage),indent=2))
