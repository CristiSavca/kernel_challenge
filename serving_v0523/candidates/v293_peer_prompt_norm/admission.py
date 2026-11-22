"""Run pre-normalization eligibility with real Torch parameters, modules and tensor views."""
import ast
import contextlib
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import patch,PropertyMock

import torch
from torch import nn

stage=Path(sys.argv[1]).resolve()
prefix="python/sglang/srt/"
peer=prefix+"layers/muse_spark/peer_prompt_pack.py"
norm=prefix+"layers/muse_spark/muse_spark_layers.py"
identity=prefix+"layers/muse_spark/muse_spark_humming_moe.py"
paths=(peer,norm,identity,prefix+"models/muse_spark.py")
def src(p):
    if (stage/p).is_file():return (stage/p).read_bytes()
    return subprocess.check_output(["git","-C",stage,"show","HEAD:"+p])
def load(p,name,scope):
    node=next(n for n in ast.parse(src(p)).body if getattr(n,"name",None)==name)
    exec("from __future__ import annotations\n"+ast.unparse(node),scope)
    return scope[name]
RMS=load(norm,"MuseSpark1xRMSNorm",dict(torch=torch,nn=nn))
W=load(peer,"PeerPromptInputs",dict(torch=torch,nn=nn))
a=ModuleType("sglang.srt.batch_invariant_ops");a.is_batch_invariant_mode_enabled=lambda:False
b=ModuleType("sglang.srt.layers.muse_spark.muse_spark_layers");b.MuseSpark1xRMSNorm=RMS
w=W.__new__(W);w.group=object();w.norm_module=object();w.storage=torch.empty((16,32),dtype=torch.bfloat16)
class Custom(RMS):pass
class Mark(torch.Tensor):pass
accepted=[];declined=[]
with patch.dict(sys.modules,{a.__name__:a,b.__name__:b}),patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),torch.no_grad():
    for case in ("frozen","plain","no_gain","uncached","float_gain","misaligned_gain","strided_gain","training","wrong_width","hook","global_hook","marked_gain","marked_shard","scalar","strided_input","misaligned_input","over_capacity","weight_none_with_cache","own_forward","own_getter","compiled","subclass","epsilon","autocast","grad","invariant","dispatch","torch_function","no_kernel"):
        with contextlib.ExitStack() as ctx:
            n=(Custom if case=="subclass" else RMS)(32,zero_centered_gamma=case!="plain").eval().bfloat16()
            n.precompute_effective_weight()
            x=w.storage[:3]
            if case=="no_gain":n.weight=None;n._frozen_weight=None
            if case=="uncached":n._frozen_weight=None
            if case=="float_gain":n._frozen_weight=n._frozen_weight.float()
            if case=="misaligned_gain":n._frozen_weight=torch.empty(33,dtype=x.dtype)[1:]
            if case=="strided_gain":n._frozen_weight=torch.zeros(64,dtype=x.dtype)[::2]
            if case=="training":n.train()
            if case=="wrong_width":n.hidden_size=16
            if case=="hook":ctx.enter_context(n.register_forward_hook(lambda *args:None))
            if case=="global_hook":ctx.enter_context(nn.modules.module.register_module_forward_pre_hook(lambda *args:None))
            if case=="marked_gain":n._frozen_weight=n._frozen_weight.as_subclass(Mark)
            if case=="marked_shard":x=x.as_subclass(Mark)
            if case=="scalar":x=x.view(-1)
            if case=="strided_input":x=torch.empty((32,3),dtype=x.dtype).t()
            if case=="misaligned_input":x=torch.empty(x.numel()+1,dtype=x.dtype)[1:].view_as(x)
            if case=="over_capacity":x=torch.empty((20,32),dtype=x.dtype)
            if case=="weight_none_with_cache":n.weight=None
            if case=="own_forward":n.forward=lambda x:x
            if case=="own_getter":n._get_effective_weight=lambda:None
            if case=="compiled":n._compiled_call_impl=lambda x:x
            if case=="epsilon":n.eps=None
            if case=="autocast":ctx.enter_context(patch.object(torch,"is_autocast_enabled",return_value=True))
            if case=="grad":ctx.enter_context(torch.enable_grad())
            if case=="invariant":ctx.enter_context(patch.object(a,"is_batch_invariant_mode_enabled",return_value=True))
            if case=="dispatch":ctx.enter_context(patch.object(torch._C,"_len_torch_dispatch_stack",return_value=1))
            if case=="torch_function":ctx.enter_context(patch.object(torch._C,"_is_torch_function_mode_enabled",return_value=True))
            if case=="no_kernel":ctx.enter_context(patch.object(w,"norm_module",None))
            binding=w.source_norm(x,n)
            if case in ("frozen","plain"):
                assert binding[0] is n._get_effective_weight() and binding[1]==n.eps
                accepted.append(case)
            else:
                assert binding is None,case;declined.append(case)
print(json.dumps(dict(accepted=accepted,declined=declined,sources={p:sha256(src(p)).hexdigest() for p in paths},
    scope="Production guard and native norm class, real Tensor/Parameter views, module hooks and inference controls. Simulated device capability."),indent=2))
