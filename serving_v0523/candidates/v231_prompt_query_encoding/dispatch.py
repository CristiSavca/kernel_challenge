"""Exercise V231's admission method with tensor metadata and asymmetric boundary cases."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
from unittest.mock import patch

import torch


class BaseFusedOp(torch.nn.Module):
    def _forward_backend_dynamic(self): pass


class FusedQKNorm(BaseFusedOp):
    def __init__(self):
        super().__init__()
        self._forward_method=self._forward_backend_dynamic
        self._dynamic_backend_candidates=("triton",)
        self._num_q_heads,self._num_kv_heads,self._head_dim=16,2,64
        self.is_torch_compile=False

    def backend_eligible(self, backend, x, pos, cos, store):
        return (backend == "triton" and x.ndim==2 and store is None and
                (pos is None or cos.dtype==torch.float32 and cos.shape[-1]==self._head_dim))


class RadixAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim=self.v_head_dim=64
        self.tp_q_head_num,self.tp_k_head_num=16,2
        self.is_cross_attention=False
        self.attn_type="decoder"
        self.use_prefill_attention_wrapper=True


class FlashAttentionBackend:
    def __init__(self):
        self._muse_verify_encodings=torch.empty(65536, device="meta", dtype=torch.uint8)
        self._direct_fp8_prompt=True
        self.fa_skip_kv_cache=self.kv_cache_is_mxfp8=False


if __name__=="__main__":
    stage=Path(sys.argv[1])
    directory=Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).resolve().parent
    path=stage/"python/sglang/srt/layers/attention/flashattention_backend.py"
    manifest=json.loads((directory/"validation.json").read_bytes())
    source=path.read_bytes()
    f=next(n for n in ast.walk(ast.parse(source)) if getattr(n,"name","")=="muse_prompt_query_encodings")
    f.body=[n for n in f.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    flags={}
    scope=dict(torch=torch,BaseFusedOp=BaseFusedOp,FusedQKNorm=FusedQKNorm,RadixAttention=RadixAttention,
        FlashAttentionBackend=FlashAttentionBackend,ForwardMode=NS(EXTEND="extend"),AttentionType=NS(DECODER="decoder"),
        KernelBackend=NS(TRITON="triton"),get_fused_op_backend=lambda:flags.get("override"),
        is_in_full_prefill_graph=lambda:flags.get("full",False),is_cp_active=lambda b:flags.get("cp",False),
        _trace_enabled=False)
    exec(compile(ast.Module([f],[]),str(path),"exec"),scope)
    guard=scope["muse_prompt_query_encodings"]
    admitted=declined=0
    tests = ["native","q_scale_nonunit","short","wrong_kv_heads","wrong_q_heads","cross","full","cp",
        "trace","mxfp8","skipcache","layout","same_meta_other_dtype","not_extend","dcp","lse",
        "gradient","deterministic","radix_hook","norm_hook","output_hook","global_pre_hook","backend_override",
        "runtime_override","instance_dispatch","dynamo","alternative_priority","graph_backend_override",
        "no_table","wrong_device","unobserved_backend","wide_head","pos_dtype","pos_device","pos_shape",
        "cos_dtype","cos_pitch","output_width","row_pitch_overflow","native_eligibility_denial"]
    for rope in (False,True):
        for case in tests:
            flags.clear()
            owner=FlashAttentionBackend()
            norm=FusedQKNorm()
            layer=RadixAttention()
            batch=NS(forward_mode="extend",mha_return_lse=False,dcp_kv_mask=None)
            n=4096
            x=torch.empty((n,1280), device="meta",dtype=torch.bfloat16)
            pos=torch.empty((n,),device="meta",dtype=torch.int64) if rope else None
            cos=torch.empty((10000,64),device="meta") if rope else None
            if case.startswith("pos_") or case.startswith("cos_"):
                if not rope: continue
            hook=None
            if case=="q_scale_nonunit": layer.k_scale,layer.v_scale=7.0,0.25
            elif case=="short": x=x[:4095]
            elif case=="wrong_q_heads": norm._num_q_heads=8
            elif case=="wrong_kv_heads": norm._num_kv_heads=8
            elif case=="cross": layer.is_cross_attention=True
            elif case in {"cp","full"}: flags[case]=True
            elif case=="mxfp8": owner.kv_cache_is_mxfp8=True
            elif case=="skipcache": owner.fa_skip_kv_cache=True
            elif case=="layout": owner._direct_fp8_prompt=False
            elif case=="same_meta_other_dtype": x=x.float()
            elif case=="not_extend": batch.forward_mode="verify"
            elif case=="dcp": batch.dcp_kv_mask=object()
            elif case=="lse": batch.mha_return_lse=True
            elif case=="radix_hook": hook=layer.register_forward_pre_hook(lambda *args:None)
            elif case=="norm_hook": hook=norm.register_forward_pre_hook(lambda *args:None)
            elif case=="output_hook": hook=norm.register_forward_hook(lambda *args:None)
            elif case=="global_pre_hook": hook=torch.nn.modules.module.register_module_forward_pre_hook(lambda *args:None)
            elif case=="backend_override": flags["override"]="native"
            elif case=="runtime_override": owner.prepare_paged_mha_query=lambda *a:None
            elif case=="instance_dispatch": norm.forward_triton=lambda *a:None
            elif case=="dynamo": norm.is_torch_compile=True
            elif case=="alternative_priority": norm._dynamic_backend_candidates=("helion","triton")
            elif case=="graph_backend_override": norm._forward_method=lambda:None
            elif case=="no_table": owner._muse_verify_encodings=None
            elif case=="wrong_device": x=torch.empty((n,1280),dtype=torch.bfloat16)
            elif case=="unobserved_backend": layer.attn_type="encoder_only"
            elif case=="wide_head": layer.head_dim=layer.v_head_dim=norm._head_dim=128
            elif case=="pos_dtype": pos=pos.float()
            elif case=="pos_device": pos=torch.empty((n,),dtype=torch.int64)
            elif case=="pos_shape": pos=pos[1:]
            elif case=="cos_dtype": cos=cos.to(torch.bfloat16)
            elif case=="cos_pitch": cos=cos.t()
            elif case=="output_width": x=x[:,:1279]
            elif case=="row_pitch_overflow": x=torch.empty_strided(x.shape,(2**21,1),dtype=x.dtype,device=x.device)
            elif case=="native_eligibility_denial": norm.backend_eligible=lambda *args:False
            scope["_trace_enabled"]=case=="trace"
            with torch.set_grad_enabled(case=="gradient"), patch.object(torch,"are_deterministic_algorithms_enabled",return_value=case=="deterministic"):
                result=guard(owner,norm,layer,batch,x,pos,cos)
            if hook is not None: hook.remove()
            expected=case in ("native","q_scale_nonunit")
            assert (result is not None)==expected,(case,rope)
            if expected:
                assert result is owner._muse_verify_encodings
                admitted+=1
            else: declined+=1
    print(json.dumps(dict(candidate=manifest["candidate"],patch_sha256=manifest["patch_sha256"],
        source_sha256=hashlib.sha256(source).hexdigest(), admitted=admitted,declined=declined,cases=tests,
        tensor_device="meta and CPU",physical_H200=False),indent=2))
