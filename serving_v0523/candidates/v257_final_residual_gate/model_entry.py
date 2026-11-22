"""Reach the last decoder block from the normal Muse body and compare its residual paths."""
import ast
from contextlib import ExitStack, nullcontext
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MODEL = "python/sglang/srt/models/muse_spark.py"
LAYERS = "python/sglang/srt/layers/muse_spark/muse_spark_layers.py"
KERNEL = "python/sglang/kernels/ops/layernorm/muse_spark_norm_residual.py"
IMPORT = "sglang.kernels.ops.layernorm.muse_spark_norm_residual"


def method(source, cls, name, env):
    container = ast.parse(source)
    if cls:
        container = next(n for n in container.body if getattr(n, "name", "") == cls)
    node = next(n for n in container.body if getattr(n, "name", "") == name)
    exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return env[name]


def run(stage):
    manifest = json.loads((HERE / "validation.json").read_bytes())
    assert sha256((HERE / "patch.diff").read_bytes()).hexdigest() == manifest["patch_sha256"]
    sources = {}
    for path in (MODEL, LAYERS, KERNEL):
        raw = (stage / path).read_bytes()
        assert raw == subprocess.check_output(["git", "-C", stage, "show", manifest["tree"] + ":" + path])
        sources[path] = raw.decode()
    parent = subprocess.check_output(["git", "-C", stage, "show", manifest["parent_tree"] + ":" + LAYERS]).decode()
    assert sources[MODEL].encode() == subprocess.check_output(["git","-C",stage,"show",manifest["parent_tree"]+":"+MODEL])
    env = dict(torch=torch, F=F, _is_cpu=False)
    gate_new = method(sources[LAYERS], "MuseSpark1xNormPreservingResidualConnection", "forward", env)
    gate_old = method(parent, "MuseSpark1xNormPreservingResidualConnection", "forward", env)
    post = method(sources[LAYERS], "MuseSpark1xRMSNormWithInputScale", "forward", env)
    norm = method(sources[LAYERS], "MuseSpark1xRMSNorm", "forward", env)
    block = method(sources[MODEL], "MuseSpark1xDecoderLayer", "forward", env)
    model = method(sources[MODEL], "MuseSpark1xModel", "forward", env)
    env.update(get_forward=lambda:NS(set=lambda *_:None))
    calls = []
    class HostGate:
        def __getitem__(self, grid):
            def launch(stream, branch, alpha, beta, output, count, width, threads, **kw):
                assert kw == {"enable_fp_fusion":False} and grid == ((count + 511)//512,)
                calls.append([count,width])
                output.copy_(alpha * stream + beta * branch.to(stream.dtype))
            return launch
    env.update(triton=NS(cdiv=lambda x,y:(x+y-1)//y), _gated_residual_kernel=HostGate())
    probe = ModuleType(IMPORT)
    probe.gated_residual = method(sources[KERNEL], None, "gated_residual", env)
    cases = []
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        stack.enter_context(patch.dict(sys.modules, {IMPORT:probe}))
        stack.enter_context(patch.object(torch.Tensor, "is_cuda", property(lambda x:True)))
        stack.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=(9,0)))
        stack.enter_context(patch.object(torch.cuda, "device", return_value=nullcontext()))
        for idx,(mode,logical,width) in enumerate([("prefill",16384,8192),("mixed",16640,128),("verify",128,1024),("prefill",1506,63)]):
            sp = mode != "verify"
            rows = (logical+7)//8 if sp else logical
            # The source selects rank-owned rows. The test data repeats a resident channel
            # vector to avoid allocating replicated 16K by 8192 activations on a CPU.
            vec = (torch.arange(width,dtype=torch.float32) % 81-40).to(torch.bfloat16)
            embeddings=vec.expand(logical,width)
            env["layernorm_sp"] = NS(runs_sp=lambda mode:sp, sp_entry_scatter=lambda x:x[:rows])
            ids = torch.arange(logical)
            taps = [1,16,30,44,59]
            draws = torch.Generator().manual_seed(252+idx)
            gate = NS(_alpha=torch.rand(1,width,generator=draws),_beta=torch.rand(1,width,generator=draws))
            cp = NS(hidden_size=width,_get_effective_weight=lambda:None,eps=1.e-5)
            post_fn = lambda x:post(cp,x)
            role = NS(forward_mode=mode)
            def serve(gate_impl):
                counts = {"terminal_experts":0,"terminal_rows":0}
                class Middle:
                    def self_attn_input_norm(self,stream):return stream.to(torch.bfloat16)
                    def __call__(self,positions,stream,attn_input,forward_batch):return stream,attn_input
                def expert(x,_router,_batch):
                    counts["terminal_experts"]+=1;counts["terminal_rows"]+=x.shape[0]
                    return x
                final=NS(_router_dtype=None,_fused_ffn_boundary=None,is_moe_layer=True,operator_dtype=torch.bfloat16,
                    self_attn=lambda positions,x,fb:x,
                    _fused_attn_boundary=lambda stream,act,**kw:(act,stream,None),
                    mlp=expert,post_feedforward_layernorm=post_fn,post_feedforward_residual_gate=lambda x,y:gate_impl(gate,x,y))
                body=NS(embed_tokens=lambda _:embeddings,embed_norm=None,residual_dtype=torch.float32,operator_dtype=torch.bfloat16,
                    layers=[*[Middle() for _ in range(61)],lambda *args:block(final,*args)],
                    layers_to_capture=taps,norm=lambda x:norm(cp,x))
                hidden,captured = model(body,ids,ids,role)
                assert counts["terminal_rows"]==rows and counts["terminal_experts"]==1 and len(captured)==5
                return hidden,captured
            old,old_taps=serve(gate_old)
            before=len(calls);new,new_taps=serve(gate_new)
            assert len(calls)==before+1 and calls[-1]==[rows*width,width]
            assert torch.equal(old.view(torch.int16),new.view(torch.int16))
            assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(old_taps,new_taps))
            cases.append(dict(role=mode,logical_rows=logical,owner_rows=rows,width=width,
                hidden_bf16_words=old.numel(),tap_bf16_words=sum(x.numel() for x in old_taps),terminal_launches=1))
            del old,new,old_taps,new_taps
        gate=NS(_alpha=torch.ones(1,4),_beta=torch.ones(1,4))
        base=torch.arange(28,dtype=torch.float32).reshape(7,4)
        b=base.bfloat16(); before=len(calls)
        for data,branch,factors in [(base,b.float(),gate),(base[::2],b[::2],gate),(base,b,NS(_alpha=torch.ones(1),_beta=torch.ones(1)))]:
            assert torch.equal(gate_old(factors,data,branch),gate_new(factors,data,branch))
        assert len(calls)==before
    # CPU ordinary dispatch also preserves the native path. Actual new kernel exercised separately.
    prior=len(calls); assert torch.equal(gate_old(gate,base,b),gate_new(gate,base,b)); assert len(calls)==prior
    print(json.dumps(dict(patch_sha256=manifest["patch_sha256"],cases=cases,negative_cases=4,
        scope="Native model/terminal-layer/gate source on CPU with synthetic preceding layers, experts and SP ownership. Torch stands in for the launched gate. The independent Triton receipt exercises device code on its interpreter."),indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(Path(sys.argv[1]).resolve())
