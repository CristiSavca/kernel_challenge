"""Run the frozen MoE entry and both expert adapters with CPU tensors and modeled CUDA streams."""
import ast
import contextlib
import __future__
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
CHECK = json.loads((HERE / "validation.json").read_text())
ROOT = "python/sglang/srt/"
HUM = "sglang.srt.layers.muse_spark.muse_spark_humming_moe"
torch.set_num_threads(1)


class Graph:
    def __init__(self):
        self.nodes = []
        self.stream = "main"
        self.tails = {"main": set(), "router": set()}
        self.before = {}

    def event(self, name):
        assert name not in self.nodes, name
        self.before[name] = self.tails[self.stream].copy()
        self.nodes.append(name)
        self.tails[self.stream] |= {name}

    def has(self, *deps):
        assert set(deps) <= self.tails[self.stream], (self.stream, deps, self.tails)

    @contextlib.contextmanager
    def use(self, stream):
        old = self.stream
        self.stream = stream.name
        try:
            yield
        finally:
            self.stream = old


class Stream:
    def __init__(self, graph, name):
        self.g, self.name = graph, name

    def wait_stream(self, other):
        self.g.tails[self.name] |= self.g.tails[other.name]


def native(ref, path, cls, names, scope):
    src = subprocess.check_output(["git", "-C", str(STAGE), "show", f"{ref}:{ROOT}{path}"])
    parsed = ast.parse(src)
    body = parsed.body if cls is None else next(x for x in parsed.body if getattr(x, "name", None) == cls).body
    methods = [x for x in body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(methods) == len(names), (path, names)
    for method in methods:
        method.decorator_list = []
    program = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(program, path, "exec", flags=__future__.annotations.compiler_flag), scope)
    return {f.name: scope[f.name] for f in methods}


class Op:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, grid):
        return self.func


class Fixture:
    def __init__(self, ref, version):
        self.g = None
        self.fast = version == "V272"
        self.main = Stream(None, "main")
        self.side = Stream(None, "router")
        self.spy = []
        run = self
        self.plan_calls = []
        self.gathered = None
        self.prepared = None
        self.linear_weights = None

        class Norm(nn.Module):
            def forward(self, x):
                run.g.has("pool")
                run.g.event("norm")
                return torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight, 1.e-6).to(x.dtype)

        class Gate(nn.Module):
            def forward(self, x):
                run.g.event("gate")
                return x.float() @ run.linear_weights

            def forward_split2(self, x):
                return self.forward(x)

        class BaseFusedOp(nn.Module):
            def _forward_backend_dynamic(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        class Routing(BaseFusedOp):
            def forward(self, _a, logits, k, _b, want_packed):
                run.g.has("gate")
                run.g.event("topk")
                scores, ids = logits.topk(k)
                return torch.softmax(scores, 1), ids.int(), None

        class Expert(nn.Module):
            pass

        class MoE(nn.Module):
            pass

        self.op = Routing()
        self.op._forward_method = self.op._forward_backend_dynamic
        self.op.sol_topk_out = None
        self.op.is_torch_compile = False
        self.fused = SimpleNamespace(BaseFusedOp=BaseFusedOp, get_fused_op_backend=lambda:None)
        self.model = MoE()
        gate = self.model.gate = Gate()
        gate.e_score_correction_bias = self.op.expert_bias = torch.zeros(256)
        gate.weight = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.model._fused_routing_op = self.op
        self.model._routing_op = lambda: self.op
        self.model._side_stream = lambda: self.side
        self.model.pre_expert_norm = Norm()
        self.model.nvfp4_checkpoint = True
        self.model.router_in_fp32 = True
        self.model.tp_size = 1
        self.model.sp_replicate_expert_input = False
        experts = self.model.experts = Expert()
        experts.use_w4a16 = experts.use_humming = True
        experts._post_expert_norm = experts.post_expert_norm = nn.Identity()
        experts._post_expert_norm_eps = 1.e-6
        experts._post_expert_norm_weight_first = True
        experts.global_num_experts = 256
        experts.num_local_experts = 32
        experts.humming_decode_tunings = {key: [(2048, "fixed")] for key in ("w13", "w2")}
        experts.w13_global_scale = experts.w2_global_scale = torch.ones(32)

        def pool(x, w, bias):
            self.g.event("pool")
            return (x * w).to(torch.bfloat16)

        def finish(values, scores, *, slot_expert_ids, slot_row_indices, out, **kw):
            self.g.has("w2")
            self.g.event("combine")
            valid = slot_row_indices >= 0
            selected = torch.where(valid.unsqueeze(-1), values[slot_row_indices.clamp(min=0).long()], 0.).float()
            selected = selected * torch.rsqrt(selected.square().mean(-1, keepdim=True) + 1.e-6)
            out.copy_((selected * (scores * valid).unsqueeze(-1)).sum(1))
            return out

        def planner(ids, mapping, local, offsets, keys, inverse, slots, **kw):
            self.g.has("topk")
            self.g.event("plan")
            self.plan_calls.append(self.g.stream)
            flat = ids.flatten().long()
            mapped = mapping[flat] if kw["HAS_EXPERT_MAP"] else flat
            local.copy_(mapped.reshape_as(local))
            bucket = torch.where((mapped >= 0) & (mapped < kw["EXPERTS"]), mapped+1, 0)
            span = kw["SLOT_SPAN"]
            # Masked lanes occupy the remote prefix and participate in the stock sort.
            slots_padded = torch.arange(span)
            padded = torch.zeros(span, dtype=torch.int64)
            padded[:slots] = bucket
            ordered = torch.sort(padded * span + slots_padded).values
            keys.copy_(ordered[span-slots:])
            offsets.copy_(torch.bincount(bucket.long(), minlength=len(offsets)).cumsum(0))
            inv = torch.full((span,), -1, dtype=torch.int32)
            live = ordered >= span
            inv[ordered[live] % span] = (slots_padded[live] - (span - slots)).int()
            inverse.copy_(inv[:slots].view_as(inverse))
            self.prepared = [v.clone() for v in (local, offsets, keys, inverse)]

        def gather(x, keys, offsets, out, slots, **kw):
            self.g.has("norm", "plan")
            self.g.event("gather")
            out.zero_()
            start = int(offsets[0])
            out[start:] = x[(keys[start:] % kw["SLOT_SPAN"] // kw["TOPK"]).long()]
            self.gathered = out.clone()

        def projection(layer, *, inputs, outputs, expert_layout, sublayer_name, **kw):
            self.g.has("gather" if sublayer_name == "w13" else "silu")
            self.g.event(sublayer_name)
            first = int(expert_layout[0])
            if sublayer_name == "w13":
                width = inputs.shape[-1]
                outputs[first:, :width] = inputs[first:] * 1.25
                outputs[first:, width:] = inputs[first:] * .75
            else:
                outputs[first:] = inputs[first:] * .875

        def silu(x, out, keys, offsets, scales, span):
            self.g.has("w13")
            self.g.event("silu")
            first = int(offsets[0])
            a,b = x[first:].float().chunk(2,1)
            out[first:] = torch.nn.functional.silu(a) * b

        self.method = ModuleType("humming.layer")
        self.method.HummingMethod = SimpleNamespace(forward_layer=projection)
        self.hum = ModuleType(HUM)
        self.hum._MAX_DECODE_SLOTS = 2048
        scope = dict(torch=torch, _is_cpu=False, contextlib=contextlib,
                     MuseSpark1xRMSNorm=Norm, MuseSpark1xFp4Experts=Expert,
                     MuseSpark1xRouter=Gate, FusedMuseSparkRouting=Routing,
                     get_global_experts_capturer=lambda: None, _use_split_bf16_router=lambda n:n>=32,
                     get_forward=lambda: SimpleNamespace(sp_active=False), bf16_gemm_dispatch=pool,
                     triton=SimpleNamespace(next_power_of_2=lambda n:1 << (n-1).bit_length()),
                     _plan_grouped_decode=Op(planner), _gather_grouped_decode=Op(gather),
                     _MAX_DECODE_SLOTS=2048, _GROUPED_COMPUTE="native", _expert_scaled_silu_module=lambda:SimpleNamespace(run=silu),
                     fused_post_expert_norm_weighted_sum=finish)
        hum_file = "layers/muse_spark/muse_spark_humming_moe.py"
        funcs=["humming_decode_eligible","humming_grouped_decode_outputs"]
        if self.fast:
            funcs += ["allocate_decode_plan","fill_decode_plan"]
        for name,fn in native(ref,hum_file,None,funcs,scope).items():
            setattr(self.hum,name,fn)
        self.hum.local_humming_expert_ids = self.hum.humming_expert_outputs = None
        names=["_forward_experts"] + (["_capture_decode_plan"] if self.fast else [])
        for name, fn in native(ref,"models/muse_spark.py","MuseSpark1xMoE",names,scope).items():
            setattr(MoE,name,fn)
        for name,fn in native(ref,"layers/muse_spark/muse_spark_fp4_moe.py","MuseSpark1xFp4Experts",
                              ["forward","_forward_w4a16"],scope).items():
            setattr(Expert,name,fn)

    def execute(self, x, router, rank, k, mode):
        self.g = self.main.g = self.side.g = Graph()
        self.plan_calls.clear()
        self.linear_weights = router
        h = x.shape[1]
        exp = self.model.experts
        self.model.top_k = k
        exp.intermediate_size_per_partition = h
        self.model.pre_expert_norm.weight = nn.Parameter(torch.linspace(.75,1.25,h).bfloat16(), requires_grad=False)
        exp._post_expert_norm_weight = torch.ones(h, dtype=x.dtype)
        exp._expert_map = torch.where(torch.arange(256)//32 == rank, torch.arange(256)%32, -1).int()
        self.model.pre_expert_proj = SimpleNamespace(weight=torch.linspace(.5,1.,h).bfloat16())
        def observe(_mod, args):
            self.spy.append(next(a.shape for a in args if isinstance(a, torch.Tensor)))
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            extra = dict(fused_op=self.fused)
            stack.enter_context(patch.dict(sys.modules, {HUM:self.hum, "humming.layer":self.method,
                                                        "sglang.kernels":SimpleNamespace(**extra)}))
            if mode == "oot":
                stack.enter_context(patch.object(self.op, "_forward_method", lambda *a: self.op(*a)))
            if mode == "backend":
                stack.enter_context(patch.object(self.op, "forward_jit", self.op.forward, create=True))
            if mode == "forced":
                stack.enter_context(patch.object(self.fused, "get_fused_op_backend", return_value="plugin"))
            stack.enter_context(patch.object(torch.cuda,"is_available",return_value=True))
            stack.enter_context(patch.object(torch.cuda,"is_current_stream_capturing",return_value=mode!="eager"))
            stack.enter_context(patch.object(torch.cuda,"current_stream",return_value=self.main))
            stack.enter_context(patch.object(torch.cuda,"stream",side_effect=self.g.use))
            stack.enter_context(patch.object(torch.Tensor,"record_stream",lambda *_:None))
            modules = {"expert_hook":exp,"norm_hook":self.model.pre_expert_norm,"router_hook":self.op,"gate_hook":self.model.gate}
            if mode in modules:
                stack.callback(modules[mode].register_forward_pre_hook(observe).remove)
            y = self.model._forward_experts(x, grouped_decode=True)
        should_overlap = mode=="capture" and self.fast
        assert self.plan_calls == (["router"] if should_overlap else ["main"]), (self.plan_calls,mode)
        if should_overlap:
            assert "norm" not in self.g.before["plan"] and "plan" not in self.g.before["norm"]
            assert "pool" not in self.g.before["plan"] and "topk" not in self.g.before["norm"]
        else:
            assert {"norm", "topk"} <= self.g.before["plan"]
        self.g.has("pool","gate","topk","norm","plan","gather","w13","silu","w2","combine")
        return self.prepared,self.gathered,y


def main():
    versions=[Fixture(CHECK[k],v) for k,v in (("parent_tree","V68"),("tree","V272"))]
    gen=torch.Generator().manual_seed(783)
    counters=dict(calls=0, captured_parallel_calls=0, fallback_calls=0, metadata_words=0, staged_bf16=0, final_bf16=0)
    for rows,hidden,topk in [(3,32,3),(8,64,8),(32,96,8),(128,64,8),(256,64,8),(8,4096,8),(128,4096,8)]:
        router=torch.randn(hidden,256,generator=gen)
        for mode in ("capture","expert_hook","norm_hook","gate_hook","router_hook","eager","oot","backend","forced"):
            for rank in (0,3,7):
                x=torch.randn(rows,hidden,generator=gen).bfloat16()
                (p0,s0,y0),(p1,s1,y1)=[v.execute(x,router,rank,topk,mode) for v in versions]
                assert all(torch.equal(a,b) for a,b in zip(p0,p1))
                assert torch.equal(s0,s1) and torch.equal(y0,y1), (rows,hidden,mode,rank, (s0!=s1).sum(), (y0!=y1).sum(),y0.isnan().sum(),y1.isnan().sum())
                counters["calls"]+=1
                counters["captured_parallel_calls" if mode=="capture" else "fallback_calls"]+=1
                counters["metadata_words"]+=sum(v.numel() for v in p1)
                counters["staged_bf16"]+=s1.numel()
                counters["final_bf16"]+=y1.numel()
    print(json.dumps(dict(patch_sha256=CHECK["patch_sha256"],counts=counters,source="Frozen outer forward, public experts forward, W4A16 and grouped adapters",
        scope="CPU torch BF16 with modeled streams, emulated routing metadata and expert kernels. No accelerator execution or measured gain."),indent=2))


if __name__=="__main__":
    main()
