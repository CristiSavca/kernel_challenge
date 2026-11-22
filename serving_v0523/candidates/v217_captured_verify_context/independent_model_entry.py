"""Replay Muse's complete model-to-logits entry and the V217 FC capture boundary on CPU."""
import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace as NS

import torch
from torch import nn

DIR = Path(__file__).resolve().parent
ROOT = DIR.parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
SRT = "python/sglang/srt/"
BASE = ROOT / "serving_v0523/sglang"
PARENT = (DIR.parent/"v215_decode_reduction_paired_context/patch.diff").read_text()
CHILD = (DIR/"patch.diff").read_text()
STAGE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
HASHES = {}


def read_source(path, child):
    path = SRT + path
    patch = CHILD if child else PARENT
    header = f"diff --git a/{path} b/{path}"
    section = patch.split(header,1)[1].split("\ndiff --git ",1)[0] if header in patch else ""
    if "new file mode" in section:
        raw = ""
    else:
        raw = subprocess.check_output(["git", "-C", BASE, "show", PIN+":"+path]).decode()
    if section:
        original, output, consumed = raw.splitlines(keepends=True), [], 0
        for line in section.splitlines():
            if line.startswith("@@"):
                start = max(0, int(re.match(r"@@ -(\d+)", line).group(1))-1)
                output.extend(original[consumed:start])
                consumed = start
            elif line[:1] in (" ", "-", "+") and not line.startswith(("--- ", "+++ ")):
                val = line[1:]+"\n"
                if line[0] in " -":
                    assert original[consumed] == val, (path,line,consumed)
                    consumed += 1
                if line[0] in " +":
                    output.append(val)
        raw = "".join(output+original[consumed:])
    if child and STAGE is not None:
        assert raw == (STAGE/path).read_text(), path
    HASHES[("v217/" if child else "v215/")+path] = sha256(raw.encode()).hexdigest()
    return raw


def source(path, child):
    return ast.parse(read_source(path,child))


def bind(tree, cls, name, env):
    container = tree if cls is None else next(n for n in tree.body if getattr(n,"name",None)==cls)
    node = next(n for n in container.body if getattr(n,"name",None)==name)
    node.decorator_list = []
    scope = dict(env)
    exec("from __future__ import annotations\n"+ast.unparse(node), scope)
    return scope[name]


@dataclass
class Output:
    next_token_logits: torch.Tensor
    hidden_states: torch.Tensor = None
    last_hidden_states: torch.Tensor = None
    mm_input_embeds: torch.Tensor = None
    dflash_verify_fc_shard: torch.Tensor = None


class Layer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.self_attn_input_norm = nn.Identity()

    def forward(self, pos, stream, attn, batch):
        out = stream * (1.0 + self.i/120) + torch.sin(pos.float())[:,None]/11
        return out, out.to(torch.bfloat16)


def make_target(child):
    model = source("models/muse_spark.py", child)
    logits = source("layers/logits_processor.py", child)
    pack = source("layers/aux_hidden_states.py", child)
    layernorm = source("layers/layernorm_sp.py", child)
    flags = NS(set=lambda *args:None)
    no_sp = lambda mode: False
    sp = NS(runs_sp=no_sp, maybe_exit_gather=bind(layernorm,None,"maybe_exit_gather",dict(runs_sp=no_sp)))
    env = dict(torch=torch, layernorm_sp=sp, get_forward=lambda:flags, ForwardBatch=type("ForwardBatch",(),{}),
               _autotune_run_lm_head=None, LogitsProcessorOutput=Output)
    env["pack_aux_hidden_states"]=bind(pack,None,"pack_aux_hidden_states",env)
    class Module(nn.Module):
        pass
    class Body(Module): pass
    class Wrapper(Module): pass
    class Target(Module): pass
    class Processor(Module): pass
    for clsname, typ in (("MuseSpark1xModel",Body),("MuseSparkModel",Wrapper),("MuseSparkTextForCausalLM",Target)):
        typ.forward=bind(model,clsname,"forward",env)
    for name in ("forward","_get_pruned_states","_get_hidden_states_to_store","_get_last_hidden_states_to_store"):
        setattr(Processor,name,bind(logits,"LogitsProcessor",name,env))
    p = Processor()
    p.expected_aux_hidden_states=None
    p.keep_last_hidden_states=False
    p._get_logits=lambda h,head,m: h.float().sum(dim=-1,keepdim=True)
    b=Body()
    b.residual_dtype=torch.float32
    b.operator_dtype=torch.bfloat16
    b.layers=nn.ModuleList(Layer(i) for i in range(8))
    b.layers_to_capture=[0,1,3,5,7]
    b.norm=nn.Identity()
    w=Wrapper()
    w.language_model=b
    t=Target()
    t.model=w
    t.lm_head=nn.Identity()
    t.logits_processor=p
    t.capture_aux_hidden_states=True
    return t


class Draft(nn.Module):
    def project_target_hidden(self,raw):
        return self.hidden_norm(self.fc(raw))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(21801)
    parent, child = make_target(False), make_target(True)
    mod = {"__name__":"v217_published_projection"}
    exec(read_source("layers/dflash_context_projection.py",True),mod)
    d=Draft()
    d.fc=nn.Linear(40,8,bias=False,dtype=torch.bfloat16)
    d.hidden_norm=nn.RMSNorm(8,dtype=torch.bfloat16)
    d.quantizable=False
    mode=NS(is_target_verify=lambda:True,is_dllm_extend=lambda:False,is_decode_or_idle=lambda:False)
    batch=NS(forward_mode=mode,capture_hidden_mode=NS(is_full=lambda:True,need_capture=lambda:True),
             mixed_verify_logits_indices=None, draft_extend_select_index=None, extend_return_logprob=False,
             mm_input_embeds=None, dflash_verify_raw_buffer=None)
    shared=torch.full((16,40),float("nan"),dtype=torch.bfloat16)
    ctx=mod["_PrefillContext"](d,d.fc,d.hidden_norm,d.project_target_hidden,NS(world_size=4,rank_in_group=1),
                               child,child.logits_processor)
    fc=mod["DFlashVerifyFCGraph"](ctx)
    fc.local_weight=d.fc.weight.detach()[2:4].clone()
    values=0
    ids=set()
    with torch.inference_mode():
        for rows in (16,8,16,1,3):
            before=shared.clone()
            batch.dflash_verify_raw_buffer=shared[:rows]
            inputs=torch.randn((rows,8))
            pos=torch.arange(rows)
            native=parent(None,pos,batch,input_embeds=inputs)
            fast=child(None,pos,batch,input_embeds=inputs)
            assert fast.hidden_states is batch.dflash_verify_raw_buffer
            assert fast.hidden_states.shape==native.hidden_states.shape==(rows,40)
            assert fast.hidden_states.view(torch.int16).equal(native.hidden_states.view(torch.int16))
            assert fast.next_token_logits.equal(native.next_token_logits)
            assert shared[rows:].view(torch.int16).equal(before[rows:].view(torch.int16))
            fc.capture(fast,batch)
            assert fast.dflash_verify_fc_shard.equal(d.fc(native.hidden_states)[:,2:4])
            raw_id=fast.hidden_states.untyped_storage().data_ptr()
            ids.add(raw_id)
            values+=rows*40
        assert len(ids)==1
        # The ordinary logits processor has a legitimate source preference that
        # replaces captured taps with pre-norm states. The packer cannot claim it.
        alt=child.logits_processor(torch.arange(rows),inputs.to(torch.bfloat16),child.lm_head,batch,
                                   fast.hidden_states,hidden_states_before_norm=inputs.to(torch.bfloat16))
        assert alt.hidden_states.shape==(rows,8)
        fc.capture(alt,batch)
        assert alt.dflash_verify_fc_shard is None
    print(json.dumps(dict(cases=5, raw_bfloat16_values=values, narrow_fc_equal=True,
          output_storage_ids=len(ids), padding_unchanged=True, before_norm_declines=True,
          source_sha256=HASHES, patch_sha256=sha256(CHILD.encode()).hexdigest(),
          torch=str(torch.__version__), device="CPU",
          scope="Pinned Muse inner/outer/token and generic logits forward, pruning and capture selection, candidate FC capture. Synthetic decoder layers and scalar logits calculation; no GPU graph or network."),indent=2))


if __name__=="__main__":
    main()

