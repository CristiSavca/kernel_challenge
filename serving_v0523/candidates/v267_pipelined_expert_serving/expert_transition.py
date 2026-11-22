"""Run the real prompt entry and then small verification on the same expert object and Gloo ranks."""
import ast
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
V258 = HERE.parent / "v258_compact_prompt_route_plan/dispatch_check.py"


def transition(model, tree, reference, state, stats, definitions, hum, moe, root, fake_triton, mod, finalize):
    layer = model.experts
    width = layer.hidden_size
    layer.humming_decode_tunings = {"w13":[(2048,"z")],"w2":[(2048,"z")]}
    layer.w13_weight = torch.ones(1,1)
    layer.w13_weight_scale = torch.ones(1,1)
    layer.w13_global_scale = torch.ones(layer.num_local_experts)
    bindings = tuple((x.data_ptr(),x.shape) for x in (layer.w13_weight,layer.w13_weight_scale))
    record = []
    class Kernel:
        def __init__(self,op):
            self.op=op
        def __getitem__(self,grid):
            return self.op

    def plan(ids,mapping,local,offsets,keys,inverse,slots,**kw):
        expert = torch.where(ids<0,-1,mapping[ids.clamp_min(0).long()])
        local.copy_(expert)
        flat = local.flatten()
        span = kw["SLOT_SPAN"]
        bucket = torch.zeros(span,dtype=torch.int64)
        bucket[:slots] = torch.where(flat < 0,0,flat+1)
        sorted_keys = torch.sort(bucket*span+torch.arange(span)).values[span-slots:]
        keys.copy_(sorted_keys.int())
        offsets.copy_(torch.tensor([int((flat<n).sum()) for n in range(layer.num_local_experts+1)],dtype=torch.int32))
        inverse.fill_(-1)
        live = sorted_keys//span>0
        inverse.view(-1)[sorted_keys[live]%span]=torch.arange(slots,dtype=torch.int32)[live]
        record.append("plan")
    def gather(x,keys,offsets,out,slots,**kw):
        first = int(offsets[0])
        out[first:] = x[(keys[first:].long()%kw["SLOT_SPAN"])//kw["TOPK"]]
        record.append("gather")
    class Launch:
        def run(self,x,out,keys,offsets,scales,maps,handle,shared,tile,sms):
            direct=len(x)!=len(keys)
            record.append("direct" if direct else "expanded")
            if direct and state["reject_direct"]:
                return False
            first=int(offsets[0])
            chosen=(keys[first:].long() % fake_triton.next_power_of_2(len(keys))) // (len(keys)//len(x))
            out[first:]=x[chosen] if direct else x[first:]
            return True
    class Arithmetic:
        @staticmethod
        def forward_layer(layer, *, inputs,outputs,expert_layout,sublayer_name,**kw):
            assert sublayer_name=="w2"
            first=int(expert_layout[0]);outputs[first:]=inputs[first:].flip([1]);record.append("w2")
    def combine(values,scores,**kw):
        finalize(values,scores,**kw)
        return kw["out"]

    stage=dict(torch=torch,MuseSpark1xFp4Experts=type(layer),triton=fake_triton,_MAX_DECODE_SLOTS=2048,_GROUPED_COMPUTE="x",
               _plan_grouped_decode=Kernel(plan),_gather_grouped_decode=Kernel(gather))
    definitions(tree,hum,stage,{"humming_decode_eligible","humming_grouped_decode_outputs"})
    public=dict(torch=torch,Optional=__import__("typing").Optional,nn=torch.nn,fused_post_expert_norm_weighted_sum=combine)
    definitions(tree,moe,public,{"_forward_w4a16"},"MuseSpark1xFp4Experts")
    gate={}
    gate_path = root+"layers/muse_spark/decode_gate_up.py"
    definitions(tree,gate_path,gate,{"run_gate_up"})
    hmod=mod(**{name:stage[name] for name in ("humming_decode_eligible","humming_grouped_decode_outputs")},
             humming_expert_outputs=None,local_humming_expert_ids=None)
    modules={"humming.layer":mod(HummingMethod=Arithmetic),
             "sglang.srt.layers.muse_spark.decode_gate_up":mod(run_gate_up=gate["run_gate_up"]),
             "sglang.srt.layers.muse_spark.muse_spark_humming_moe":hmod}
    keys = ("fresh","reject","hot","remote")
    with patch.dict(sys.modules,modules):
        for index,label in enumerate(keys):
            # Ragged token counts and changing route ownership follow the same prompt leg.
            rows=7 if index==1 else 8
            x=((torch.arange(rows*width).reshape(rows,width)*3+index)%41-20).bfloat16()
            ids=(torch.arange(rows*8).reshape(rows,8)+index) % (dist.get_world_size()*4)
            if label=="hot": ids.fill_(3)
            if label=="remote": ids.fill_(-1)
            scores=(torch.arange(rows*8).reshape(rows,8)%11+1).float()/19
            layer.humming_decode_gate_up=(Launch(),None,bindings,{"z":(None,42,100,8)},132)
            state["reject_direct"] = label=="reject"
            record.clear()
            out=public["_forward_w4a16"](layer,x,scores,ids.int(),layer._post_expert_norm,grouped_decode=True)
            wanted=(["plan","gather","expanded","w2"] if reference else
                    ["plan","direct","gather","expanded","w2"] if state["reject_direct"] else ["plan","direct","w2"])
            assert record==wanted,(label,record,wanted)
            dist.all_reduce(out)
            if reference:
                state["verify_"+label]=out.clone()
            else:
                assert torch.equal(out.view(torch.int16),state["verify_"+label].view(torch.int16))
                stats["paired_successive_verify_calls"]+=1
                stats["paired_successive_BF16_words"]+=out.numel()
                stats["rejected_direct_recovered"]+=int(state["reject_direct"])


def main():
    # Reuse V258's fixtures; change their comparator to V248, the integrated parent.
    source=V258.read_text()
    module=ast.parse(source)
    assert isinstance(module.body[-1],ast.Expr) and isinstance(module.body[-1].value,ast.Call)
    ns={"__file__":str(V258),"__name__":"fixture"}
    exec(compile(ast.Module(module.body[:-1],[]),str(V258),"exec"),ns)
    distributed=ns["source"].replace("v256_window_backed_mixed_serving","v248_shared_verify_raster")
    needle="                outs.append(run(model,x,scores,ids))"
    assert distributed.count(needle)==1
    distributed=distributed.replace(needle,needle+'''
                if kind=="mixed":
                    transition(model,tree,tree==PARENT['tree'],state,stats,definitions,HUM,MOE,ROOT,
                               fake_triton,module,finalize)
''')
    settled="                counts.append(stats['gathers']-before)"
    assert distributed.count(settled)==1
    distributed=distributed.replace(settled,settled+"""
                if kind=='mixed':
                    # Re-enter the same prompt path *after* a refused direct map and later small verifications.
                    again=run(model,-x,scores.flip([1]),ids.flip([1]))
                    if tree==PARENT['tree']:
                        state['prompt_reentry']=again.clone()
                    else:
                        assert torch.equal(again.view(torch.int16),state['prompt_reentry'].view(torch.int16))
                        stats['post_verify_prompt_words']+=again.numel()
                    assert not torch.equal(again.view(torch.int16),outs[-1].view(torch.int16))
""")
    check="stats['barriers']-barriers==4*fast"
    assert distributed.count(check)==1
    distributed=distributed.replace(check,"stats['barriers']-barriers==4*(fast+int(kind=='mixed'))")
    # Older binding assumed all tensors contained nonnegative IDs or -1; the
    # synthetic maps here retain that contract. Long prompt arithmetic uses the V258 stand-ins.
    env={"__file__":str(ns["ORIGINAL"]),"__name__":"__main__","transition":transition}
    exec(compile(distributed,str(ns["ORIGINAL"]),"exec"),env)


if __name__=="__main__":
    main()
