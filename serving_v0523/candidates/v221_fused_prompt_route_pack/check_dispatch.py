"""Run both grouped production callbacks plus finalization, modeling accelerator launches with CPU tensors."""
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch
import torch

here=Path(__file__).resolve().parent
subject=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else here
manifest=json.loads((subject/'validation.json' if subject.is_dir() else subject).read_bytes())
parent=json.loads((here.parent/'v219_verify_graph_sliding_tiles'/'validation.json').read_bytes())
base=runpy.run_path(str(next(here.parent.glob('v207_*/check_callback.py'))))
f=base['fixture']; host=base['host']; layer=f['layer']
events=Counter()

class Pack:
    def __getitem__(self,grid):
        def launch(hidden,slots,inputs,rowmap,first,live,total,rs,cs,**kw):
            assert grid==(live+(total+255)//256,) and kw['num_warps']==4
            assert total==rowmap.numel() and slots.shape==(total,) and first+live==total
            assert (rs,cs)==hidden.stride() and rowmap.is_contiguous() and inputs.is_contiguous()
            assert kw['BLOCK_H']==min(4096,1<<(hidden.size(1)-1).bit_length())
            assert kw['BLOCK_MAP']==256 and kw['HIDDEN']==hidden.size(1)
            assert kw['TOPK']==rowmap.size(1)
            inputs.copy_(hidden[(slots[first:]//kw['TOPK']).long()])
            rowmap.view(-1)[slots]=torch.arange(-first,live,device=slots.device,dtype=rowmap.dtype).clamp_min(-1)
            events['pack']+=1
        return launch

def indexed(owner,x,ids):
    events['indexed']+=1
    return base['indexed'](owner,x,ids)

f['scope']['_post_expert_norm_weighted_sum_kernel']=base['Finalizer']()
groups=[]
for tree in (parent['tree'],manifest['tree']):
    env=dict(host['env'],triton=NS(next_power_of_2=lambda n:1<<(n-1).bit_length(),
              cdiv=lambda v,d:(v+d-1)//d),_pack_prompt_routes=Pack(),humming_expert_outputs=indexed)
    group=base['definition'](tree,base['hum'],'humming_grouped_expert_outputs',env)
    combine=base['definition'](tree,base['moe'],'_grouped_humming_and_combine_impl',f['scope'].copy(),'MuseSpark1xFp4Experts')
    groups.append((group,combine))

def runs():
    records=[]
    # The actual grouped entry admits >=8192 rows. Slots include repeated experts/routes, remote rows and real top-8.
    for count,route in ((8192,'mixed'),(8201,'skew'),(16384,'mixed'),(8192,'empty'),(8192,'indexed')):
        x=host['rand'](count,128)
        if count==8201: x=x.T.contiguous().T
        ids=torch.full((count,8),-1,dtype=torch.int32)
        if route=='mixed':
            ids[::113,1]=3;ids[::113,6]=3;ids[::127,3]=0
        if route=='skew':ids[::47,4]=7
        if route=='indexed':ids[:]=torch.arange(8,dtype=ids.dtype)
        scores=host['rand'](count,8).float()
        layer.intermediate_size_per_partition=64 if route=='indexed' else 128
        for gate_mode,down_mode in (('paired','paired'),('disabled','ordinary')):
            layer.humming_prompt_gate_up=host['pstate'](gate_mode)
            layer.humming_prompt_down=f['bind']() if down_mode=='paired' else None
            for weight_first in (False,True):
                layer._post_expert_norm_weight_first=weight_first
                out=[]; masks=[]
                for ix,(group,combine) in enumerate(groups):
                    def capture(*args,**kw):
                        y,m=group(*args,**kw);masks.append(m.clone());return y,m
                    f['model_hum'].humming_grouped_expert_outputs=capture
                    layer._grouped_humming_and_combine=MethodType(combine,layer)
                    prior=events.copy()
                    out.append(f['forward'](x,scores,ids))
                    assert events['pack']-prior['pack']==int(ix==1 and route not in ('empty','indexed'))
                    assert events['indexed']-prior['indexed']==int(route=='indexed')
                assert len(masks)==2 and torch.equal(*masks)
                assert torch.equal(*(z.contiguous().view(torch.int16) for z in out))
                records.append(dict(rows=count,route=route,live=int((ids>=0).sum()),gate=gate_mode,down=down_mode,
                                    weight_first=weight_first,output_words=out[0].numel(),slots=masks[0].numel()))
    return records

with torch.inference_mode(),patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
    records=runs()
print(json.dumps(dict(physical_H200=False,patch_sha256=manifest['patch_sha256'],parent_sha256=parent['patch_sha256'],cases=records,pack_launches=events['pack'],
                      compared_final_bf16=sum(i['output_words'] for i in records),
                      compared_map_entries=sum(i['slots'] for i in records),
                      method='Real grouped host selection, stable route sort, projections and mapped finalizer adapters with CPU tensors. Numeric kernels modeled here; kernel_check independently executes the Triton pack.'),indent=2))
