"""Execute the published grouped W4A16 caller with shape-only tensors and recording operators."""
import ast
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

class Tensor:
    def __init__(self,shape,bank=None):
        self.shape=(shape,) if isinstance(shape,int) else tuple(shape)
        self.bank=object() if bank is None else bank
    def numel(self):
        from math import prod
        return prod(self.shape)
    def new_empty(self,shape): return Tensor(shape)
    def __getitem__(self,s): return Tensor(s.stop,self.bank)
    def view(self,*shape): return Tensor(self.shape if shape==('uint8',) else shape,self.bank)

class Launch:
    def __init__(self,call): self.call=call
    def __getitem__(self,grid): return self.call

def main(stage):
    file=stage/'python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py'
    f=next(n for n in ast.parse(file.read_text()).body if getattr(n,'name','')=='humming_grouped_decode_outputs')
    f.body=[n for n in f.body if not isinstance(n,ast.ImportFrom)]
    torch=SimpleNamespace(uint8='uint8',int32='int32',Tensor=Tensor,empty_like=lambda t,**kwargs:Tensor(t.shape))
    cases=0
    for rows,skinny,down,norm,fused in itertools.product([1,8,32,33,128,256],[False,True],[False,True],[False,True],[False,True]):
        ops=[]; links={}
        def call(name,i,o):
            assert i.bank is not o.bank
            ops.append(name); links[name]=(i.bank,o.bank)
        def humming(layer,inputs,outputs,sublayer_name,**kwargs): call(sublayer_name,inputs,outputs)
        def custom(name):
            return lambda:SimpleNamespace(run=lambda i,o,*args:call(name,i,o))
        stage_only=Launch(lambda *a,**kw: None)
        sc=dict(torch=torch,MuseSpark1xFp4Experts=object,HummingMethod=SimpleNamespace(forward_layer=humming),
                triton=SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length(),cdiv=lambda n,d:(n+d-1)//d),
                _GROUPED_COMPUTE='grouped',_skinny_gate_up_module=custom('skinny'),
                _activated_down_module=custom('activated_down'),_expert_scaled_silu_module=custom('activation'),
                _stage_token_routes=stage_only,_stage_grouped_decode=stage_only,
                _plan_grouped_decode=stage_only,_gather_grouped_decode=stage_only,
                envs=SimpleNamespace(SGLANG_ENABLE_MUSE_SPARK_HUMMING_FUSED_STAGE=SimpleNamespace(get=lambda:fused)))
        exec(compile(ast.Module([f],type_ignores=[]),str(file),'exec'),sc)
        layer=SimpleNamespace(humming_skinny_gate_up=skinny,humming_activated_down=down,
                              intermediate_size_per_partition=4096,num_local_experts=32,global_num_experts=256,
                              humming_decode_tunings={'w13':[(2048,1)],'w2':[(2048,1)]},_expert_map=None)
        for key in ['w13_weight','w13_weight_scale','w13_global_scale','w2_weight','w2_weight_scale']:
            setattr(layer,key,Tensor((32,)))
        out,_,_=sc[f.name](layer,Tensor((rows,4096)),Tensor((rows,8)),Tensor((4096,)) if norm else None)
        if skinny and rows<=32:
            expected=['skinny','w2']
        elif down and rows>32:
            expected=['w13','activated_down']
        else:
            expected=['w13','activation','w2']
        assert ops==expected and out.bank is links[ops[-1]][1]
        assert all(links[a][1] is links[b][0] for a,b in zip(ops,ops[1:]))
        cases+=1
    print(json.dumps(dict(dispatch_cases=cases,disjoint_inputs_outputs=True,producer_consumer_aliases=True),indent=2))

if __name__=='__main__': main(Path(sys.argv[1]))
