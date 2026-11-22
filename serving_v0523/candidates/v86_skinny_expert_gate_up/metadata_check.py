"""Evaluate the patch guard using the pinned Humming layer metadata and repack shape."""

import ast
import dataclasses
import json
import math
import pathlib
import sys
from types import SimpleNamespace
from typing import ClassVar


class Shape:
    is_cuda = True
    device = 'cuda:0'

    def __init__(self, shape, dtype, device=None):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def size(self, index):
        return self.shape[index]

    def is_contiguous(self):
        return True


def selected(path, names, context):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if getattr(n, 'name', '') in names]
    exec(compile(tree, str(path), 'exec'), context)


def main(vendor, checkout):
    torch = SimpleNamespace(
        Tensor=Shape, dtype=str, int32='torch.int32', float8_e4m3fn='torch.float8_e4m3fn',
        float8_e8m0fnu='torch.float8_e8m0fnu', float8_e5m2='torch.float8_e5m2', float16='torch.float16',
        bfloat16='torch.bfloat16', float32='torch.float32', empty=Shape,
        cuda=SimpleNamespace(get_device_capability=lambda: (9, 0)),
    )
    import re
    ctx = dict(torch=torch, dataclasses=dataclasses, re=re)
    dtype_tree = ast.parse((vendor/'dtypes.py').read_text())
    dtype_tree.body = [n for n in dtype_tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(dtype_tree, 'dtypes.py', 'exec'), ctx)
    d = SimpleNamespace(**{k:v for k,v in ctx.items() if not k.startswith('_')})
    ctx.update(dtypes=d, math=math, ClassVar=ClassVar)
    exec((vendor/'config/enum.py').read_text(),ctx)
    ctx['BaseHummingConfig'] = type('Base',(),{'to_str': lambda self: 'shape-emulation'})
    selected(vendor/'config/config.py', {'LayerConfig'}, ctx)
    selected(vendor/'ops/weight.py', {'repack_weight'}, ctx)
    ctx.update(FakeTensor=Shape)
    n,k,experts = 8192,4096,32
    meta = ctx['LayerConfig'](shape_n=n,shape_k=k,num_experts=experts,a_dtype=d.bfloat16,
                              b_dtype=d.float4e2m1,bs_dtype=d.float8e4m3,c_dtype=d.bfloat16,
                              weight_scale_group_size=16,weight_scale_group_size_n=1)
    weight = ctx['repack_weight'](Shape((experts,n,k//8),torch.int32),4,16,True,
                                  use_wgmma=True,use_packed_k_layout=meta.use_packed_k_layout)
    # transform_humming_weight_scale transposes the original N×groups, permutes
    # 64 columns in contiguous groups, and restores its transposed shape.
    scales = Shape((experts,k//16,n),torch.float8_e4m3fn)
    p=checkout/'python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py'
    assigns=[v for v in ast.walk(ast.parse(p.read_text())) if isinstance(v,ast.Assign)]
    prepared = next(v for v in assigns if any(isinstance(t,ast.Attribute) and t.attr=='humming_skinny_gate_up'
                                               for t in v.targets)).value
    use_skinny = next(v for v in assigns if any(isinstance(t,ast.Name) and t.id=='skinny_gate_up'
                                                 for t in v.targets)).value
    flag=SimpleNamespace(get=lambda:True)
    ctx.update(n=n,k=k,experts=experts,meta=meta,prepare_grouped_decode=True,tensors={'weight':weight,'weight_scale':scales},
               envs=SimpleNamespace(SGLANG_ENABLE_MUSE_SPARK_HUMMING_SKINNY_GATE_UP=flag),
               _skinny_gate_up_module=lambda:True)
    enabled=eval(compile(ast.Expression(prepared),str(p),'eval'),ctx)
    assert enabled
    eligible = {}
    for rows in [1,8,32,33,128,256]:
        ctx.update(slots=8*rows,layer=SimpleNamespace(humming_skinny_gate_up=enabled,global_num_experts=256))
        eligible[rows]=eval(compile(ast.Expression(use_skinny),str(p),'eval'),ctx)
    assert list(eligible.values())==[True,True,True,False,False,False]
    rejects = 0
    for where,key,value in [(ctx,'prepare_grouped_decode',False),(ctx,'_skinny_gate_up_module',lambda:None),
                            (ctx['tensors'],'weight',Shape((experts,1,n),torch.int32)),
                            (ctx['tensors'],'weight_scale',Shape((experts,n,k//16),torch.float8_e4m3fn))]:
        old=where[key]; where[key]=value
        assert not eval(compile(ast.Expression(prepared),str(p),'eval'),ctx)
        where[key]=old
        rejects += 1
    print(json.dumps(dict(mma=meta.mma_type.value,packed_k=meta.use_packed_k_layout,
                          fused_e8m0=meta.use_fused_e8m0_scale,scales_on_output=meta.should_apply_bs_on_c,
                          weight_shape=weight.shape,scale_shape=scales.shape,guard=enabled,
                          eligible_rows=eligible,fallback_cases=rejects),indent=2))


if __name__ == '__main__':
    main(*map(pathlib.Path,sys.argv[1:]))
