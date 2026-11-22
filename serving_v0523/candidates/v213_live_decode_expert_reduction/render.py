"""Render the exact native startup variants and exercise constructor fallback."""
import contextlib, io, json, runpy, sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch

here=Path(__file__).resolve().parent
root=Path(sys.argv[1]).resolve()
model=root/'sglang/python/sglang/srt/layers/muse_spark'
for name in ('decode_mma_pair','residual_prompt_mma','live_decode_reduce'):
    full='sglang.srt.layers.muse_spark.'+name
    mod=ModuleType(full);mod.__file__=str(model/(name+'.py'))
    exec(compile(Path(mod.__file__).read_bytes(),mod.__file__,'exec'),mod.__dict__)
    sys.modules[full]=mod
with contextlib.redirect_stdout(io.StringIO()):
    d=runpy.run_path(str(here.parent/'v197_grouped_prompt_gate_up/render.py'))
vendor=d['vendor']
native=vendor['objects'][vendor['metas']['w2']]
configs=vendor['scope']['Sm90Heuristics'].get_configs(layer_config=native,gemm_type=vendor['GemmType'].GROUPED_CONTIGUOUS)
small_w2=[json.dumps(t|dict(use_stream_k=False,use_tma_c=False)) for a,b,t in configs if a<256]
runtime=d['runtime']
runtime.check_context=lambda self:None
cg=ModuleType('humming.jit.compiler');cg.NVCCCompiler=type('Compiler',(),{});sys.modules[cg.__name__]=cg
w2=sys.modules['sglang.srt.layers.muse_spark.decode_mma_pair']._prepare_down
denied=set();history=[];trial={}
def emit(k):
    name=type(k).__name__;history.append(name)
    if name in denied:raise ValueError('offline variant refusal')
    w=k.block_shape[0];k.func=w;k.kernel_id=w
    tier='live' if k.live_reduction else 'base' if k.both_pairs else 'ordinary'
    ph='w13' if k.shape_n==8192 else 'w2'
    directory=root/tier;directory.mkdir(exist_ok=True)
    (directory/f'{ph}_{w}.cu').write_text(k.code+'\nauto ptr = reinterpret_cast<void*>(&'+k.kernel_expr+');\n')
with (patch.object(runtime,'prepare',emit),patch.object(torch.cuda,'get_device_capability',return_value=(9,0)),
      patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(multi_processor_count=132))):
    for ph,fn,tunings in [('w13',d['g']['_compile'],d['small']),('w2',w2,small_w2)]:
        meta=d['vendor']['metas'][ph];compute=d['compute']
        for t in tunings:
            result=[];denied.clear()
            for ctx in range(100,104):
                history.clear();handle=fn(ctx,meta,compute,t)
                result.append([list(history),handle is not None])
                if handle is not None:denied.add(history[-1])
            assert [x[1] for x in result]==[True,True,True,False]
            trial[ph+'_'+str(json.loads(t)['block_shape'][0])]=result
print(json.dumps(trial,indent=2))
