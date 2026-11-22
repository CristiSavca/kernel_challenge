"""Render the source created from each target model's actual selected startup tuning."""
import ast,hashlib,json,logging,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
vendor,stage,target=map(Path,sys.argv[1:])
driver=here.parent/'v169_grouped_expert_tile_cohorts/vendor_codegen.py'
exec(compile(driver.read_text().split('sp=importlib.util.spec_from_file_location',1)[0],str(driver),'exec'))
src=stage/'python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py'
func=next(n for n in ast.parse(src.read_bytes()).body if getattr(n,'name','')=='_prepare_decode_tuning')
binding=dict(json=json,logging=logging,subprocess=subprocess,_GROUPED_COMPUTE='native')
exec(compile(ast.Module(body=[func],type_ignores=[]),str(src),'exec'),binding)
compiled=json.loads((here/'assembly.json').read_bytes())['variants'];cases={}
def codegen(k):
    cu=k.code+'\nauto ptr = reinterpret_cast<void*>(&'+k.kernel_expr+');\n'
    digest=hashlib.sha256(cu.encode()).hexdigest()
    entry=compiled[tag]
    assert digest==entry['source_sha256']
    cases[tag]=dict(source_sha256=digest,matched_cubin_sha256=entry['cubin_sha256'])
OfflineRuntime.prepare=codegen
def prepare(meta,compute,tuning):
    assert compute=='native'
    HummingKernel(shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,**json.loads(tuning))
with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(multi_processor_count=132)):
    for ph,n in [('w13',8192),('w2',4096)]:
        obj=LayerConfig(shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16)
        for lo,hi,config in scope['Sm90Heuristics'].get_configs(obj,gemm_type=GemmType.GROUPED_CONTIGUOUS):
            if lo>=256:continue
            m=config['block_shape'][0];tag=f"{ph}_{m}_"+('native' if m==8 else 'wide' if ph=='w13' else 'wide3')
            binding['_prepare_decode_tuning'](prepare,'meta',config,ph,True)
assert len(cases)==6
print(json.dumps(dict(adapter_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),selected=cases,
    scope='Actual submitted selection fed to pinned Humming CUDA generator. Source hashes match the offline-compiled cubins.'),indent=2))
