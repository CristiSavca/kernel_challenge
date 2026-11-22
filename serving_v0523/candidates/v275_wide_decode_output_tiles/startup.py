"""Execute the submitted startup selection against the pinned tuner at every dispatch boundary."""
import ast,hashlib,json,logging,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
vendor,stage,target=map(Path,sys.argv[1:])
driver=here.parent/'v169_grouped_expert_tile_cohorts/vendor_codegen.py'
exec(compile(driver.read_text().split('sp=importlib.util.spec_from_file_location',1)[0],str(driver),'exec'))
source=stage/'python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py'
func=next(n for n in ast.parse(source.read_bytes()).body if getattr(n,'name','')=='_prepare_decode_tuning')
scope2=dict(json=json,logging=logging,subprocess=subprocess,_GROUPED_COMPUTE='native')
exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),'exec'),scope2)
select=scope2['_prepare_decode_tuning'];calls=[];rows=[]
def prep(meta,compute,cfg):
    assert meta=='actual' and compute=='native'
    obj=json.loads(cfg);calls.append(obj)
    if refuse and obj['warp_shape'][1]==64:raise RuntimeError('test refusal')
with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)), patch.object(
        torch.cuda,'get_device_properties',return_value=SimpleNamespace(multi_processor_count=132)):
    for name,n in [('w13',8192),('w2',4096)]:
        op=LayerConfig(shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16)
        configs=scope['Sm90Heuristics'].get_configs(op,gemm_type=GemmType.GROUPED_CONTIGUOUS)
        for lo,hi,base in configs:
            if lo>=256:continue
            m=base['block_shape'][0]
            for allow in (False,True):
              for refuse in (False,True):
                calls.clear()
                cfg=json.loads(select(prep,'actual',base,name,allow))
                assert cfg==calls[-1]
                enabled=allow and m in (16,24)
                assert len(calls)==1+int(enabled and refuse)
                chosen=enabled and not refuse
                native=json.loads(json.dumps(base|dict(use_tma_c=False,use_stream_k=False)))
                if chosen:
                    expected='wide' if name=='w13' else 'wide3'
                    assert cfg==native|dict(block_shape=[m,2*native['block_shape'][1],native['block_shape'][2]],
                        warp_shape=[m,64,64],num_stages=4 if name=='w13' else 3)
                else:assert cfg==native
                rows.append(dict(projection=name,min_exclusive=lo,max_inclusive=hi,allowed=allow,refused=refuse,
                    attempts=len(calls),chosen=cfg['block_shape'],stages=cfg['num_stages']))
print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tests=rows),indent=2))
