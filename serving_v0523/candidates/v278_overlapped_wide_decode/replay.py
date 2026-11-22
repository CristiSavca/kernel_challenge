"""Check V278 stage dependencies and the high-load tuning passed to the model's actual Humming callback."""
import hashlib,json,runpy,sys
from pathlib import Path
import torch
here=Path(__file__).resolve().parent
v=json.loads((here/'validation.json').read_bytes())
common=runpy.run_path(str(here.parent/'v272_parallel_decode_route_plan/replay.py'))
Fixture=common['Fixture']
control=Fixture(v['parent_tree'],'V68')
joined=Fixture(v['tree'],'V272')
tuning=json.loads((here/'startup.json').read_bytes())
counter=dict(calls=0,wide_W13_calls=0,wide_W2_calls=0,routing_overlapped=0,native_projection_fallback=0)
observed=[]
def setup(f,high,refuse):
    defs={}
    for ph in ('w13','w2'):
        defs[ph]=[(r['max_inclusive'],json.dumps({'block_shape':r['chosen'],'num_stages':r['stages']}))
            for r in tuning['tests'] if r['allowed']==high and r['refused']==refuse and r['projection']==ph]
    f.model.experts.humming_decode_tunings=defs

for f in (control,joined):
    callback=f.method.HummingMethod.forward_layer
    def record(layer,*,_f=f,_fn=callback,**kwargs):
        if _f is joined:
            observed.append((kwargs['sublayer_name'],json.loads(kwargs['tuning_config'])))
        return _fn(layer,**kwargs)
    f.method.HummingMethod.forward_layer=record

rng=torch.Generator().manual_seed(278)
# Use the real 4096-channel hidden shape for the high-load route tables.
for rows in (8,32,128,256):
    h=4096
    router=torch.randn(h,256,generator=rng)
    x=torch.randn(rows,h,generator=rng).bfloat16()
    for mode in ('capture','eager','router_hook'):
      for refuse in (False,True):
        for f in (control,joined):setup(f,True,refuse)
        observed.clear()
        a=control.execute(x,router,3,8,mode);b=joined.execute(x,router,3,8,mode)
        assert len(observed)==2 and [pair[0] for pair in observed]==['w13','w2']
        for (ph,cfg) in observed:
            high=rows>=128 and not refuse
            expected_n=(512 if ph=='w13' else 256) if high else (256 if ph=='w13' else 128)
            assert cfg['block_shape'][1]==expected_n and cfg['num_stages']==(3 if high and ph=='w2' else 4)
            if high:counter['wide_'+ph.upper()+'_calls']+=1
        assert all(torch.equal(c,d) for c,d in zip(a[0],b[0])) and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
        counter['calls']+=1
        counter['routing_overlapped']+=mode=='capture'
        counter['native_projection_fallback']+=rows>=128 and refuse
print(json.dumps(dict(sha256=v['patch_sha256'],counts=counter,source_closure_hashes={
    n:hashlib.sha256((Path(sys.argv[1])/n).read_bytes()).hexdigest() for n in v['changed_from_parent']},
    test_scope='Native model, plan/stream dependency assertions, grouped adapter and callback config with emulated projection/plan GPU execution. Original comparison uses the V275 tree.'),indent=2))
