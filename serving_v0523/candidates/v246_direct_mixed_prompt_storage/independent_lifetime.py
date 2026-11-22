"""Delayed-read simulation of the real pinned cache writer, shared FP8 views and guard."""
import argparse
import ast
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
ROOT='python/sglang/'
FILES={key:ROOT+path for key,path in {
 'direct':'srt/layers/attention/direct_prompt_kv.py','paged':'kernels/ops/attention/paged_verify_attention.py',
 'pack':'kernels/ops/attention/batched_prompt_kv.py','backend':'srt/layers/attention/flashattention_backend.py',
 'mem':'srt/mem_cache/memory_pool.py','swa':'srt/mem_cache/swa_memory_pool.py','op':'kernels/ops/kvcache/kvcache.py',
 'jit':'kernels/jit/csrc/elementwise/kvcache.cuh','util':'kernels/jit/include/sgl_kernel/utils.cuh'}.items()}
TRITON=NS(cdiv=lambda n,b:-(-n//b))
LIMIT=32*1024**2

def fun(source,name,env):
    fn=next(n for n in ast.parse(source).body if getattr(n,'name','')==name)
    tree=ast.Module([ast.parse('from __future__ import annotations').body[0],fn],[])
    exec(compile(tree,'<production-source>','exec'),env)
    return env[name]

def bind(stage,candidate):
    m=json.loads((candidate/'validation.json').read_bytes())
    full=(candidate/'patch.diff').read_bytes()
    assert sha256(full).hexdigest()==m['patch_sha256']
    with tempfile.TemporaryDirectory() as tmp:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(tmp)/'index')}
        def git(*args,data=None):
            return subprocess.check_output(['git','-C',str(stage),*args],input=data,env=env)
        git('read-tree',m['base']);git('apply','--cached','-',data=full)
        assert git('write-tree').decode().strip()==m['tree']
        src={k:git('show',':'+p).decode() for k,p in FILES.items()}
    assert all(s==(stage/FILES[k]).read_text() for k,s in src.items())
    return m,src

def audit(s):
    assert 'pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool' in s['swa'] and 'pool.set_kv_buffer(' in s['swa']
    assert 'if cache_k.dtype != self.dtype:' in s['mem'] and 'return store_cache(' in s['mem']
    assert 'cuda_files=["elementwise/kvcache.cuh"]' in s['op'] and 'module.store_cache(' in s['op']
    assert all(s['jit'].index(x)<s['jit'].index('PDLTriggerSecondary') for x in ('const auto k = warp::load_bytes','const auto v = warp::load_bytes'))
    assert 'LaunchKernel(num_blocks' in s['jit'] and 'TVMFFIEnvGetStream(device.device_type, device.device_id)' in s['util']
    f=s['backend'].split('    def forward_extend(',1)[1]
    labels=['write_k.copy_(k)','self.token_to_kv_pool.set_kv_buffer(layer, write, write_k, write_v',
            'tail_result = paged_verify_attention','tail_result = flash_attn_with_kvcache',
            'write_k, write_v = write_k[:','dense_k, dense_v = pack_prompt_rows']
    positions=[f.index(x) for x in labels];assert positions==sorted(positions)
    assert 'charged = prompt_storage.nbytes if prompt_storage is not None' in f
    return labels

class Stream:
    def __init__(self): self.jobs=[]
    def append(self,fn):
        done=False
        def event():
            nonlocal done
            if not done:
                fn()
                done=True
        self.jobs.append(event)
    def wait_stream(self,other):
        prior=tuple(other.jobs)
        self.append(lambda:[e() for e in prior])

class Device:
    def __init__(self):
        self.main,self.alt=Stream(),Stream()
        self.active=self.main
    def current_stream(self):return self.active
    @contextmanager
    def stream(self,other):
        saved,self.active=self.active,other
        try:yield
        finally:self.active=saved
    def finish(self):
        for e in self.main.jobs:e()
        for e in self.alt.jobs:e()

class Bank:
    def __init__(self,tensor,device):self.t,self.dev=tensor,device
    def __setitem__(self,idx,v):self.dev.active.append(lambda:self.t.index_copy_(0,idx,v))

def threshold(allocate,work):
    rows=[]
    for nt,h,d,g,sms in [(1,2,64,8,132),(15,2,64,8,132),(31,2,64,8,132),(32,2,64,8,132),(40,2,64,8,132),(31,2,64,8,0),(3,1,32,4,96)]:
        q=4096;n=q+8*nt;layer=NS(tp_q_head_num=h*g)
        backend=NS(_hopper_verify_sms=sms,kv_cache_dtype=torch.float8_e4m3fn)
        k=torch.empty(n,h,d,dtype=torch.bfloat16,device='meta');tail=(q,q,8,None,None)
        scratch=work(nt,h,8*g,d,-1,sms)[1] if sms else 0
        maxrows=(LIMIT-scratch)//(2*h*d)
        for delta in (-1,0,1):
            prefix=maxrows-max(q+256,n)+delta
            layout=(None,torch.empty(1),prefix+q+256,prefix+q)
            result=allocate(backend,layout,tail,k,layer,nt+1)
            assert (result is not None)==(delta<=0)
            if result is not None:assert result.shape==(2,maxrows+delta,h,d) and result.nbytes+scratch<=LIMIT
            rows.append(dict(tail=nt,sms=sms,h=h,d=d,delta=delta,accepted=result is not None,scratch=scratch,bank_bytes=(maxrows+delta)*2*h*d))
        assert allocate(backend,(None,torch.empty(2),1,1),tail,k,layer,nt+2) is None
        assert allocate(backend,layout,None,k,layer,nt+1) is None
    return rows

def replay(src,allocate,cut,prefix,nt,mode,omit_wait=False):
    torch.manual_seed(prefix+nt)
    h,d=2,64;n=cut+nt*8
    tail=(cut,cut,8,None,None)
    layout=(torch.tensor([0,prefix+cut+256]),torch.tensor([prefix+cut]),prefix+cut+256,prefix+cut)
    # Disjoint strided BF16 operands expose all 65,536 encodings over K and V.
    raw=torch.arange(2*n*h*d+8,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    bf=[raw[i:i+2*n*h*d:2].view(n,h,d) for i in (0,1)]
    ref=[x.to(torch.float8_e4m3fn,memory_format=torch.contiguous_format).view(torch.uint8).clone() for x in bf]
    buf=allocate(NS(_hopper_verify_sms=132,kv_cache_dtype=torch.float8_e4m3fn),layout,tail,bf[0],NS(tp_q_head_num=16),nt+1)
    buf.view(torch.uint8).fill_(173)
    a,b=buf[:,prefix:prefix+n].unbind()
    a.copy_(bf[0]);b.copy_(bf[1]);w=[a.view(torch.uint8),b.view(torch.uint8)]
    assert all(t.is_contiguous() and torch.equal(t,r) for t,r in zip(w,ref))
    size=prefix+n+9
    physical=[torch.full((size,h,d),72+i,dtype=torch.uint8) for i in range(2)]
    slots=torch.randperm(size-1)+1;old,new=slots[:prefix],slots[prefix:prefix+n]
    prior=[torch.randint(0,256,(prefix,h,d),dtype=torch.uint8) for _ in range(2)]
    for tensor,history in zip(physical,prior):tensor.index_copy_(0,old,history)
    expected=[tensor.clone() for tensor in physical]
    for tensor,value in zip(expected,ref):tensor.index_copy_(0,new,value)
    dev=Device()
    def native(k,v,kc,vc,indices,**kw):
        assert k.data_ptr()==w[0].data_ptr() and v.data_ptr()==w[1].data_ptr() and kw['row_bytes']==128
        dev.main.append(lambda:(kc.index_copy_(0,indices,k),vc.index_copy_(0,indices,v)))
    mem=src['mem']
    if omit_wait:mem=mem.replace('current_stream.wait_stream(alt_stream)','pass',1)
    writer=fun(mem,'_set_kv_buffer_impl',dict(torch=torch,_is_cuda=True,_is_hip=False,_is_xpu=False,
        _is_cpu=False,can_use_store_cache=lambda *_:mode=='jit',store_cache=native))
    runner=ModuleType('sglang.srt.model_executor.runner');runner.get_is_capture_mode=lambda:mode=='alt'
    with patch.dict(sys.modules,{runner.__name__:runner}):
        banks=physical if mode=='jit' else [Bank(t,dev) for t in physical]
        writer(*w,*banks,new,128,torch.uint8,dev,size,dev.alt)
    seen=[]
    dev.main.append(lambda:seen.extend([t[new[cut:]].clone() for t in physical]))
    class Fill:
        def __getitem__(self,grid):
            def launch(*args,**kw):
                assert kw['INLINE_NEW'] and kw['GUARD']==256 and kw['ROW']==128
                assert all(args[2+i].data_ptr()==w[i].data_ptr() for i in (0,1))
                def gather():
                    for i in (0,1):
                        output=args[9+i]
                        assert output.data_ptr()==buf[i].data_ptr()
                        output[:prefix].copy_(physical[i][old])
                        output[prefix+cut:prefix+cut+256].zero_()
                dev.main.append(gather)
            return launch
    pack=fun(src['pack'],'pack_prompt_rows',dict(torch=torch,triton=TRITON,_pack_prompt_rows=Fill()))
    packed=pack(*physical,a[:cut],b[:cut],torch.empty(1,2,dtype=torch.int32),layout[1],
        torch.tensor([0,cut]),layout,storage=buf)
    dev.finish()
    for t,history,rr in zip(packed,prior,ref):
        data=t.view(torch.uint8)
        assert torch.equal(data[:prefix],history) and torch.equal(data[prefix:prefix+cut],rr[:cut])
        assert not data[prefix+cut:].any()
    good=[torch.equal(x,y) for x,y in zip(physical,expected)]
    reads=[torch.equal(x,y[cut:]) for x,y in zip(seen,ref)]
    assert good==reads==([True,False] if omit_wait else [True,True])
    if nt*8>256:
        assert all(torch.equal(buf[i,prefix+cut+256:prefix+n].view(torch.uint8),ref[i][cut+256:]) for i in (0,1))
    return dict(cut=cut,prefix=prefix,tails=nt,mode=mode,omitted_alt_join=omit_wait,
        allocation_bytes=buf.nbytes,converted_bytes=sum(t.nbytes for t in ref),tail_guard_overlap_rows=min(nt*8,256),
        physical_bytes_checked=sum(t.nbytes for t in physical),physical_KV_exact=good,tail_observed_exact=reads,
        prefix_current_guard_exact=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);p.add_argument('--candidate',type=Path,default=HERE);args=p.parse_args()
    m,src=bind(args.stage,args.candidate);steps=audit(src)
    name='sglang.kernels.ops.attention.paged_verify_attention';mod=ModuleType(name)
    work=mod.verify_partition_workspace=fun(src['paged'],'verify_partition_workspace',dict(triton=TRITON))
    alloc=fun(src['direct'],'allocate_mixed_prompt_storage',dict(torch=torch));torch.set_num_threads(1)
    with patch.dict(sys.modules,{name:mod}):
        edges=threshold(alloc,work)
        cases=[replay(src,alloc,c,p,t,mode) for c,p,t in ((4096,0,1),(4097,1037,31),(4096,333,40)) for mode in ('jit','alt','naive')]
        bad=replay(src,alloc,4097,1037,31,'alt',True)
    print(json.dumps(dict(tree=m['tree'],patch_sha256=m['patch_sha256'],source_sha256={FILES[k]:sha256(v.encode()).hexdigest() for k,v in src.items()},
        forward_order=steps,cases=cases,negative_control=bad,thresholds=edges,physical_H200=False,
        boundary='Real allocator, writer dispatch and host pack. CPU tensor copies deferred until their CUDA stream dependencies allow them to run. CUDA store and Triton refill use instrumented boundaries.'),indent=2))
if __name__=='__main__':main()
