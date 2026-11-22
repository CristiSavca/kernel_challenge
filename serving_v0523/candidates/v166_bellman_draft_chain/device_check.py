"""Compile or interpret the two exact production functions without allocating a GPU."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl


def load(path, directory):
    text=path.read_text();lines=text.splitlines(True);entries={x.name:x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
    names=['_prefix_value_plan','_selector_walk_kernel'] if '_prefix_value_plan' in entries else ['_selector_walk_kernel']
    file=directory/(path.stem+'_jit.py')
    file.write_text('import triton\nimport triton.language as tl\n' + '\n'.join(
        ''.join(lines[entries[n].decorator_list[0].lineno-1:entries[n].end_lineno]) for n in names))
    spec=importlib.util.spec_from_file_location(file.stem,file);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod._selector_walk_kernel


def aot(kernel, folder):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource
    answer=[]
    for d,k in ((1,2),(2,2),(3,4),(5,8),(7,16),(8,16),(6,8),(7,32)):
        for enable in (False,True):
            if enable and (d<2 or k>16):continue
            c=dict(slots=d,top_k=k,PREFIX_VALUE=enable);sig=dict(zip(kernel.arg_names[:7],['*fp32','*i64','*fp32','*fp32','*i1','*i64','*fp32']))
            sig.update({n:'constexpr' for n in c});res=triton.compile(ASTSource(kernel,sig,constexprs=c),target=GPUTarget('cuda',90,32),options={'num_warps':1})
            ptx=folder/'walk.ptx';ptx.write_text(res.asm['ptx']);out=folder/'walk.cubin'
            proc=subprocess.run([get_ptxas(90).path,'-arch=sm_90a','-v',str(ptx),'-o',str(out)],capture_output=True,text=True,check=True)
            regs=int(re.search(r'Used (\d+) registers',proc.stderr).group(1))
            spills=re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',proc.stderr)
            assert spills
            answer.append(dict(slots=d,top_k=k,enable=enable,registers=regs,spill_store_bytes=int(spills[1]),spill_load_bytes=int(spills[2]),cubin_sha256=hashlib.sha256(res.asm['cubin']).hexdigest(),binary_bytes=len(res.asm['cubin'])))
    return answer


def oracle(scores):
    d,k=scores.shape[:2];val=np.zeros(k);path=[() for _ in range(k)]
    for t in reversed(range(d)):
        v=np.zeros(k);plans=[]
        for prev in range(k):
            s=scores[t,prev].astype(np.float64)
            if t==0 and prev>0:plans.append(());continue
            if (np.isnan(s)|np.isposinf(s)).any() or np.isneginf(s).all():return None
            mass=np.exp(s-np.max(s));utility=mass*(1+val);mx=utility.max()
            ix=max(np.flatnonzero(utility==mx),key=lambda i:(float(s[i]),-int(i)))
            v[prev]=mx/mass.sum();plans.append((ix,)+path[ix])
        val,path=v,plans
    return path[0]


def interpret(candidate,native):
    from interpret_decode_kernels import buf
    rng=np.random.default_rng(166);records=[];tot=changed=samples=fallback=0
    for d,k in ((2,2),(3,4),(5,8),(7,16),(8,16)):
        for case in range(12):
            b=6;s=rng.normal(0,1.35,(b,d,k,k)).astype('f4')
            ids=rng.integers(2**34,2**40,(b,d,k),dtype='int64');u=rng.uniform(.001,.999,(b,d)).astype('f4')
            temp=np.asarray([1,1,.2,2,1,.01],dtype='f4');gm=np.array([1,1,0,0,1,0],dtype=bool)
            if case==1:s[0]=0
            if case==2:s[1,1,0,0]=np.nan
            if case==3:s[1,-1,-1]=np.inf
            if case==4:s[0,-1,0]=-np.inf
            if case==5:s[0,0,1:]=np.nan
            if case==6:s[0,:,0,-1]=-np.inf
            if case==7:ids[:,:,:2]=ids[:,:,:1]
            if case==8:s[0]=np.round(s[0],1)
            def launch(kernel,flag=None):
                tok=np.empty((b,d),dtype='i8');q=np.empty_like(ids,dtype='f4')
                args=[buf(x,ty) for x,ty in zip((s,ids,u,temp,gm,tok,q),(tl.float32,tl.int64,tl.float32,tl.float32,tl.int1,tl.int64,tl.float32))]
                kw={} if flag is None else dict(PREFIX_VALUE=flag)
                kernel[(b,)](*args,slots=d,top_k=k,num_warps=1,**kw);return tok,q
            base,base_q=launch(native); off,off_q=launch(candidate,False); out,q=launch(candidate,True)
            assert np.array_equal(base,off) and np.array_equal(base_q,off_q,equal_nan=True)
            for i in range(b):
                expected=oracle(s[i]) if gm[i] else None
                if expected is None:
                    assert np.array_equal(base[i],out[i]) and np.array_equal(base_q[i],q[i],equal_nan=True)
                    samples+=int(not gm[i]);fallback+=int(gm[i])
                else:
                    want=np.eye(k,dtype='f4')[np.asarray(expected)]
                    # Rounded alternatives can tie to within FP32; only compare strict cases.
                    assert np.array_equal(q[i],want),(d,k,case,i,q[i].argmax(-1),expected)
                    assert np.array_equal(out[i],ids[i,np.arange(d),expected]);changed+=int(not np.array_equal(base[i],out[i]))
                tot+=1
    return dict(request_rows=tot,stochastic_rows_bitwise_equal=samples,malformed_greedy_rows_fallback=fallback,changed_greedy_rows=changed,all_flag_off_rows_bitwise_equal=tot)


if __name__=='__main__':
    cmd,src,native= sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        folder=Path(tmp);f=load(src,folder);g=load(native,folder)
        tested=aot(f,folder) if cmd=='aot' else interpret(f,g)
    print(json.dumps(dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),triton=triton.__version__,hardware=False,mode=cmd,result=tested),indent=2))
