"""Compile or interpret the exact submitted/parent Hopper merge bodies in a Linux Triton 3.8 environment."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf

def constants(b,h,g,q,window,parts):
    return dict(B=b,H_KV=h,GQA=g,D=64,WINDOW_LEFT=window,TILE_M=triton.next_power_of_2(g*q),
                TILE_N=64 if window>=0 else 128,PARTITIONS=parts,BLOCK_B=triton.next_power_of_2(b),
                BLOCK_SPLITS=16,VS_ROW=h,VS_HEAD=1)

def compile_cases(before,after,tmp):
    ptxas=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
    methods=[]
    for b in (1,4,16,32):
        for sliding in (False,True):
            for name,mod in [('parent',before),('tiled',after)]:
                g,h,q=8,2,8
                rows=min(4,1<<(max(1,b*h*q*g//132).bit_length()-1))
                const=constants(b,h,g,q,2047 if sliding else -1,66 if sliding else 132)
                if name=='tiled': const['ROWS']=rows
                fn=mod._merge_partitions
                sig=dict(Partial='*fp32',LSE='*fp32',Lengths='*i32',CuQ='*i32',VScale='*fp32',Out='*bf16')
                aligned={(fn.arg_names.index(k),):[['tt.divisibility',16]] for k in sig}
                sig.update({k:'constexpr' for k in const})
                binary=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=aligned),target=GPUTarget('cuda',90,32),
                                      options=dict(num_warps=4,num_stages=3))
                ttgir=binary.asm['ttgir']
                layouts=dict(line.split(' = ',1) for line in ttgir.splitlines() if line.startswith('#blocked'))
                reductions=[]
                for body in re.findall(r'"tt.reduce"\((.*?)\n\s*}\) : \(tensor<(.*?)>\) ->',ttgir,re.S):
                    operation,operand=body
                    if 'xf32' not in operand:
                        continue
                    operand=re.sub(r'#blocked\d*',lambda m:layouts[m.group()],operand)
                    combiner=re.search(r'= (arith\.\w+) ',operation).group(1)
                    reductions.append(dict(combiner=combiner,operand=operand))
                ptx=binary.asm['ptx']
                math_ops=Counter(re.findall(r'\b(?:add|sub|mul|fma|ex2|div|rcp|max|shfl)\.[^\s,;]+',ptx))
                f=tmp/'merge.ptx';f.write_text(ptx)
                stat=subprocess.run([ptxas,'-v','-arch=sm_90a',str(f),'-o',str(tmp/'out.cubin')],check=True,capture_output=True,text=True).stderr
                spills=list(map(int,re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',stat).groups()))
                reg=int(re.search(r'Used (\d+) registers',stat).group(1))
                methods.append(dict(variant=name,batch=b,window=const['WINDOW_LEFT'],rows=const.get('ROWS',1),registers=reg,spills=spills,
                    reductions=reductions,math_ops=math_ops,shared=binary.metadata.shared,local_ld_st=bool(re.search(r'\b(?:ld|st)\.local',ptx)),cubin_sha256=sha256(binary.asm['cubin']).hexdigest(),
                    vectorized_ld_global=len(re.findall(r'ld\.global[^\s]*\.v[24]',ptx))))
    for old,new in zip(methods[::2],methods[1::2]):
        assert old['reductions'] == new['reductions'] and len(old['reductions']) == 3
        for opcode in ('fma.rn.f32','add.f32','ex2.approx.ftz.f32','shfl.sync.bfly.b32'):
            assert old['math_ops'][opcode] == new['math_ops'][opcode]
        assert new['spills']==[0,0] and not new['local_ld_st']
    return {'compiles':methods,'matched_reductions':['max(LSE)','sum(weight * partial)','sum(weight)'],
            'matched_math':['instructions per loop body for FP32 additions, fma, exp2 and butterfly shuffle',
                            'all three SM90 reduction operand layouts after resolving compiler aliases']}

def interpret(before,after):
    rng=np.random.default_rng(236)
    counts=Counter()
    for b,h,g,q,rows,window,parts in [(1,2,8,8,1,-1,132),(4,2,8,8,2,2047,66),(4,2,8,8,2,-1,132),
        (16,2,8,8,4,-1,132),(32,2,8,8,4,2047,66),(7,1,3,6,4,-1,41),(3,2,5,7,8,0,7),(3,2,8,8,8,2047,1)]:
        const=constants(b,h,g,q,window,parts); m=const['TILE_M']; d=64; n=parts+b-1
        for strat in ('all_live','ragged','zero_queries','zero_lengths','single_dominant'):
            if strat=='all_live':qrows=np.full(b,q,dtype='i4')
            else:qrows=rng.integers(0,q+1,size=b,dtype='i4');qrows[-1]=q
            if strat=='zero_queries':qrows.fill(0)
            cu=np.concatenate(([0],qrows.cumsum())).astype('i4')
            length=rng.integers(0,130000,size=b,dtype='i4')+qrows
            if strat=='zero_lengths':length.fill(0)
            if strat=='single_dominant':length[0]=130000;length[1:]=qrows[1:]
            lse=rng.uniform(-40,40,(n,h,m)).astype('f4');lse[rng.random(lse.shape)<.15]=-np.inf
            partial=rng.normal(size=(n,h,m,d)).astype('f4')
            scales=rng.uniform(.03,2,(b,h)).astype('f4');eps=qrows.sum()
            # Sentinel prefix/suffix reveals padded writes. The original code accepts
            # zero query rows and empty key sets without reading unowned partials.
            outputs=[]
            for mod,tile,name in [(before,1,'parent'),(after,rows,'tiled')]:
                out=np.full((max(1,int(eps)+3),h*g,d),0x7f81,dtype='u2')
                kw=dict(const)
                if name=='tiled':kw['ROWS']=tile
                scale=buf(scales,tl.float32) if strat!='ragged' else None
                mod._merge_partitions[(b,h,triton.cdiv(q*g,tile))](buf(partial,tl.float32),buf(lse,tl.float32),
                       buf(length,tl.int32),buf(cu,tl.int32),scale,buf(out,tl.bfloat16),**kw)
                assert np.all(out[int(eps):]==0x7f81)
                outputs.append(out)
            assert np.array_equal(*outputs),(b,g,q,window,strat,int((outputs[0]!=outputs[1]).sum()))
            counts['comparisons']+=1;counts['bf16_words']+=int(outputs[0].size);counts['live_words']+=int(eps*h*g*d)
    return dict(counts=counts,reference='Frozen V235 merge on identical FP32 partials/LSEs and per-request scales')

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('compile','interpret'));a.add_argument('before',type=Path);a.add_argument('after',type=Path)
    args=a.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        tmp=Path(scratch);before=load_jit_module(args.before,tmp);after=load_jit_module(args.after,tmp)
        result=compile_cases(before,after,tmp) if args.action=='compile' else interpret(before,after)
    print(json.dumps(dict(triton=triton.__version__,physical_h200=False,
        source_sha256=[sha256(p.read_bytes()).hexdigest() for p in (args.before,args.after)],**result),indent=2))
if __name__=='__main__':main()
