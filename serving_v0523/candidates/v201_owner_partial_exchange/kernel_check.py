"""Run the frozen rank-ordered BF16 owner-reduction kernel in Triton or compile/disassemble it for SM90."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
import triton.language as tl


def aot(kernel):
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    records=[]
    assembler=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
    disasm=Path.home()/'nvcc126/toolchain/usr/local/cuda-12.9/bin/cuobjdump'
    with tempfile.TemporaryDirectory() as wd:
        tmp=Path(wd)
        for size in (2,3,4,5,7,8,13,16):
            for rank in range(size):
                for rowtype in (('i32','i64') if size==8 and rank in (0,3,7) else ('i32',)):
                    cs=dict(SIZE=size,RANK=rank,BLOCK=256)
                    sig=dict(Local='*bf16',Remote='*bf16',Out='*bf16',count='i32',stride=rowtype)
                    sig.update({k:'constexpr' for k in cs})
                    attrs={(i,):[['tt.divisibility',16]] for i in (0,1,2)}
                    compiled=triton.compile(ASTSource(kernel,sig,constexprs=cs,attrs=attrs),
                        target=GPUTarget('cuda',90,32),options=dict(num_warps=4,num_stages=3))
                    pt=tmp/'probe.ptx';pt.write_text(compiled.asm['ptx']);cub=tmp/'reg.cubin'
                    proc=subprocess.run([assembler,'-arch=sm_90a','-v',pt,'-o',cub],capture_output=True,text=True,check=True)
                    asm=compiled.asm['ptx']
                    fp32_adds=len(re.findall(r'\badd(?:\.rn)?\.f32\b',asm))
                    bf16_adds=len(re.findall(r'\badd(?:\.rn)?\.bf16(?:x2)?\b',asm))
                    rounds_bf16='cvt.rn.bf16' in asm
                    two_source_fold = size==2 and bf16_adds>0 and fp32_adds==0
                    assert '.ftz.' not in asm
                    assert two_source_fold or (rounds_bf16 and fp32_adds>0 and bf16_adds==0)
                    sass=subprocess.check_output([disasm,'-sass',cub],text=True)
                    regs=int(re.search(r'Used (\d+) registers',proc.stderr)[1])
                    spills=list(map(int,re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',proc.stderr).groups()))
                    records.append(dict(size=size,rank=rank,stride_type=rowtype,regs=regs,spill_bytes=spills,
                        shared=compiled.metadata.shared,cubin_bytes=len(compiled.asm['cubin']),
                        cubin_sha256=hashlib.sha256(compiled.asm['cubin']).hexdigest(),
                        fp32_adds=fp32_adds,bf16_adds=bf16_adds,ptx_rne_bf16_cast=rounds_bf16,
                        two_source_bf16_add=two_source_fold,
                        sass_fadds=len(re.findall(r'\bFADD\b',sass)),sass_round_bf16='BF16' in sass,
                        register_only_ctas_per_h200_sm=min(16,65536//(128*((regs+7)//8*8)))))
    return dict(cases=len(records),target='sm90',num_warps=4,block=256,records=records)


def interpret(kernel):
    import numpy as np
    from interpret_decode_kernels import buf
    def decode(words):return (np.asarray(words,dtype='u4')<<16).view('f4')
    def rne(x):
        b=np.asarray(x,dtype='f4').view('u4')
        return ((b+np.uint32(32767)+((b>>16)&1))>>16).astype('u2')
    def rtz(x):return (np.asarray(x,dtype='f4').view('u4')>>16).astype('u2')
    rng=np.random.default_rng(201)
    ct=dict(calls=0,zero_count=0,numeric_values=0,guard_words=0,src_words_preserved=0,
            cast_sensitive_values=0,rank_order_sensitive_values=0,repeated_bf16_rounding_disagreements=0,
            exact_interpreter_rne=0,interpreter_trunc_only=0,other=0,fp32_diagnostic_values=0,
            fp32_diagnostic_guard_words=0,software_rne_diagnostic_matches=0,negative_zero_outputs=0)
    records=[]
    for size in (2,3,4,5,7,8,13,16):
        for rank in range(size):
            for count in (0,1,17,255,256,257,773,4099):
                pre,post=1+rank%5,13
                stride=max(count,1)+[0,7,31][(rank+count)%3]
                sent=np.uint16(0x564d)
                local=np.full(pre+max(count,1)+post,sent,'u2')
                remote=np.full(pre+(size-1)*stride+post,sent,'u2')
                result=np.full(pre+max(count,1)+post,sent,'u2')
                inputs=rne(rng.normal(0,8,(size,count)).astype('f4'))
                if count:
                    # Alternating exact integers, cancellation of large powers of two and tie-to-even on both parity bits.
                    lanes=np.arange(count)
                    for j in range(size):
                        for key,vals in enumerate([[2**28,1.,-2**28], [2**28,-2**28,1.], [256.,1.,-256.],
                                [1.,2**-8], [1+2**-7,2**-8], [-1.,-2**-8], [-0.]*size, [2**-100,-2**-100,2**-110]]):
                            inputs[j,lanes%17==key]=rne([vals[j] if j<len(vals) else 0.])[0]
                local[pre:pre+count]=inputs[rank]
                for j in range(size):
                    if j!=rank:
                        off=pre+(j-(j>rank))*stride
                        remote[off:off+count]=inputs[j]
                original_local=local.copy();original_remote=remote.copy()
                expected=decode(inputs[0]).copy();rev=decode(inputs[-1]).copy();bf=expected.copy()
                for j in range(1,size):
                    expected=np.add(expected,decode(inputs[j]),dtype='f4')
                    rev=np.add(rev,decode(inputs[-j-1]),dtype='f4')
                    bf=decode(rne(np.add(bf,decode(inputs[j]),dtype='f4')))
                truth=rne(expected);trunc=rtz(expected)
                kernel[(max(1,triton.cdiv(count,256)),)](buf(local[pre:],tl.bfloat16),buf(remote[pre:],tl.bfloat16),
                    buf(result[pre:],tl.bfloat16),count,stride,SIZE=size,RANK=rank,BLOCK=256)
                # Same JIT with an FP32 destination exposes its accumulator before the implicit final cast.
                diagnostic=np.full(pre+max(count,1)+post,999999.5,'f4')
                kernel[(max(1,triton.cdiv(count,256)),)](buf(local[pre:],tl.bfloat16),buf(remote[pre:],tl.bfloat16),
                    buf(diagnostic[pre:],tl.float32),count,stride,SIZE=size,RANK=rank,BLOCK=256)
                np.testing.assert_array_equal(diagnostic[pre:pre+count].view('u4'),expected.view('u4'))
                np.testing.assert_array_equal(rne(diagnostic[pre:pre+count]),truth)
                assert np.all(diagnostic[:pre]==999999.5) and np.all(diagnostic[pre+count:]==999999.5)
                ct['fp32_diagnostic_values']+=count;ct['fp32_diagnostic_guard_words']+=diagnostic.size-count
                ct['software_rne_diagnostic_matches']+=count
                actual=result[pre:pre+count]
                negative_zeros=np.arange(count)%17==6
                assert np.all(truth[negative_zeros]==0x8000) and np.all(actual[negative_zeros]==0x8000)
                ct['negative_zero_outputs']+=int(negative_zeros.sum())
                good=actual==truth;trunc_only=(~good)&(actual==trunc)
                other=(~good)&(~trunc_only)
                if other.any():raise AssertionError((size,rank,count,np.flatnonzero(other)[:12].tolist()))
                assert np.all(result[:pre]==sent) and np.all(result[pre+count:]==sent)
                assert np.array_equal(local,original_local) and np.array_equal(remote,original_remote)
                ct['calls']+=1;ct['zero_count']+=int(count==0);ct['numeric_values']+=count
                ct['guard_words']+=int(result.size-count);ct['src_words_preserved']+=int(local.size+remote.size)
                ct['cast_sensitive_values']+=int((truth!=trunc).sum())
                ct['rank_order_sensitive_values']+=int((truth!=rne(rev)).sum())
                ct['repeated_bf16_rounding_disagreements']+=int((truth!=rne(bf)).sum())
                ct['exact_interpreter_rne']+=int(good.sum());ct['interpreter_trunc_only']+=int(trunc_only.sum())
    return ct
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['aot','interpret']);parser.add_argument('source',type=Path)
    parser.add_argument('tests',type=Path);a=parser.parse_args();sys.path.insert(0,str(a.tests.resolve()))
    from compile_decode_sm90 import load_jit_module
    with tempfile.TemporaryDirectory() as tmp:
        fn=load_jit_module(a.source,Path(tmp))._sum_owned_rows
        print(json.dumps(dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),triton=triton.__version__,
            physical_h200=False,**(aot if a.mode=='aot' else interpret)(fn)),indent=2))
