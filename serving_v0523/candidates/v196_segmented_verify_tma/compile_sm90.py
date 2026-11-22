"""AOT compile and disassemble V195, V196 and alternative segment planners on Hopper without a GPU."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def main():
    src,tools,dumper=map(Path,sys.argv[1:4])
    sys.path.insert(0,str(tools.resolve()))
    from compile_decode_sm90 import load_jit_module
    ptxas=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
    results=[]
    with tempfile.TemporaryDirectory() as name:
        tmp=Path(name)
        for variant,file in [('V195','prior.py'),('V196','paged_verify_attention.py'),('fixed4','fixed4.py'),
                             ('fixed8','fixed8.py'),('bounded_scan32','run32.py')]:
            path=src/file
            if not path.is_file() and variant not in ('V195','V196'):
                path=tmp/file
                patch=Path(__file__).resolve().parent/'lowerings'/file.replace('.py','.diff')
                subprocess.run(['patch','-s','-o',path,src/'paged_verify_attention.py'],
                               input=patch.read_bytes(),check=True,capture_output=True)
            kernel=load_jit_module(path,tmp)._partition_attention
            if variant in ('V195','V196'):
                cases=[(b,s,aligned,active) for b in (1,3,4,7,16,32) for s in (False,True)
                       for aligned,active in [(True,True)]]+[(16,True,False,True),(32,True,False,True),
                                                           (16,True,True,False),(32,True,True,False)]
            else:cases=[(b,True,True,True) for b in (16,32)]
            for b,scale,aligned,active in cases:
                cs=dict(PAGE_ROW_STRIDE=131088,PAGE_SIZE=1,K_PAGE=128,K_TOKEN=128,K_HEAD=64,V_PAGE=128,
                    V_TOKEN=128,V_HEAD=64,KS_ROW=0,KS_HEAD=0,B=b,H_KV=2,GQA=8,D=64,SCALE_LOG2=math.log2(math.e)/8,
                    WINDOW_LEFT=-1,MAX_Q=8,TILE_M=64,TILE_N=128,PARTITIONS=132,BLOCK_B=triton.next_power_of_2(b),
                    ORIGIN_STRIDE=1025 if active else 0,CAPACITY=3420200 if active else 0)
                sig=dict(zip(kernel.arg_names[:9],['*fp8e4nv']*3+['*i32']*3+['*fp32']*3))
                if not scale:del sig['KScale'];cs['KScale']=None
                if active:
                    sig.update(Origins='*i32',KDesc='tensordesc<fp8e4nv[128,64]>',VDesc='tensordesc<fp8e4nv[128,64]>')
                else:cs.update(Origins=None,KDesc=None,VDesc=None)
                attrs={(kernel.arg_names.index(arg),):[['tt.divisibility',16]] for arg,ty in sig.items()
                       if aligned and ty.startswith('*')}
                sig.update({k:'constexpr' for k in cs})
                binary=triton.compile(ASTSource(kernel,sig,constexprs=cs,attrs=attrs),
                    target=GPUTarget('cuda',90,32),options=dict(num_warps=4,num_stages=3))
                ptx=tmp/'kernel.ptx';ptx.write_text(binary.asm['ptx']);cub=tmp/'kernel.cubin';cub.write_bytes(binary.asm['cubin'])
                stats=subprocess.run([ptxas,'-v','-arch=sm_90a',ptx,'-o',tmp/'reassembled.cubin'],
                    capture_output=True,text=True,check=True).stderr
                sass=subprocess.check_output([dumper,'-sass',cub],text=True)
                instr=collections.Counter(re.findall(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z0-9]+)',sass))
                interesting={k:v for k,v in instr.items() if 'TMA' in k or 'MMA' in k or k in ('LDL','STL','BRA')}
                direct_ops=sum(v for k,v in interesting.items() if 'TMA' in k)
                assert (direct_ops>0)==active
                results.append(dict(variant=variant,batch=b,scale=scale,aligned=aligned,enabled=active,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shared=binary.metadata.shared,
                    cubin_bytes=len(binary.asm['cubin']),cubin_sha256=hashlib.sha256(binary.asm['cubin']).hexdigest(),
                    regs=int(re.search(r'Used (\d+) registers',stats).group(1)),
                    spill=list(map(int,re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',stats).groups())),
                    sass=interesting))
    print(json.dumps(dict(triton=triton.__version__,target='sm90',hardware_execution=False,records=results),indent=2))
if __name__=='__main__':main()
