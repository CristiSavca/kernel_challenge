"""Run in scratch with the pin's jit/include, v68.cuh, v254.cuh and native toolchain."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

base = Path(sys.argv[1]).resolve()
tool = Path.home() / 'nvcc126'
bin = tool / 'toolchain/usr/local/cuda-12.9/bin'
inc = base / 'python/sglang/kernels/jit/include'
includes = ['container/tensor.h','dtype.h','error.h','extra/c_env_api.h','function.h']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
choices = {}
for pdl in ('false', 'true'):
    for warps in (1, 4):
        for provider in ('v68', 'v254_stock', 'v254_split'):
            header = 'v68' if provider=='v68' else 'v254'
            symbol = ('launch_muse_spark_routing' if provider.endswith('split') else 'fused_muse_spark_routing')
            args = f'8,8,0,{warps},{pdl}' + (',true' if provider.endswith('split') else '')
            name = f'{provider}_{warps}_{pdl}'
            src = base/(name+'.cu')
            src.write_text(''.join(f'#include <tvm/ffi/{i}>\n' for i in includes) +
                           f'#include "{header}.cuh"\nnamespace sglang {{\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_muse_spark_routing, ({symbol}<{args}>));\n}}\n')
            obj=base/(name+'.o')
            cmd=[str(bin/'nvcc'),'-O3','-std=c++20','-DSGL_CUDA_ARCH=900','--expt-relaxed-constexpr',
                 '-gencode=arch=compute_90a,code=sm_90a','-Xcompiler','-fPIC','-Xptxas','-v',
                 '-I'+str(tool/'ffi/tvm_ffi/include'),'-I'+str(tool/'compat'),'-I'+str(inc),'-c',str(src),'-o',str(obj)]
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if proc.returncode: raise RuntimeError(proc.stdout)
            asm=subprocess.check_output([str(bin/'cuobjdump'),'-sass',str(obj)]).decode()
            ptx=subprocess.check_output([str(bin/'nvcc'),*cmd[1:cmd.index('-gencode=arch=compute_90a,code=sm_90a')],
                '-arch=compute_90a',*cmd[cmd.index('-Xcompiler'):cmd.index('-c')],'-ptx',str(src),'-o','-'],stderr=subprocess.DEVNULL).decode()
            hot=ptx.split('.visible .entry ')[-1]
            codes=re.findall(r'/\*\w+\*/\s+(?:@!?P\d+ )?(\w+(?:\.\w+)*)\b',asm)
            entry=dict(regs=int(re.search(r'Used (\d+) registers',proc.stdout)[1]),
                       spill_bytes=[int(n) for n in re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',proc.stdout).groups()],
                       fadd=codes.count('FADD'),non_ftz_explicit_add=hot.count('add.rn.f32'),
                       global_reads=sum(op.startswith('LDG') for op in codes),sass_sha=hashlib.sha256(asm.encode()).hexdigest(),
                       object_sha=sha(obj),wrapper_sha=sha(src),header_sha=sha(base/(header+'.cuh')))
            assert entry['spill_bytes']==[0,0]
            if provider.endswith('split'):
                prior=choices[f'v254_stock_{warps}_{pdl}']
                assert entry['non_ftz_explicit_add']==prior['non_ftz_explicit_add']+8,(name,entry,prior)
                assert entry['global_reads']>=prior['global_reads']+2
            choices[name]=entry
print(json.dumps(dict(nvcc=subprocess.check_output([str(bin/'nvcc'),'--version']).decode().strip(),arch='sm_90a',
    projection='native PyTorch torch.mm unchanged',wrappers=choices,physical_execution=False),indent=2))
