"""Execute the pinned coordinator's CUDA branches with observable CPU transport doubles."""
import ast
import hashlib
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

HERE=Path(__file__).resolve().parent
TRACK=HERE.parents[1]
PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
SRC='python/sglang/srt/distributed/parallel_state.py'


def load_coordinator():
    source=subprocess.check_output(['git','-C',TRACK/'sglang','show',PIN+':'+SRC],text=True)
    cls=next(n for n in ast.parse(source).body if getattr(n,'name',None)=='GroupCoordinator')
    methods=[n for n in cls.body if getattr(n,'name',None) in ('all_gather','_all_gather_into_tensor','all_gather_into_tensor')]
    ns={'torch':torch,'is_hip':lambda:False,'_is_cpu':False}
    exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=methods,type_ignores=[])),ns)
    return ns


def run():
    torch.set_num_threads(1)
    ns=load_coordinator()
    receipts={}
    with torch.inference_mode(), patch.object(torch.Tensor,'is_cpu',property(lambda _:False)):
        for backend in ('c10d','pynccl','symm-pynccl','pymscclpp'):
            cases=[]
            for m in (1,8,32,128,256):
                # Distinct rank/row and channel IDs even beyond BF16 integer precision.
                ranks=8;local_n=256
                raw=(torch.arange(ranks*m*local_n).reshape(ranks*m,local_n)%211).to(torch.bfloat16)
                local=raw[:m].clone(); group=SimpleNamespace(world_size=ranks,unique_name='draft-tp',hpu_communicator=None,
                    is_allocation_symmetric=lambda:False,use_symmetric_memory=lambda *a,**k:nullcontext(),
                    is_symmetric_memory_enabled=lambda:backend=='symm-pynccl',device_group=object(),ca_comm=None,
                    debug_check_symmetric_mempool=lambda *args:None)
                real=[]
                def send(out, inp, selected):
                    assert tuple(out.shape)==(ranks*m,local_n) and tuple(inp.shape)==(m,local_n)
                    assert out.is_contiguous() and inp.is_contiguous() and out.dtype==inp.dtype==torch.bfloat16
                    real.append((selected,out.data_ptr()));out.copy_(raw)
                group.pynccl_comm=SimpleNamespace(disabled=backend!='pynccl',change_state=lambda **kw:nullcontext(),
                    all_gather=lambda out,inp:send(out,inp,'symm-pynccl' if backend=='symm-pynccl' else 'pynccl'))
                group.pymscclpp_comm=SimpleNamespace(should_mscclpp_allgather=lambda out,inp:backend=='pymscclpp',
                    all_gather=lambda out,inp:send(out,inp,'pymscclpp'))
                ns['all_gather_single']=lambda out,inp,group:send(out,inp,'c10d')
                ns['reg_all_gather_into_tensor']=lambda out,inp,group_name:ns['_all_gather_into_tensor'](group,out,inp)
                group.all_gather_into_tensor=lambda out,inp:ns['all_gather_into_tensor'](group,out,inp)
                old=ns['all_gather'](group,local,-1)
                bank=ns['all_gather'](group,local,0).view(ranks,m,local_n)
                assert [x[0] for x in real]==[backend,backend]
                assert bank.data_ptr()==real[1][1] and (old.data_ptr()==real[0][1])==(m==1)
                reference=bank.permute(1,0,2).reshape(m,-1);assert torch.equal(old,reference)
                for side in (0,1):
                    for tap in (0,1):
                        pair=2*(2*side+tap)
                        rank_major_tap=bank[pair:pair+2].permute(1,0,2)
                        expected=old[:,(2*side+tap)*512:(2*side+tap+1)*512].reshape(m,2,local_n)
                        assert torch.equal(rank_major_tap,expected)
                        assert rank_major_tap.untyped_storage().data_ptr()==bank.untyped_storage().data_ptr()
                cases.append({'rows':m,'local_payload_bytes':local.nbytes,'gather_output_bytes':bank.nbytes,
                              'old_copy_required':m!=1,'rank_major_gather_output_alias':True})
            receipts[backend]=cases
    return receipts


if __name__=='__main__':
    metadata=json.loads((HERE/'validation.json').read_text())
    assert hashlib.sha256((HERE/'patch.diff').read_bytes()).hexdigest()==metadata['patch_sha256']
    print(json.dumps({'candidate':metadata['patch_sha256'],'torch':torch.__version__,'cuda_transport_modeled':True,
                      'pinned_group_modes':run()},indent=2))
