"""Run V201 slicing through the pinned PyNccl C ABI, then match every rank simultaneously at groupEnd."""
import ast
import ctypes
import hashlib
import json
import math
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

STAGE=Path(sys.argv[1])
NCCL="python/sglang/srt/distributed/device_communicators/pynccl.py"
SRC="python/sglang/srt/layers/communication/sp_owner_sum.py"
MATH=runpy.run_path(str(Path(__file__).with_name("independent_budget.py")))
f32,bf16=MATH["f32"],MATH["bf16"]


def word(value):
    return struct.unpack("!I",struct.pack("!f",bf16(value)))[0] >> 16


def number(value):
    return struct.unpack("!f",struct.pack("!I",value<<16))[0]


class View:
    def __init__(self,shape,device,backing=None,offset=0):
        self.shape,self.device,self.offset,self.dtype=tuple(shape),device,offset,"bf16"
        self.backing=backing if backing is not None else (ctypes.c_uint16*(math.prod(shape)+4))()
        if backing is None:
            self.offset=2
            for j in range(len(self.backing)):self.backing[j]=0x7fc1
    def numel(self):return math.prod(self.shape)
    def data_ptr(self):return ctypes.addressof(self.backing)+2*self.offset
    def __getitem__(self,ix):
        indices=ix if isinstance(ix,tuple) else (ix,)
        dims=[];offset=self.offset
        for j,idx in enumerate(indices):
            size=self.shape[j]
            if isinstance(idx,slice):
                start,end,step=idx.indices(size)
                assert step==1
                dims.append(max(0,end-start))
            else:
                start=idx
            offset+=start*math.prod(self.shape[j+1:])
        return View(dims+list(self.shape[len(indices):]),self.device,self.backing,offset)
    def get(self,i):return self.backing[self.offset+i]
    def set(self,i,val):self.backing[self.offset+i]=val
    def zero_(self):
        for i in range(self.numel()):self.set(i,0)


def compile_methods():
    source=(STAGE/SRC).read_text()
    parsed=ast.parse(source)
    exchange=next(k for k in parsed.body if isinstance(k,ast.FunctionDef) and k.name=="_exchange")
    class EndGroup(ast.NodeTransformer):
        def visit_Expr(self,node):
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                    and node.value.func.attr=="group_end"):
                return [node,ast.Expr(ast.Yield(ast.Constant("flush")))]
            return node
    EndGroup().visit(exchange)
    def finish(local,received,out,rank,size):
        pitch=received.shape[1]*received.shape[2]
        for j in range(local.numel()):
            acc=None
            for source_rank in range(size):
                remote_slot=source_rank if source_rank<rank else source_rank-1
                value=local.get(j) if source_rank==rank else received.get(remote_slot*pitch+j)
                acc=number(value) if acc is None else f32(acc+number(value))
            out.set(j,word(acc))
    env={"_finish":finish}
    exec(compile(ast.fix_missing_locations(ast.Module([exchange],[])),SRC,"exec"),env)
    api=subprocess.check_output(["git","-C",STAGE,"show","HEAD:"+NCCL],text=True)
    cls=next(n for n in ast.parse(api).body if isinstance(n,ast.ClassDef) and n.name=="PyNcclCommunicator")
    names={"send","recv","group_start","group_end","_resolve_stream","change_state"}
    pyenv=dict(contextmanager=contextmanager,buffer_type=int,cudaStream_t=int,
               ncclDataTypeEnum=SimpleNamespace(from_torch=lambda dtype:7 if dtype=="bf16" else -1))
    module=ast.parse("from __future__ import annotations")
    module.body.extend(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names)
    exec(compile(module,NCCL,"exec"),pyenv)
    comm_type=type("Communicator",(),{k:pyenv[k] for k in names})
    return env["_exchange"],comm_type,pyenv,hashlib.sha256(api.encode()).hexdigest()


def replay(exchange,comm_type,pyenv,size,rows,width,window):
    span=(rows+size-1)//size
    plans=[];inputs=[];outs=[];receives=[];communicators=[];events=[[] for _ in range(size)]; active=[False]*size
    def record(src,dst,op,p,n,dtype,comm,stream):
        me=src if op=="send" else dst
        assert comm==me and stream==0x1000+me and dtype==7 and n>0 and active[me]
        events[me].append((src,dst,op,p,n))
    def state(me,value):
        assert active[me] != value
        active[me]=value
    for rank in range(size):
        full=View((rows,width),rank); received=View((size-1,window,width),rank); out=View((span,width),rank)
        inputs.append(full);receives.append(received);outs.append(out)
        for i in range(full.numel()):
            val=((i*13+rank*53)%967-483)*2.0**(rank%8-2)
            if i%23==0:val=([2**30,1.,-(2**30),1.]+[0.]*size)[rank]
            if i%37==11:val=-0.
            full.set(i,word(val))
        comm=comm_type();comm.device=rank;comm.comm=rank;comm.disabled=True;comm.available=True
        comm.nccl=SimpleNamespace(ncclGroupStart=lambda me=rank:state(me,True),
            ncclGroupEnd=lambda me=rank:state(me,False),
            ncclSend=lambda p,n,t,dst,me,s:record(me,dst,"send",p,n,t,me,s),
            ncclRecv=lambda p,n,t,src,me,s:record(src,me,"recv",p,n,t,me,s))
        communicators.append(comm)
        group=SimpleNamespace(pynccl_comm=comm,rank_in_group=rank,world_size=size)
        plans.append(exchange(full,group,out,received))
    messages=0;elements=0
    for offset in range(0,span,window):
        for me,fn in enumerate(plans):
            pyenv["get_current_device_stream_fast"]=lambda me=me:SimpleNamespace(cuda_stream=0x1000+me)
            assert next(fn)=="flush"
        assert not any(active)
        lookup={}
        for me in range(size):
            for src,dst,kind,ptr,n in events[me]:
                key=(src,dst,kind)
                assert key not in lookup
                lookup[key]=(ptr,n)
            events[me].clear()
        for src in range(size):
            for dst in range(size):
                count=max(0,min(window,span-offset,rows-dst*span-offset))
                pulled=lookup.pop((src,dst,"recv"),None);sent=lookup.pop((src,dst,"send"),None)
                if src==dst or not count:
                    assert pulled is None and sent is None
                    continue
                slot=src if src<dst else src-1
                assert sent==(inputs[src].data_ptr()+2*width*(dst*span+offset),width*count)
                assert pulled==(receives[dst].data_ptr()+2*slot*width*window,width*count)
                ctypes.memmove(pulled[0],sent[0],2*sent[1]); messages+=1;elements+=sent[1]
        assert not lookup
    assert all(next(fn,None) is None for fn in plans)
    assert all(c.disabled for c in communicators)
    for rank,out in enumerate(outs):
        for i in range(out.numel()):
            expected=0.
            k=rank*span*width+i
            if k<rows*width:
                expected=number(inputs[0].get(k))
                for arr in inputs[1:]:expected=f32(expected+number(arr.get(k)))
                if k%37==11:assert word(expected)==0x8000
            assert out.get(i)==word(expected)
        for view in (out,receives[rank],inputs[rank]):
            assert list(view.backing[:2])+list(view.backing[-2:])==[0x7fc1]*4
    assert elements==(size-1)*rows*width
    return messages,elements,size*span*width


if __name__=="__main__":
    exchange,comm,pyenv,nccl_sha=compile_methods()
    cases=messages=wire=outputs=0
    for size in (2,3,4,5,7,8,16):
        for rows in (size,size+1,2*size-1,4*size,4*size+1,23*size-2):
            for width,window in ((1,1),(5,3),(65,7)):
                m,e,n=replay(exchange,comm,pyenv,size,rows,width,window)
                messages+=m;wire+=e;outputs+=n;cases+=1
    print(json.dumps(dict(cases=cases,peer_messages=messages,BF16_words_on_wire=wire,
            output_and_padding_words=outputs,communicator_sizes=[2,3,4,5,7,8,16],
            source_sha256=hashlib.sha256((STAGE/SRC).read_bytes()).hexdigest(),
            pinned_nccl_sha256=nccl_sha, transport="actual PyNccl group/send/recv/device/stream wrappers, simulated NCCL C calls and exact live pointer/count checks at every groupEnd"),indent=2))
