"""Exercise the actual compiled-source row reduction using host replacements for CUDA intrinsics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl
from triton.language.core import tensor
from triton.runtime.interpreter import TensorHandle

from sm90 import load
from interpret_decode_kernels import buf


def tanh(x):
    return tensor(TensorHandle(np.tanh(x.handle.data).astype(np.float32),tl.float32), x.type)


def main(a):
    grids=rows=values=guards=0
    rng=np.random.default_rng(237)
    with tempfile.TemporaryDirectory() as tmp:
        parent=load(a.parent,a.numerics,Path(tmp))
        new=load(a.new,a.numerics,Path(tmp))
        for mod in (sys.modules[a.parent.stem+'_test'],sys.modules[a.new.stem+'_test']):
            mod.libdevice.tanh=tanh
            mod.mul_rn_f32=lambda x,y: x*y
        for width,valid in ((1,0),(7,6),(4095,4095),(4096,4081),(4097,4097),(25256,25256)):
            pitch=width+3
            logits=rng.normal(0,35,(10,pitch)).astype(np.float32)
            logits[1]=0.; logits[1,1::2]=-0.
            logits[2]=np.nan; logits[3,1::7]=np.nan
            logits[4]=np.inf; logits[5]=-np.inf
            logits[6,:]=-2000.; logits[6,::4096]=np.inf
            logits[7]=np.float32(-1e-40)
            logits[8]=np.float32(4.25); logits[8,::427]=np.float32(10000.)
            logits[9]=-99.; logits[9,:2]=np.array([0x41000011,0x41000012],dtype='u4').view('f4')
            logits[:,valid:]=np.nan
            for scale in (0.17677669529663687,1.625,np.finfo('f4').tiny.item(),1e20):
                for first in (True,False):
                    inputs=logits.copy()
                    if not first:
                        # The stock interpreter raises on an all-NaN 4,096-lane tile.
                        inputs[2,0:valid:4096] = 0.
                    output=[]
                    with np.errstate(all='ignore'):
                        for kernel,inp,extra in ((parent,inputs*np.float32(scale),{}),(new,inputs,dict(SCALE=scale))):
                            out=np.full((22,),177.,dtype='f4')
                            kernel[(10,)](buf(inp,tl.float32),buf(out[1:-1],tl.float32),pitch,valid,2*width,20.,
                                          BLOCK_SIZE=4096,FIRST_NAN=first,num_warps=8,**extra)
                            assert np.all(out[[0,-1]]==177.)
                            output.append(out.copy().view('u4'))
                    assert np.array_equal(*output)
                    grids+=2; rows+=10; values+=10*valid; guards+=4
    return dict(kernel_calls=grids,all_nan_interpreter_limit="Native all-NaN tile (FIRST_NAN=False) cannot run on NumPy nanargmax. Each such fixture tile retains one numeric lane; FIRST_NAN=True includes all-NaN tiles.",paired_winner_rows=rows,live_vocab_scores=values,guard_words=guards,
        scope='Actual Triton interpreter reduction. Host NumPy tanh and FP32 multiply replace CUDA libdevice and inline PTX for these CPU calls. SM90 compilation independently checks the actual intrinsics.',
        source_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in vars(a).items()},physical_h200=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('parent','new','numerics'): ap.add_argument(name,type=Path)
    print(json.dumps(main(ap.parse_args()),indent=2))
