"""Replay both production kernels on CPU. Pass the applied SGLang checkout and test-tool directory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import ml_dtypes
import numpy as np
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="directory containing production metadata.py and paged_verify_attention.py")
    parser.add_argument("tools", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tools.resolve()))
    from compile_decode_sm90 import load_jit_module
    from interpret_decode_kernels import buf
    from triton.runtime import interpreter as interp

    # The stock interpreter performs BF16 dot on the UINT16 storage representation.
    # Correct this CPU operation for both sides; the actual SM90 code is also compiled separately.
    original_dot = interp.InterpreterBuilder.create_dot
    def numeric_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.dtype == tl.bfloat16 and b.dtype == tl.bfloat16:
            av = a.data.view(ml_dtypes.bfloat16).astype("f4")
            bv = b.data.view(ml_dtypes.bfloat16).astype("f4")
            return interp.TensorHandle(av @ bv + d.data, d.dtype.scalar)
        return original_dot(self, a, b, d, input_precision, max_num_imprecise_acc)
    interp.InterpreterBuilder.create_dot = numeric_dot
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        meta = load_jit_module(args.src/"metadata.py", work)._fused_decode_metadata_kernel
        att = load_jit_module(args.src/"paged_verify_attention.py", work)
        old = load_jit_module(args.src/"incumbent.py", work)
        rng = np.random.default_rng(192)
        ct = dict(metadata_invocations=0, selected_origins=0, rejected_live_tiles=0,
                  compared_physical_ids=0, attention_invocations=0, fp32_partials=0,
                  bf16_outputs=0, dense_reference_outputs=0, direct_segments=0, indexed_segments=0)
        for B in (1, 3, 4):
            N, rows, H, D, G, W = 705, 1456, 2, 64, 8, 8
            page_table = np.full((B,N), -38, "i4")
            origins = np.full((B,triton.cdiv(N,128)), -213, "i4")
            cache = np.full(B, 0, "i4")
            prefix = np.full(B+1, 0, "i4")
            req = np.arange(B, dtype="i4")[::-1].copy()
            for i,(lens,scenario) in enumerate([
                (np.resize([511,560,257,321],B), "affine"),
                (np.resize([650,383,704,457],B), "mixed"),
                (np.resize([224,640,399,272],B), "zero"),
                (np.resize([128,256,643,519],B), "shuffled"),
                (np.resize([257,129,512,225],B), "banklimit"),
            ]):
                lens = lens.astype("i4")
                table = np.zeros((B,N), "i4")
                for r in range(B):
                    for t in range(triton.cdiv(N,128)):
                        begin = 1 + ((r+2*t+i)%9)*129
                        part = table[r,t*128:(t+1)*128]
                        part[:] = np.arange(begin,begin+len(part))
                        if scenario == "mixed" and t%3==1 and len(part)>75:
                            part[12],part[75]=part[75],part[12]
                        if scenario == "zero" and t%2==0:
                            part[min(17,len(part)-1)]=0
                        if scenario == "shuffled" and t%3!=2:
                            rng.shuffle(part)
                block=128 if i%2==0 else 256
                meta[(B,triton.cdiv(N,block))](buf(lens-8,tl.int32),1,buf(table,tl.int32),N,1,
                    buf(req,tl.int32),1,buf(cache,tl.int32),1,buf(prefix,tl.int32),1,
                    buf(page_table,tl.int32),N,1,None,0,0,None,0,None,0,None,0,B,N,
                    PAGE_SIZE=1,SEQ_LEN_DELTA=8,BLOCK_COLS=block,SKIP_PAGE_TABLE=False,ZERO_TAIL=False,
                    WINDOW_LEFT=-1,origins=buf(origins,tl.int32),ORIGIN_STRIDE=origins.shape[1])
                np.testing.assert_array_equal(cache,lens)
                np.testing.assert_array_equal(prefix,np.r_[0,lens.cumsum()])
                ct["metadata_invocations"] += 1
                for r,L in enumerate(lens):
                    np.testing.assert_array_equal(page_table[r,:L],table[req[r],:L])
                    ct["compared_physical_ids"] += int(L)
                    for t in range(triton.cdiv(int(L),128)):
                        tile=table[req[r],t*128:min(L,128*(1+t))]
                        valid=len(tile)==128 and tile[0]>0 and np.all(tile==tile[0]+np.arange(128))
                        assert origins[r,t] == (tile[0] if valid else -1)
                        ct["selected_origins" if valid else "rejected_live_tiles"] += 1
                # Real pitched NHD banks; each tensor map uses its own row pitch and physical extent.
                pitches=(160,192)
                q=rng.integers(-3,4,(B*W,H*G,D)).astype(ml_dtypes.float8_e4m3fn)
                k=rng.integers(-3,4,(rows,pitches[0])).astype(ml_dtypes.float8_e4m3fn)
                v=rng.integers(-3,4,(rows,pitches[1])).astype(ml_dtypes.float8_e4m3fn)
                Q,K,V=[buf(arr.view("u1"),tl.float8e4nv) for arr in (q,k,v)]
                descs=[TensorDescriptor(bank,[rows,H*D],[stride,1],[128,64]) for bank,stride in zip((K,V),pitches)]
                cu=np.arange(B+1,dtype="i4")*W
                cap=300 if scenario=="banklimit" else rows
                partitions=5
                counts=(lens+127)//128
                width=triton.cdiv(int(counts.sum()),partitions)
                pos=0
                for r,n in enumerate(counts):
                    for part in range(partitions):
                        a=max(0,part*width-pos);z=min(n,(part+1)*width-pos)
                        if a<z:
                            ids=origins[r,a:z]
                            hit=bool(np.all((ids>0)&(ids<=cap-128)))
                            ct["direct_segments" if hit else "indexed_segments"]+=1
                    pos+=int(n)
                outputs=[]
                ks=(1 + np.arange(B*H,dtype="f4").reshape(B,H)%3/8).astype("f4")
                vs=np.roll(ks,1).copy()
                for mod,direct in [(old,False),(att,True)]:
                    partial=np.full((partitions+B-1,H,W*G,D),-37,"f4")
                    lse=np.full(partial.shape[:-1],-18,"f4")
                    out=np.zeros(q.shape,dtype="u2")
                    extra={} if not direct else dict(Origins=buf(origins,tl.int32),KDesc=descs[0],VDesc=descs[1],
                                                      ORIGIN_STRIDE=origins.shape[1],CAPACITY=cap)
                    mod._partition_attention[(partitions,H)](Q,K,V,buf(page_table,tl.int32),buf(lens,tl.int32),
                        buf(cu,tl.int32),buf(ks,tl.float32),buf(partial,tl.float32),buf(lse,tl.float32),
                        N,1,pitches[0],pitches[0],D,pitches[1],pitches[1],D,H,1,B,H,G,D,
                        math.log2(math.e)/8,-1,W,W*G,128,partitions,triton.next_power_of_2(B),**extra)
                    mod._merge_partitions[(B,H,W*G)](buf(partial,tl.float32),buf(lse,tl.float32),buf(lens,tl.int32),
                        buf(cu,tl.int32),buf(vs,tl.float32),buf(out,tl.bfloat16),H,1,B,H,G,D,-1,W*G,128,
                        partitions,triton.next_power_of_2(B),16)
                    outputs.append((partial,lse,out))
                for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)
                ct["attention_invocations"]+=2
                ct["fp32_partials"]+=int(partial.size)
                ct["bf16_outputs"]+=int(out.size)
                # Confirm actual causal row order on representative queries/head IDs against direct dense math.
                for r,L in enumerate(lens):
                    for token,head in [(0,0),(7,15),(2,9)]:
                        ids=table[req[r],:L-W+token+1];h=head//G
                        kval=k[ids,h*D:(h+1)*D].astype("f4")
                        vval=v[ids,h*D:(h+1)*D].astype("f4")
                        score=kval@q[r*W+token,head].astype("f4")*(ks[r,h]/8)
                        w=np.exp(score-score.max())
                        ref=(w@vval)/w.sum()*vs[r,h]
                        np.testing.assert_allclose(out[r*W+token,head].view(ml_dtypes.bfloat16).astype("f4"),
                                                   ref,rtol=0.05,atol=0.04)
                        ct["dense_reference_outputs"]+=D
        assert ct["direct_segments"]>0 and ct["indexed_segments"]>0
        print(json.dumps(dict(triton=triton.__version__,physical_h200=False,method="production interpreter with BF16 dot storage decoding",
            source_sha256={f:digest(args.src/f) for f in ["metadata.py","paged_verify_attention.py","incumbent.py"]},**ct),indent=2))
if __name__=='__main__':
    main()
