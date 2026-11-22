"""Exercise actual Triton CPU interpreter against path-based, whole-batch oracles."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np


def main():
    import triton
    import triton.language as tl
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
    from compile_decode_sm90 import load_jit_module
    from interpret_decode_kernels import buf
    from cpu_check import H, S, paths, choice, mutation, reward

    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path)
    args=ap.parse_args()
    def x(a):
        a=np.ascontiguousarray(a);return buf(a,{np.dtype('int64'):tl.int64,np.dtype('int32'):tl.int32,np.dtype('float32'):tl.float32}[a.dtype])
    rng=np.random.default_rng(1992);counts={'proposals':0,'selected_trees':0,'votes':0}
    with tempfile.TemporaryDirectory() as tmp:
        m=load_jit_module(args.source,Path(tmp))
        for w,k,cap,bs in ((2,2,3,3),(5,4,6,3),(8,16,4,3),(8,5,3,1)):
            eb=triton.next_power_of_2(w*k);block=triton.next_power_of_2(w)
            for case in range(3):
                scores=rng.normal(0,1,(bs,w-1,k,k)).astype('f4')
                if case==1:scores.fill(0.)
                if case==2:scores[...,0]=np.nan;scores[0,0,0,1]=np.inf
                ids=np.stack([np.stack([rng.permutation(k)+930 for _ in range(w-1)]) for _ in range(bs)]).astype('i8')
                anchors=np.arange(bs,dtype='i8')+29
                orig=np.full((4,cap,w),-101,dtype='i8')
                alt=np.full((2,4,cap,w),-101,dtype='i8')
                m._propose_tree[(bs,3)](x(ids),x(scores),x(anchors),*map(x,orig),1,w,k,eb,x(alt),cap*w,num_warps=4)
                for b in range(bs):
                    for i,beta in enumerate((1.,.5,2.)):
                        got=orig[:,b] if i==0 else alt[i-1,:,b]
                        np.testing.assert_array_equal(H['proposal_oracle'](ids[b],scores[b]*beta,anchors[b]),got)
                        counts['proposals']+=1
                assert np.all(alt[:,:,bs:]==-101) and np.all(orig[:,bs:]==-101)
                orig=np.ascontiguousarray(orig[:,:bs])
                state=rng.integers(-w,w+1,(bs,w+3),dtype='i4')
                base=orig.copy();selected=orig.copy()
                chain=np.empty((bs,w),dtype='i8')
                for b in range(bs):
                    rank=0;chain[b,0]=anchors[b]
                    for d in range(w-1):
                        rank=int(np.argmax(np.nan_to_num(scores[b,d,rank],nan=-np.inf))); chain[b,d+1]=ids[b,d,rank]
                    state[b,w:]=[(0,-2,4),(6,2,1),(1,6,-5)][(b+case)%3]
                m._select_spine_tree[(bs,)](*map(x,(orig,chain,state,base)),bs*w,w,block,w+3,num_warps=1)
                m._select_spine_tree[(bs,)](*map(x,(orig,chain,state,selected)),bs*w,w,block,w+3,x(alt),cap*w,num_warps=1)
                observed=np.empty(bs,dtype='i4');out=np.full((bs,w),-15555,dtype='i8')
                expected=state.copy()
                for b in range(bs):
                    shapes=S['make_family'](orig[:,b],chain[b])
                    util=np.r_[0,np.cumsum(state[b,1:w])]
                    depth=w-1 if state[b,0]>0 else 0
                    if w>2 and util[1:-1].max()>max(0,util[-1]):depth=1+int(np.argmax(util[1:-1]))
                    np.testing.assert_array_equal(base[:,b],shapes[depth][0])
                    options=[base[:,b],alt[0,:,b],alt[1,:,b]]
                    np.testing.assert_array_equal(selected[:,b],options[choice(state[b,w:])])
                    counts['selected_trees']+=1
                    union=sorted(set.union(*(set(paths(t)) for t in options)))
                    prefix=list(union[rng.integers(len(union))]);sequence=prefix+[-665]*(w-len(prefix))
                    size=reward(selected[:,b],sequence);out[b,:size]=sequence[:size];observed[b]=size
                    lower=[reward(t,sequence[:size]) for t in options]
                    for j,(a,c) in enumerate(((1,0),(2,0),(1,2))):
                        expected[b,w+j]=mutation(state[b,w+j],lower[a],lower[c],size,w)
                        counts['votes']+=1
                copied=np.full_like(state,-7654)
                m._update_spine_margins[(bs,)](*map(x,(orig[0],orig[2],chain,out,observed,state,copied)),w,block,w+3,num_warps=1)
                assert np.all(copied[:,w:]==-7654)
                m._update_spine_margins[(bs,)](*map(x,(orig[0],orig[2],chain,out,observed,state,copied)),w,block,w+3,x(alt),cap*w,num_warps=1)
                np.testing.assert_array_equal(copied[:,w:],expected[:,w:])
    print(json.dumps({'triton':triton.__version__,**counts},indent=2))


if __name__=='__main__':main()
