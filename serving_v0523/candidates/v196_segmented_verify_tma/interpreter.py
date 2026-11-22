"""Run the exact production run planner, segmented attention, V195 comparator and final merge."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import hashlib
import ml_dtypes
import numpy as np
import triton
import triton.language as tl
from triton.runtime import interpreter as it
from triton.tools.tensor_descriptor import TensorDescriptor


def expected(good):
    n=len(good);i=0;answer=[]
    while i<n:
        if n<=32:
            answer.append((n,bool(np.all(good))));break
        direct=all(good[i:i+4]) and i+4<=n
        j=i+1
        if direct:
            while j<n and good[j]:j+=1
        else:
            while j<n and not (j+4<=n and all(good[j:j+4])):j+=1
        answer.append((j,direct));i=j
    return answer


def numerical(module, prior, buf, rng, witness=None):
    # Correct the stock interpreter's BF16 dot on raw UINT16 storage for both modules.
    native=it.InterpreterBuilder.create_dot
    def dot(self,a,b,c,*flags):
        if a.dtype==tl.bfloat16 and b.dtype==tl.bfloat16:
            return it.TensorHandle(a.data.view(ml_dtypes.bfloat16).astype('f4') @
                b.data.view(ml_dtypes.bfloat16).astype('f4')+c.data,c.dtype.scalar)
        return native(self,a,b,c,*flags)
    it.InterpreterBuilder.create_dot=dot
    ct=dict(paired_rounds=0,activations={},direct_runs=0,indexed_runs=0,indexed_to_direct=0,
            direct_to_indexed=0,partials=0,lse=0,bf16=0,dense_reference_values=0)
    for b,nt,pattern in [(1,33,'boundary'),(3,68,'interior'),(5,42,'jagged'),(7,35,'tail'),
                        (1,98,'long'),(3,64,'all'),(3,63,'none'),(1,32,'boundary')]:
        # Each run is evaluated on native partition intersections. No model sees these synthetic patterns.
        parts=b;H,D,W,G=1,64,8,2;m=W*G
        counts=np.resize(np.array([nt,nt-1,nt-2,nt+1],dtype='i4'),b)
        seq=counts*128-np.resize(np.array([0,3,8],dtype='i4'),b)
        cap=8294;k_pitch,v_pitch=80,96
        width=int(counts.max())*128
        pages=np.zeros((b,width),'i4'); origins=np.full((b,width//128),-1,'i4')
        cacheid=np.arange(128,dtype='i4')
        goodrows=[]
        for r,c in enumerate(counts):
            good=np.ones(c,dtype=bool)
            if pattern=='interior':
                bad=[0,1,3,17,28,32,35,39,43,45,47,54,56,58,61,63]
            elif pattern=='jagged':bad=[3,6,7,9,15,19,22,25,28,31,38]
            elif pattern=='long':bad=[1,34,67]
            elif pattern=='none':bad=list(range(1,int(c),3))
            elif pattern=='tail':bad=[4,9,29,32]
            elif pattern=='boundary':bad=[0,28]
            else:bad=[]
            good[[n for n in bad if n<c]]=False
            for t in range(c):
                first=int(1+(r*349+t*131)%8000)
                ids=first+cacheid
                if not good[t]:
                    ids=ids.copy();ids[47],ids[83]=ids[83],ids[47]
                pages[r,t*128:(t+1)*128]=ids
                if good[t] and (t+1)*128<=seq[r]:origins[r,t]=first
            goodrows.append(origins[r,:c]>0)
        per=triton.cdiv(int(counts.sum()),parts);base=0
        for row,(good,c) in enumerate(zip(goodrows,counts)):
            for part in range(base//per,(base+c-1)//per+1):
                lo=max(0,part*per-base);hi=min(c,(part+1)*per-base)
                runs=expected(good[lo:hi])
                ct['activations'][str(hi-lo)]=ct['activations'].get(str(hi-lo),0)+1
                for j,(_,d) in enumerate(runs):
                    ct['direct_runs' if d else 'indexed_runs']+=1
                    if j:ct['indexed_to_direct' if d else 'direct_to_indexed']+=1
            base+=c
        fp8=ml_dtypes.float8_e4m3fn
        q=rng.integers(-3,4,(b*W,G,D)).astype(fp8);k=rng.integers(-3,4,(cap,k_pitch)).astype(fp8)
        v=rng.integers(-3,4,(cap,v_pitch)).astype(fp8)
        Q,K,V=[buf(x.view('u1'),tl.float8e4nv) for x in [q,k,v]]
        Kd=TensorDescriptor(K,[cap,D],[k_pitch,1],[128,D])
        Vd=TensorDescriptor(V,[cap,D],[v_pitch,1],[128,D])
        cu=np.arange(b+1,dtype='i4')*W
        ks=np.linspace(.8,1.1,b,dtype='f4');vs=np.linspace(.9,.6,b,dtype='f4')
        outs=[]
        for mod in (prior,module):
            pa=np.full((parts+b-1,H,m,D),-37,'f4');lse=np.full(pa.shape[:-1],-555,'f4')
            o=np.zeros(q.shape,'u2')
            mod._partition_attention[(parts,H)](Q,K,V,buf(pages,tl.int32),buf(seq,tl.int32),buf(cu,tl.int32),
                buf(ks,tl.float32),buf(pa,tl.float32),buf(lse,tl.float32),width,1,k_pitch,k_pitch,D,
                v_pitch,v_pitch,D,1,0,b,H,G,D,math.log2(math.e)/8,-1,W,m,128,parts,triton.next_power_of_2(b),
                Origins=buf(origins,tl.int32),KDesc=Kd,VDesc=Vd,ORIGIN_STRIDE=origins.shape[1],CAPACITY=cap)
            mod._merge_partitions[(b,H,m)](buf(pa,tl.float32),buf(lse,tl.float32),buf(seq,tl.int32),
                buf(cu,tl.int32),buf(vs,tl.float32),buf(o,tl.bfloat16),1,0,b,H,G,D,-1,m,128,
                parts,triton.next_power_of_2(b),16)
            outs.append((pa,lse,o))
        for left,right in zip(*outs):np.testing.assert_array_equal(left,right)
        if witness is not None and (b,nt,pattern)==(1,33,'boundary'):
            runs=expected(goodrows[0])
            assert runs==[(1,False),(28,True),(29,False),(33,True)]
            words=o.reshape(-1)[:256].reshape(8,32)
            assert len(np.unique(words))>=20
            witness.write_text(json.dumps(dict(
                dtype="BF16 uint16 memory",shape=[8,32],source_shape=list(o.shape),
                case="B1 33 tiles, one partition, exchanged physical rows inside tiles 0 and 28",
                seed=196,source_order="First 256 consecutive BF16 words in query, head, dimension order",
                transitions=[dict(stop=int(stop),direct=bool(d)) for stop,d in runs],
                values=words.tolist()),indent=2)+'\n')
        ct['paired_rounds']+=1;ct['partials']+=pa.size;ct['lse']+=lse.size;ct['bf16']+=o.size
        for r in range(b):
            t,h=(r+1)%W,r%G
            ids=pages[r,:seq[r]-W+t+1]
            score=k[ids,:D].astype('f4')@q[r*W+t,h].astype('f4')*(ks[r]/8)
            ex=np.exp(score-score.max());ref=(ex@v[ids,:D].astype('f4'))/ex.sum()*vs[r]
            np.testing.assert_allclose(o[r*W+t,h].view(ml_dtypes.bfloat16).astype('f4'),ref,atol=.035,rtol=.045)
            ct['dense_reference_values']+=D
    assert ct['indexed_to_direct']>0 and ct['direct_to_indexed']>0
    return ct


def planner(module, buf, rng, tmp):
    # Drive actual _next_tile_run on both sides of 32-bit mask boundaries.
    dest=Path(tmp)/'plan_probe.py'
    dest.write_text('import triton\nimport triton.language as tl\nfrom '+module.__name__+''' import _next_tile_run
@triton.jit
def scan(O, S, E, T, D, P: tl.constexpr, LIMIT: tl.constexpr):
    r=tl.program_id(0)
    s=tl.load(S+r)
    e=tl.load(E+r)
    stop,direct=_next_tile_run(O+r*P, s, e, LIMIT)
    tl.store(T+r,stop)
    tl.store(D+r,direct)
''')
    spec=importlib.util.spec_from_file_location('plan_probe',dest);m=importlib.util.module_from_spec(spec)
    sys.modules['plan_probe']=m;spec.loader.exec_module(m)
    cases=[]
    for length in [1,3,4,5,27,28,29,30,31,32,33,34,35,59,61,62,63,64,65,95,96,97,167]:
        for style in range(8):
            g=rng.random(length)<(.4 if style%2==0 else .9)
            if style==0:g[:]=True
            if style==1:g[:]=False
            if style in (2,3,4):g[:]=False;start=min(length,max(0,32-style+1));g[start:min(start+7,length)]=True
            if style==5:g[:]=True;g[min(31,length-1)]=False
            if style==6:g[:]=False;g[1::2]=True
            for n in (0,1,7):
                a=np.where(g,3,-1).astype('i4')
                # Other rejects include sink and a valid origin above the descriptor limit.
                a[(a<0)&(np.arange(length)%3==0)]=0;a[(a<0)&(np.arange(length)%3==2)]=10000
                source=np.r_[np.resize([201,17,0],n),a]
                s=n
                while s<len(source):
                    suffix=(source[s:]>0)&(source[s:]<=8192)
                    direct=len(suffix)>=4 and all(suffix[:4]);j=1
                    if direct:
                        while j<len(suffix) and suffix[j]:j+=1
                    else:
                        while j<len(suffix) and not(j+4<=len(suffix) and all(suffix[j:j+4])):j+=1
                    cases.append((source,s,s+j,direct));s+=j
    count=len(cases);pitch=max(len(row) for row,*_ in cases)+1
    o=np.zeros((count,pitch),'i4');starts=np.zeros(count,'i4');ends=starts.copy();stops=starts.copy();ds=starts.copy()
    for i,(row,s,_,_) in enumerate(cases):o[i,:len(row)]=row;starts[i]=s;ends[i]=len(row)
    m.scan[(count,)](buf(o,tl.int32),buf(starts,tl.int32),buf(ends,tl.int32),buf(stops,tl.int32),
        buf(ds,tl.int32),pitch,8192)
    np.testing.assert_array_equal(stops,[c[2] for c in cases]);np.testing.assert_array_equal(ds,[c[3] for c in cases])
    return dict(choices=count,unbroken_scanner_progress=int((stops>starts).sum()),physical_positivity_and_extent=True)


def main():
    here,tools=map(Path,sys.argv[1:3]);sys.path.insert(0,str(tools.resolve()))
    from compile_decode_sm90 import load_jit_module
    from interpret_decode_kernels import buf
    rng=np.random.default_rng(196)
    with tempfile.TemporaryDirectory() as tmp:
        p=here/'paged_verify_attention.py';old=here/'prior.py'
        src=load_jit_module(p,Path(tmp));baseline=load_jit_module(old,Path(tmp))
        result=dict(triton=triton.__version__,physical_H200=False,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    parent_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),planner=planner(src,buf,rng,tmp),
                    numerical=numerical(src,baseline,buf,rng,Path(sys.argv[3]) if len(sys.argv)>3 else None))
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
