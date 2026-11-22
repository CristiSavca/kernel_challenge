"""Execute the submitted raster producer and both consumers against the preceding tree."""
import ast
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
KERNEL = 'python/sglang/kernels/ops/attention/paged_verify_attention.py'
stage = Path(sys.argv[1])
parent = json.loads((HERE.parent/'v247_coalesced_serving_decode/validation.json').read_text())
prior = subprocess.check_output(['git','-C',stage,'show',parent['tree']+':'+KERNEL]).decode()
current = (stage/KERNEL).read_text()
old_check = next(HERE.parent.glob('v62_*/cpu_check.py'))
support = {}
tree = ast.parse(old_check.read_text())
end = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and n.targets[0].id=='parser')
exec(compile(ast.Module(tree.body[:end],[]),str(old_check),'exec'),support)
tl, Ptr, Vec = [support[x] for x in ('tl','Pointer','Vector')]
stats = Counter()
rng = np.random.default_rng(248)

def vector(x):
    return np.asarray(x).view(Vec)

def store(pointer, values, mask=True):
    offsets, live, chosen = support['bounds'](pointer,mask)
    assert len(chosen)==len(np.unique(chosen))
    array = pointer.data
    if array.dtype==np.int32:
        assert np.all(array[chosen]==-77)
    else:
        assert np.all(np.isnan(array[chosen]))
    array[chosen] = np.broadcast_to(values,offsets.shape)[live]

def dot(a,b,acc=0,**kw):
    return vector(np.einsum('ij,jk->ik',a,b,optimize=False)+acc)

tl.store, tl.dot, tl.uint32, tl.int1 = store,dot,np.uint32,np.bool_
tl.trans = lambda a: vector(np.transpose(a))
tl.where = lambda c,a,b: vector(np.where(c,a,b))
tl.zeros = lambda shape,dtype: vector(np.zeros(shape,dtype))
tl.full = lambda shape,v,dtype: vector(np.full(shape,v,dtype))
def methods(source):
    m=ast.parse(source)
    fns=[n for n in m.body if isinstance(n,ast.FunctionDef) and n.decorator_list]
    for n in fns:n.decorator_list=[]
    env={'tl':tl}
    exec(compile(ast.Module(fns,[]),KERNEL,'exec'),env)
    return env
old, new = methods(prior),methods(current)

def raster(lens,cu,window,p):
    b=len(lens); n=64 if window>=0 else 128; bb=1<<(b-1).bit_length()
    dst=np.full(2*b+2+p,-77,np.int32)
    order=rng.permutation((p+15)//16)
    for ch in order:
        support['coords']=(int(ch),0,0)
        new['_build_verify_raster'](Ptr(lens),Ptr(cu),Ptr(dst),b,p,window,n,bb,16)
    q=np.diff(cu)
    first=np.maximum(lens-q-window,0)//n if window>=0 else 0*lens
    counts=np.where(q>0,(lens+n-1)//n-first,0)
    ends=np.concatenate(([0],np.cumsum(counts)))
    width=max(1,(int(ends[-1])+p-1)//p)
    owners=np.searchsorted(ends[1:],np.arange(p)*width,side='right')
    ref=np.concatenate((ends,first,[width],owners))
    assert np.array_equal(dst,ref)
    stats['producer_tables']+=1;stats['plan_words']+=len(dst)
    return dst

class Desc:
    def __init__(self,a,tile):self.a,self.tile=a,tile
    def load(self,coords):
        i,j=map(int,coords);tile=self.tile
        assert 0<i and i+tile<=self.a.shape[0] and j==0
        return vector(self.a[i:i+tile,0,0,:])

for b,p,w in itertools.product((1,2,3,4,7,16,31,32,63,64),(1,7,16,17,66,132,264),(-1,0,63,2047)):
    for style in range(4):
        q=rng.integers(0,17,b,dtype=np.int32)
        lens=rng.integers(0,196609,b,dtype=np.int32)
        if style==1:q[::2]=0
        if style==2:q[:]=lens[:]=0
        if style==3:lens[:]=0;lens[-1]=65537;q[-1]=8
        raster(lens,np.r_[np.int32(0),q.cumsum(dtype=np.int32)],w,p)

for i in range(54):
    b=(2,3,4,7,13,32)[i%6];p=(1,3,17,14,11)[i%5];groups=(2,4)[i%2];qmax=8
    w=(-1,0,63,2047)[i%4]; n=128 if w<0 else 64; m=qmax*groups; d=64
    lens=rng.integers(8,333,b,dtype=np.int32)
    qn=np.full(b,qmax,np.int32) if i%3 else rng.integers(0,qmax+1,b,dtype=np.int32)
    if i%7==0:lens[0]=0
    cu=np.r_[np.int32(0),qn.cumsum(dtype=np.int32)]
    desc=i%3!=0
    maxlen=int(lens.max()); nt=(maxlen+n-1)//n
    table=np.empty((b,nt*n),np.int32)
    # A randomized permutation of complete token tiles gives a genuine mixed
    # map. Each certified tile is an actual contiguous physical run.
    tiles=rng.permutation(b*nt).reshape(b,nt)
    for row in range(b):table[row]=((tiles[row,:,None]*n+1)+np.arange(n)).reshape(-1)
    phys=1+b*nt*n
    k,v=(rng.integers(-8,9,(phys,1,1,d),dtype=np.int16).astype('f4')/16 for _ in range(2))
    q=rng.integers(-8,9,(int(cu[-1]),groups,d),dtype=np.int16).astype('f4')/16
    sc,sv=(rng.uniform(.5,1.5,b).astype('f4') for _ in range(2))
    origins=np.full((b,nt+1),-1,np.int32)
    for row in range(b):
        base=max(int(lens[row])-qmax-w,0)//n if w>=0 else 0
        full=int(lens[row])//n
        for t in range(base,full):
            if i%4!=1 or t%2:
                origins[row,t-base] = table[row,t*n]
    outplan=raster(lens,cu,w,p)
    def run(mod,plan=None):
        partial=np.full((p+b-1,1,m,d),np.nan,'f4');lse=np.full(partial.shape[:-1],np.nan,'f4')
        out=np.full_like(q,np.nan);bb=1<<(b-1).bit_length()
        scales=i%4!=0
        for part in rng.permutation(p):
            support['coords']=(int(part),0,0)
            kw={} if plan is None else {'Plan':Ptr(plan)}
            mod['_partition_attention'](Ptr(q),Ptr(k),Ptr(v),Ptr(table),Ptr(lens),Ptr(cu),Ptr(sc) if scales else None,
                Ptr(partial),Ptr(lse),table.shape[1],1,d,d,d,d,d,d,1,0,b,1,groups,d,.125*np.log2(np.e),
                w,qmax,m,n,p,bb,Ptr(origins) if desc else None,Desc(k,n),Desc(v,n),origins.shape[1],phys,**kw)
        rows=(1,2,4)[i%3]
        for row,tile in itertools.product(range(b),range(m//rows)):
            support['coords']=(row,0,tile)
            mod['_merge_partitions'](Ptr(partial),Ptr(lse),Ptr(lens),Ptr(cu),Ptr(sv) if scales else None,Ptr(out),
                1,0,b,1,groups,d,w,m,n,p,bb,16,rows,**kw)
        assert np.isfinite(out).all()
        return out,partial,lse
    first=run(old);second=run(new,outplan)
    for a,z in zip(first,second):
        # Includes unchanged sentinel outputs in empty intersections.
        assert np.array_equal(a.view('u4'),z.view('u4'))
    stats['projection_pairs']+=1;stats['output_components']+=q.size
    stats['scratch_words_including_guards']+=first[1].size+first[2].size
    stats['tensor_descriptor_pairs' if desc else 'indexed_pairs']+=1
print(json.dumps(dict(production_sha256=sha256(current.encode()).hexdigest(),v247_source=sha256(prior.encode()).hexdigest(),
    stats=stats,physical_H200=False,arithmetic='Checked host pointers, bitwise FP32 accumulator/output comparison; BF16 cast emulation uses nearest even.'),indent=2))
