#!/usr/bin/env python3
"""Independent abstract WS pipeline liveness model; no CUDA or performance proof."""
from dataclasses import dataclass, field
from pathlib import Path
import json, random

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class CTA:
    a0:int; a1:int; b0:int; b1:int; ns:int; nr:int; tail:int
    q:int=0; hq:int=0; epi:bool=False; tails_done:bool=False
    weights:dict=field(default_factory=dict)
    partial:dict=field(default_factory=dict)
    issued:set=field(default_factory=set)
    @property
    def na(self): return self.a1-self.a0
    @property
    def n(self): return self.na+self.b1-self.b0
    @property
    def nh(self): return self.n-min(self.tail,self.b1-self.b0)
    def issue(self,q):
        assert q<self.n and q not in self.issued
        s=q%self.ns
        assert s not in self.weights, ("overwrite unread weight slot",q,s,self.weights)
        self.weights[s]=(q,False);self.issued.add(q)


def run(jobs,grid,tail,seed,ns=2,nr=4,drop_credit=False):
    rng=random.Random(seed)
    blocks=[CTA(jobs*512*c//grid,jobs*512*(c+1)//grid,
                jobs*256*c//grid,jobs*256*(c+1)//grid,ns,nr,tail)
            for c in range(grid)]
    credits=[0]*jobs
    for b in blocks:
        assert b.na>0 or b.n==0
        # FAST1 issues min(2,nA); the helper fills the rest of the same NS ring.
        pre=min(2,b.na)
        assert pre<=ns
        for q in range(pre):b.issue(q)
        for q in range(pre,min(ns,b.n)):b.issue(q)
    events=0
    while True:
        ready=[]
        unfinished=False
        for i,b in enumerate(blocks):
            if b.tails_done and b.hq==b.nh and not b.epi:continue
            unfinished=True
            for s,(q,done) in b.weights.items():
                if not done:ready.append((i,"copy",s))
            if b.q<b.n:
                q=b.q
                allowed=q<b.na or credits[(b.b0+q-b.na)//256]==512
                if allowed and b.weights.get(q%ns)==(q,True) and q%nr not in b.partial:
                    ready.append((i,"compute",0))
            elif not b.tails_done:
                ready.append((i,"tail",0))
            if b.epi:ready.append((i,"epilogue",0))
            elif b.hq<b.nh and b.partial.get(b.hq%nr)==b.hq:
                ready.append((i,"helper_read",0))
        if not unfinished:
            assert credits==[512]*jobs
            for b in blocks:
                assert not b.weights and not b.partial
                assert b.issued==set(range(b.n))
            return dict(events=events,units=jobs*768,ctas=grid,seed=seed,tail=tail,jobs=jobs)
        if not ready:
            raise RuntimeError("deadlock")
        i,action,s=rng.choice(ready);b=blocks[i];events+=1
        if action=="copy":
            q,done=b.weights[s];assert not done;b.weights[s]=(q,True)
        elif action=="compute":
            q=b.q
            # All consumer warps read weights and publish their partial before CTA_SYNC.
            assert b.weights.pop(q%ns)==(q,True)
            assert q%nr not in b.partial;b.partial[q%nr]=q
            if q+ns<b.n:b.issue(q+ns)
            b.q+=1
        elif action=="helper_read":
            # Freeing partial storage precedes epilogue, but reduction is now in registers.
            assert b.partial.pop(b.hq%nr)==b.hq;b.epi=True
        elif action=="epilogue":
            assert b.epi
            if b.hq==b.na-1:
                for j in range(b.a0//512,(b.a1-1)//512+1):
                    n=min(b.a1,(j+1)*512)-max(b.a0,j*512)
                    if not(drop_credit and i==0):credits[j]+=n
                    assert credits[j]<=512
            b.epi=False;b.hq+=1
        elif action=="tail":
            # All compute warps have crossed final CTA_SYNC; final tails occupy distinct slots.
            for q in range(b.nh,b.n):assert b.partial.pop(q%nr)==q
            b.tails_done=True


def main():
    rows=[]
    for jobs,grid in ((0,264),(1,264),(2,528),(8,132),(44,264)):
        for tail in (1,2):
            rows.append(run(jobs,grid,tail,13+jobs+tail))
    # Small grids stress many ring wraps; large grids stress empty/short B ranges.
    for seed in range(12):
        rows.append(run(1,7,1+seed%2,100+seed))
    negative=[]
    for kw in (dict(jobs=1,grid=7,tail=2,seed=123,nr=1),
               dict(jobs=1,grid=264,tail=2,seed=456,drop_credit=True)):
        try:run(**kw)
        except RuntimeError as e:
            assert str(e)=="deadlock";negative.append(kw)
        else:raise AssertionError("negative control unexpectedly completed")
    result=dict(decision="PASS abstract lifecycle model",cases=rows,
        event_count=sum(r["events"] for r in rows),negative_controls=negative,
        assumptions=["All CTAs resident, route/ranges initialized, initial PDL wait complete.",
          "One compute event represents all compute warps completing reads/partial stores and CTA_SYNC.",
          "TMA completion and helper read/epilogue are independently interleaved.",
          "Helper credit occurs only after its last A epilogue, with original release/acquire visibility."],
        limits="Independent model of intended NS2/NRB4 lifecycle, not extracted CUDA, numerical arithmetic, compiler or hardware proof.")
    out=ROOT/"experiments/iterations/20261104-small-moe-ring-critic.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))

if __name__=="__main__":main()
