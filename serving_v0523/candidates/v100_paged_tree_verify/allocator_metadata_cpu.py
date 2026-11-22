"""Execute V60's allocator and full/SWA metadata on the integrated V100 source."""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main(source):
    em = runpy.run_path(str(Path(__file__).resolve().parents[1]/"v70_dflash_selector_tree/cpu_check.py"))
    ptr = em["Pointer"]
    # Triton integer division truncates toward zero, including negative sentinels.
    class Signed(em["Array"]):
        def __floordiv__(self, by):
            val=np.asarray(self);return np.where(val>=0,val//by,-((-val)//by)).view(Signed)
    triton=SimpleNamespace(jit=lambda fn=None,**kwargs:(lambda f:f) if fn is None else fn,
                           next_power_of_2=lambda x:1<<(int(x)-1).bit_length())
    def setup(path,names):
        pid,fs=em["program"](source / ("python/sglang/kernels/ops/"+path),names,dependencies={"triton":triton})
        tl=fs[0].__globals__["tl"]
        tl.load=lambda p,mask=True,other=0,**kw:em["load"](p,mask,other).view(Signed)
        tl.store=lambda p,val,mask=True,**kw:em["store"](p,val,mask)
        return pid,fs
    pid,(alloc,)=setup("memory/allocator.py",("alloc_extend_kernel",))
    mid,(pageid,draft,_,decode)=setup("attention/metadata.py",(
        "_draft_page_index","_draft_extend_metadata_kernel","_resolve_metadata_page_ids","_fused_decode_metadata_kernel",
    ))
    rng=np.random.default_rng(100); allocations=drafts=decodes=signed=0
    for page in (1,2,3,16,31,64,127,128,160,192,256):
        shift=page.bit_length()-1 if page&(page-1)==0 else -1
        vals=np.arange(-page*3-2,page*3+3,dtype=np.int32)
        np.testing.assert_equal(pageid(vals.view(Signed),page,shift),vals//page)
        signed+=len(vals)
        for b in (1,3,6):
            bpad=triton.next_power_of_2(b)
            for trial in range(24):
                # Each old partial page stays within its privately owned range;
                # fragmented new pages come from a distinct shuffled free pool.
                prefix=rng.integers(0,5*page+1,b,dtype=np.int64)
                prefix[0]=(0,page-1,page,page+1)[trial%4]
                extra=rng.integers(0,page*3+1,b,dtype=np.int64)
                extra[0]=(0,1,page-1,page,page+1,4095,4096,4097)[trial%8]
                end=prefix+extra
                oldpage=rng.permutation(b)+2
                last=np.where(prefix>0,oldpage*page+(prefix-1)%page,-1)
                npages=(end+page-1)//page-(prefix+page-1)//page
                free=10+b+rng.permutation(max(1,int(npages.sum())))
                out=np.full(int(extra.sum())+4,-999,dtype=np.int64)
                cursor=0;expected=[]
                for row in range(b):
                    pid[0]=row
                    alloc(ptr(prefix),ptr(end),ptr(last),ptr(free),ptr(out,2),bpad,page)
                    first_new=(int(prefix[row])+page-1)//page
                    for token in range(int(prefix[row]),int(end[row])):
                        page_nr=token//page
                        phys=oldpage[row] if page_nr<first_new else free[cursor+page_nr-first_new]
                        expected.append(phys*page+token%page)
                    cursor+=int(npages[row])
                np.testing.assert_equal(out[2:-2],expected)
                assert (out[:2]==-999).all() and (out[-2:]==-999).all()
                allocations+=1
            for epoch in range(6):
                lens=rng.integers(0,8*page+1,b,dtype=np.int32)
                lens[0]=(0,page-1,page,page+1,2*page-1,2*page+1)[epoch]
                starts=(lens+page-1)//page
                cols=int(starts.max())+2
                physical=(1+rng.permutation(b*cols)).reshape(b,cols)
                swa_pages=(1+rng.permutation(b*cols)).reshape(b,cols)
                order=rng.permutation(b)
                table=(physical[order,:,None]*page+np.arange(page)).reshape(b,-1)
                mapping=np.zeros(((b*cols+1)*page,),dtype=np.int64)
                for row in range(b):
                    for col in range(cols):
                        st=physical[row,col]*page
                        mapping[st:st+page]=swa_pages[row,col]*page+np.arange(page)
                req=np.argsort(order);tokens_per_req=4
                hl=np.full(b,-1,np.int32);ck=np.full(b+1,-1,np.int32);cq=ck.copy()
                ext=np.minimum(lens,tokens_per_req).astype(np.int32)
                writes=np.stack([table[req[i],:tokens_per_req] if table.shape[1]>=tokens_per_req
                                 else np.resize(table[req[i]],tokens_per_req) for i in range(b)])
                for swa in (False,True):
                    full=np.full((b,cols),-23,np.int32);short=full.copy();write_out=np.full_like(writes,-8)
                    for row in range(b):
                        for tile in range((cols+3)//4):
                            mid[:]=[row,tile,0]
                            draft(ptr(lens),1,ptr(ext),1,ptr(table),table.shape[1],1,ptr(req),1,
                                  ptr(hl),1,ptr(ck),1,ptr(cq),1,ptr(full),cols,1,ptr(mapping),
                                  ptr(short),ptr(writes),ptr(write_out),b,cols,tokens_per_req,page,shift,4,swa,4)
                    np.testing.assert_equal(hl,lens);np.testing.assert_equal(ck,np.r_[0,np.cumsum(lens)])
                    np.testing.assert_equal(cq,np.r_[0,np.cumsum(ext)])
                    for row,n in enumerate(starts):
                        np.testing.assert_equal(full[row,:n],physical[row,:n]);assert (full[row,n:]==-23).all()
                        if swa: np.testing.assert_equal(short[row,:n],swa_pages[row,:n])
                        assert (short[row,n:]==-23).all()
                    if swa:np.testing.assert_equal(write_out,mapping[writes])
                    drafts+=1
                    for delta in (0,8):
                        for zero_tail in (False,True):
                            full.fill(-23);short.fill(-23)
                            for row in range(b):
                                for tile in range((cols+3)//4):
                                    mid[:]=[row,tile,0]
                                    decode(ptr(lens),1,ptr(table),table.shape[1],1,ptr(req),1,
                                        ptr(hl),1,ptr(ck),1,ptr(full),cols,1,ptr(short) if swa else None,cols,1,
                                        ptr(mapping) if swa else None,1,None,0,None,0,b,cols,page,delta,4,False,zero_tail)
                            np.testing.assert_equal(hl,lens+delta)
                            np.testing.assert_equal(ck,np.r_[0,np.cumsum(lens+delta)])
                            expected_counts=np.minimum((lens+delta+page-1)//page,cols)
                            for row,n in enumerate(expected_counts):
                                np.testing.assert_equal(full[row,:n],physical[row,:n]);assert (full[row,n:]==(0 if zero_tail else -23)).all()
                                if swa:
                                    np.testing.assert_equal(short[row,:n],swa_pages[row,:n]);assert (short[row,n:]==(0 if zero_tail else -23)).all()
                            decodes+=1
    return dict(device="CPU masked Triton-source emulator",page_sizes=[1,2,3,16,31,64,127,128,160,192,256],
        allocation_cases=allocations,draft_extend_cases=drafts,decode_verify_metadata_cases=decodes,
        signed_page_index_values=signed,coverage="Fragmented page allocation, row tails, 4096-lane boundaries, shuffled request slots, SWA remaps, padded metadata and signed sentinels.")


if __name__=="__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
