"""Exercise the real V106 packer against token-address and FA3 offset oracles."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V105 = runpy.run_path(str(HERE.parent / 'v105_swa_suffix_prefill/independent_swa_oracle.py'))
SIM = runpy.run_path(str(HERE.parent / 'v70_dflash_selector_tree/cpu_check.py'))


def snippets(nodes, path, scope):
    unit = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), scope)


def table_builders(source):
    asts, = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name=='FlashAttentionBackend']
    init, = [n for n in ast.walk(asts) if isinstance(n, ast.FunctionDef) and n.name == 'init_forward_metadata']
    graph, = [n for n in ast.walk(asts) if isinstance(n, ast.FunctionDef) and n.name == '_init_full_cg_prefill_metadata']
    translate, = [n for n in ast.walk(init) if isinstance(n, ast.Assign) and
                  ast.unparse(n.targets[0]) == 'metadata.swa_page_table' and
                  'translate_loc_from_full_to_swa(metadata.page_table)' in ast.unparse(n.value)]
    convert, = [n for n in ast.walk(init) if isinstance(n, ast.If) and
                ast.unparse(n.test) == 'self.page_size > 1 and (not _unified_read)']
    cg, = [n for n in ast.walk(graph) if isinstance(n, ast.If) and ast.unparse(n.test) == 'max_seq_len_k > 0'
           and 'full_cg_prefill_strided_indices' in ast.unparse(n)]
    return lambda scope:snippets([translate, convert],source,scope), lambda scope:snippets([cg],source,scope)


def check_table_builders(backend, counts):
    eager, graph = table_builders(backend)
    rng = np.random.default_rng(611219)
    for page, length, b in itertools.product([16,128,160,256,320], [1,159,160,161,319,320,321,4097], [1,3]):
        n = (length + page - 1)//page
        number = n*b
        full = rng.permutation(number).reshape(b,n) + 3
        swa = rng.permutation(number).reshape(b,n) + 3
        remap = torch.zeros((number+5)*page, dtype=torch.int64)
        offsets = np.arange(page)
        raw = (full[:,:,None]*page + offsets).reshape(b, -1)[:,:length]
        for f, s in zip(full.ravel(),swa.ravel()):
            remap[f*page:(f+1)*page] = torch.arange(s*page,(s+1)*page)
        links = torch.from_numpy(np.array(raw))
        ids = torch.arange(b).flip(0)
        owner = SimpleNamespace(page_size=page,device='cpu',use_sliding_window_kv_pool=True,topk=1,
                                req_to_token=links,full_cg_prefill_strided_indices=torch.arange(0,length,page),
                                token_to_kv_pool=SimpleNamespace(translate_loc_from_full_to_swa=lambda slots:remap[slots]))
        batch = SimpleNamespace(req_pool_indices=ids)
        meta = SimpleNamespace(page_table=links[ids],swa_page_table=None)
        env = dict(torch=torch,self=owner,metadata=meta,_unified_read=False,forward_batch=batch)
        eager(env)
        assert torch.equal(meta.page_table,torch.tensor(full)[ids])
        assert torch.equal(meta.swa_page_table,torch.tensor(swa,dtype=torch.int32)[ids])
        m = SimpleNamespace(page_table=torch.full((b,n+2),-671, dtype=torch.int32),swa_page_table=torch.full((b,n+2),-894,dtype=torch.int32))
        graph({**env,'m':m,'max_seq_len_k':length,'bs':b})
        assert torch.equal(m.page_table[:,:n],meta.page_table)
        assert torch.equal(m.swa_page_table[:,:n],meta.swa_page_table)
        assert (m.swa_page_table[:,n:]==-894).all()
        counts['eager_and_graph_page_tables'] += 1


def launch_api(module, counts):
    pid,(kernel,) = SIM['program'](module, ['_pack_swa_suffix'])
    grid_dims = [0,0]
    kernel.__globals__['tl'].uint32 = np.uint32
    kernel.__globals__['tl'].num_programs = lambda axis:grid_dims[axis]
    lane = []
    def ptr(t):
        n = t.untyped_storage().nbytes()//t.element_size()
        return SIM['Pointer'](t.as_strided((n,), (1,),storage_offset=0).numpy(), t.storage_offset())
    class Launch:
        def __getitem__(self,grid):
            def invoke(*args,**kwargs):
                assert all(args[i].dtype==args[0].dtype for i in (1,5,6))
                lane[:] = [args[0].dtype]
                for tensor in args[5:7]:
                    tensor.view(torch.uint8).fill_(127)
                for tensor in args[7:10]:
                    tensor.fill_(-78757)
                pointers=[ptr(t) for t in args[:10]]
                grid_dims[:]=grid
                for x,y in itertools.product(range(grid[0]),range(grid[1])):
                    pid[:2]=[SIM['array'](x),SIM['array'](y)]
                    kernel(*pointers,*args[10:],**kwargs)
            return invoke
    scope=dict(torch=torch,triton=SimpleNamespace(cdiv=lambda x,d:(x+d-1)//d),_pack_swa_suffix=Launch())
    tree=ast.parse(module.read_text())
    snippets([n for n in tree.body if getattr(n,'name','') in ('pack_swa_prefill','can_pack_swa_prefill')],module,scope)
    return scope['pack_swa_prefill'], scope['can_pack_swa_prefill'], lane


def physical_storage(rng, pages, p, heads, dim, misaligned, gap):
    # Exercise independent page, token and head strides with extra source
    # storage between physical pages. Tile arithmetic cannot flatten pages.
    bump=1 if misaligned else 4
    pitch=heads*dim+gap*bump
    stride_page=p*pitch+3*gap*bump
    offset=1 if misaligned else 4
    bits=rng.integers(0,256,size=pages*stride_page+offset,dtype=np.uint8)
    bits[bits==127]=1
    bits[bits==255]=129
    t=torch.from_numpy(bits).as_strided((pages,p,heads,dim),(stride_page,pitch,dim,1),offset).view(torch.float8_e4m3fn)
    assert t.stride(0)!=p*t.stride(1) and ((t.data_ptr()%4)!=0)==misaligned
    return t


def run(stage):
    root=stage/'python/sglang'
    module=root/'kernels/ops/attention/swa_prefill_kv.py'
    backend=root/'srt/layers/attention/flashattention_backend.py'
    counts=Counter()
    check_table_builders(backend,counts)
    pack,eligible,lanes=launch_api(module,counts)
    rng=np.random.default_rng(110917)
    cases=[]
    for p in (1,16,128,160,256,320):
        cases.extend([
            (p,[2047+p-1,2047+p,2047+p+1,0],[513,512,511,0],7,2047),
            (p,[p-1,p,p+1,3*p-1],[1,2,p+1,512],31,63),
            (p,[0,1,2000],[0,0,0],512,2047),
        ])
    cases.extend([(160,[74160],[753],0,2047), (160,[32749],[16320],64,2047),
                  (160,[2037,0,80003],[0,3,753],2000,2047)])
    for case,(p,prefix,qlens,padding,left) in enumerate(cases):
        b=len(prefix)
        lens=np.add(prefix,qlens)
        qcap=sum(qlens)+padding
        cols=max(1,(int(lens.max())+p-1)//p)
        nphys=min(cols+13,430)
        table=torch.full((b,cols+4),-400000,dtype=(torch.int64 if case%2 else torch.int32))[:,1:cols+1]
        for r,(pr,qn) in enumerate(zip(prefix,qlens)):
            if not qn:continue
            first=max(0,pr-left)//p
            last=(pr+qn+p-1)//p
            if last>first:
                # Repeated as well as shuffled physical pages, irrespective of
                # their logical neighbors. Earlier SWA pages stay poisoned.
                table[r,first:last]=torch.tensor(rng.integers(0,nphys,size=last-first))
        k=physical_storage(rng,nphys,p,2,64,case%3==1,1)
        v=physical_storage(rng,nphys,p,2,64,case%3==2,3)
        cu=torch.tensor(np.cumsum([0,*qlens]),dtype=torch.int32)
        lengths=torch.tensor(lens,dtype=torch.int32)
        maxq=max(512,max(qlens))
        assert eligible(k,v,table,lengths,cu,qcap,left)
        dk,dv,cq,ck,used,maxk=pack(k,v,table,lengths,cu,qcap,maxq,left)
        assert maxk==min(cols*p,maxq+left) and maxk>=int(used.max())
        assert lanes==[torch.uint8 if case%3 else torch.int32]
        counts['byte_copy_batches' if case%3 else 'aligned_word_copy_batches']+=1
        if p==160:
            counts['native_page160_batches']+=1
        for pr,qn in zip(prefix,qlens):
            if qn:
                off=max(0,pr-left)
                counts['suffix_begins_inside_page']+=int(off%p>0)
                counts['last_visible_row_precedes_page_end']+=int((pr+qn)%p>0)
        V105['validate_packed'](k.view(torch.uint8).numpy(),v.view(torch.uint8).numpy(),
              table.numpy(),prefix,qlens,qcap,p,left,(dk.view(torch.uint8).numpy(),dv.view(torch.uint8).numpy(),
              cq.numpy(),ck.numpy(),used.numpy()),counts)
    for pitches in (
        (k.stride(0), k.stride(1), 1, 1),
        (k.stride(0), k.stride(1), k.stride(2), 0),
        (k.stride(0), k.shape[2] * k.shape[3] - 1, k.stride(2), 1),
        (k.stride(0) // 2, k.stride(1), k.stride(2), 1),
    ):
        bad = k.as_strided(k.shape, pitches)
        assert not eligible(bad, v, table, lengths, cu, qcap, left)
        counts['overlapping_or_wrong_head_stride_fallbacks'] += 1
    return {'scope':'Executed frozen native Python pack/guard and Triton body with checked CPU pointers and true tensor storage strides. Exercised pinned eager and capture-time SWA table builders. Float64 math oracle, no hardware performance result.',
            'source_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
            'backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest(),
            'widths':sorted({case[0] for case in cases}),'torch':torch.__version__,'counts':counts}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    print(json.dumps(run(p.parse_args().stage),indent=2))
