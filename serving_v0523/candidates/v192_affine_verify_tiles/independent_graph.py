"""Check the real registration/consumer boundary and adversarial physical-table replay lifetimes."""
import ast
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


def method(src, name):
    tree=ast.parse(src)
    n=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)
    return ast.get_source_segment(src,n)


def fields(src):
    c=next(x for x in ast.parse(src).body if getattr(x,"name",None)=="FlashAttentionMetadata")
    return [x.target.id for x in c.body if isinstance(x,ast.AnnAssign)]


def check(root):
    path="python/sglang/srt/layers/attention/flashattention_backend.py"
    prod=(root/path).read_text();parent=subprocess.check_output(["git","-C",root,"show",":"+path],text=True)
    old=fields(parent); assert fields(prod)==old+["verify_tile_origins"]
    changes=prod.split('self.target_verify_metadata["verify_tile_origins"] = torch.full(',1)
    assert len(changes)==2 and 'not self.is_draft_runner' in changes[0][-700:]
    assert 'not self.kv_index_translator.reads_are_translated' in changes[0][-700:]
    assert 'metadata.verify_tile_origins = self.target_verify_metadata["verify_tile_origins"][:bs]' in prod
    assert 'page_table is metadata.page_table' in prod and 'window_size[0] < 0' in prod
    assert 'normal_decode_set_metadata(' in method(prod,"_set_decode_page_metadata") and '"origins": metadata.verify_tile_origins' in method(prod,"_set_decode_page_metadata")
    assert 'self._set_decode_page_metadata(' in method(prod,"_apply_cuda_graph_metadata")
    kernel=(root/"python/sglang/kernels/ops/attention/metadata.py").read_text()
    attention=(root/"python/sglang/kernels/ops/attention/paged_verify_attention.py").read_text()
    assert 'col_start >= num_live_pages' in kernel and '(tiles + 1) * 128 <= num_live_pages' in kernel
    assert 'lo = tl.min(candidates' in kernel and 'hi = tl.max(candidates' in kernel
    assert 'origins <= CAPACITY - TILE_N' in attention and 'torch.compiler.is_compiling()' in attention
    rng=random.Random(124987); checked=direct=rejected=0; cap=16384
    for block in (128,256,512):
        # Unwritten positions deliberately preserve stale *valid* origins from earlier replay.
        for rows in (384,513,1024):
            shared=[rng.randrange(1,cap-128) for _ in range((rows+127)//128)]
            for step in range(80):
                live=rng.choice((1,127,128,129,255,256,257,rows-1,rows))
                physical=[]
                for tile in range(len(shared)):
                    origin=rng.choice((0,1,cap-128,cap-127,rng.randrange(1,cap-128)))
                    piece=list(range(origin,origin+128))
                    if rng.randrange(3)==0:
                        # Endpoints continue to agree in these malformed middle rows.
                        at=rng.randrange(1,127);piece[at],piece[at+1]=piece[at+1],piece[at]
                    physical.extend(piece)
                for begin in range(0,live,block):
                    for tile in range(begin//128,min((begin+block)//128,len(shared))):
                        values=physical[128*tile:128*(tile+1)]; spans=[v-i for i,v in enumerate(values)]
                        shared[tile]=spans[0] if spans[0]>0 and min(spans)==max(spans) and 128*(tile+1)<=live else -1
                visible=(live+127)//128
                for lo in range(visible):
                    for hi in range(lo+1,visible+1):
                        certified=all(0<o<=cap-128 for o in shared[lo:hi]);checked+=1
                        if certified:
                            assert 128*hi<=live
                            assert [o+i for o in shared[lo:hi] for i in range(128)]==physical[lo*128:hi*128]
                            direct+=1
                        else:rejected+=1
    return dict(backend=hashlib.sha256(prod.encode()).hexdigest(),metadata=hashlib.sha256(kernel.encode()).hexdigest(),
                attention=hashlib.sha256(attention.encode()).hexdigest(),retained_constructor_fields=len(old),
                graph_replays=720,intersections=checked,direct=direct,fallback=rejected,
                scope="Production registration/source checks and independent CPU table oracle. No physical accelerator or CUDA Graph replay.")

if __name__=="__main__": print(json.dumps(check(Path(sys.argv[1])),indent=2))
