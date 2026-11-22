"""Replay V204's backend admission and V311's public launch with instrumented device tensors."""

import ast
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import runpy
import sys
from types import SimpleNamespace as NS


HERE = Path(__file__).resolve().parent
fixture = runpy.run_path(str(next(HERE.parent.glob("v301_*/independent_runtime.py"))))
KERNEL = fixture["KERNEL"]
BACKEND = fixture["BACKEND"]
class Tensor(fixture["Tensor"]):
    def contiguous(self):
        return self

    def view(self, *dims):
        total = math.prod(self.shape)
        known = math.prod(n for n in dims if n != -1)
        shape = tuple(total // known if n == -1 else n for n in dims)
        assert math.prod(shape) == total
        return self if shape == self.shape else Tensor(shape, self.device, self.dtype)


def backend_methods(source):
    cls = next(n for n in ast.parse(source).body if getattr(n, "name", None) == "FlashAttentionBackend")
    methods = [f for f in cls.body if getattr(f, "name", None) in {"forward_extend","_mxfp8_sf_kwargs"}]
    assert len(methods) == 2
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    return compile(module, BACKEND, "exec")
cdiv = fixture["ceil"]


def parent_kernel():
    published = next(HERE.parent.glob("v204_*/patch.diff")).read_bytes()
    receipt = json.loads(next(HERE.parent.glob("v204_*/validation.json")).read_bytes())
    assert sha256(published).hexdigest() == receipt["patch_sha256"]
    section = published.split(f"+++ b/{KERNEL}\n".encode(), 1)[1].split(b"diff --git ", 1)[0]
    header, *lines = section.splitlines(keepends=True)
    assert header.startswith(b"@@ -0,0") and all(line.startswith(b"+") for line in lines)
    return b"".join(line[1:] for line in lines)


def backend_dispatch(backend):
    # Execute the admission guard, real call with metadata, and its complete native fallback.
    for node in ast.walk(ast.parse(backend)):
        if not isinstance(node, ast.If):
            continue
        for block in (node.body, node.orelse):
            for index, child in enumerate(block):
                if isinstance(child, ast.If) and any(
                    isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "paged_verify_attention"
                    for n in child.body[:1] for n in ast.walk(n)
                ) and "_hopper_verify_sms" in ast.unparse(child.test):
                    tail = block[index - 1:index + 2]
                    assert len(tail) == 3 and ast.unparse(tail[0]) == "result = None"
                    assert ast.unparse(tail[-1].test) == "result is None"
                    return compile(ast.Module(tail, []), BACKEND, "exec")
    raise AssertionError("missing verification dispatch")


def call(env, code, b, m, g, h, page, cols, win, sms, mode="static", full_backend=None):
    q = Tensor((b * m, g * h, 64), dtype="bf16" if mode == "wrong_dtype" else "fp8")
    bank = Tensor((16384, page, h, 64), dtype="fp8")
    p = Tensor((b, cols)); l = Tensor((b,)); cu = Tensor((b+1,))
    cert = Tensor((b, cdiv(cols, 128))) if mode != "missing_origin" else None
    dest = Tensor(q.shape, dtype="bf16")
    state = dict(self=NS(_hopper_verify_sms=sms, topk=1, _disable_scheduler_metadata_precompute=False,
                         kv_index_translator=NS(is_translating=False), num_splits=1, fa_impl_ver=3),
                 forward_batch=NS(forward_mode=NS(is_target_verify=lambda:True), spec_algorithm=NS(is_dflash=lambda:True),
                                  spec_info=NS(ragged_verify_layout=object() if mode=="ragged_layout" else None), dcp_kv_mask=None),
                 layer=NS(scaling=.125, logit_cap=2 if mode=="softcap" else 0),
                 page_table=p, metadata=NS(page_table=p, verify_tile_origins=cert),
                 query=q, key_cache=bank, value_cache=bank, cache_seqlens=l, cu_seqlens_q=cu, cu_seqlens_k=None,
                 max_seqlen_q=m, window_size=(win,0), kwargs={}, fa_k_descale=None, fa_v_descale=None, _fa_out=dest,
                 use_local_attn=False, use_cascade_attn=False, direct_prompt_kv=None, causal=True, return_lse=False)
    native = []
    def fallback(**kw):
        native.append(kw)
        return kw["out"]
    state.update(paged_verify_attention=env["paged_verify_attention"], flash_attn_with_kvcache=fallback)
    p_calls, m_calls = env["_partition_attention"].calls, env["_merge_partitions"].calls
    p_calls.clear(); m_calls.clear()
    if full_backend is None:
        exec(code, state)
        assert state["result"] is dest
    else:
        md = state["metadata"]
        md.swa_page_table = Tensor((b, cols))
        md.swa_spec_metadata = md.image_token_ranges = None
        md.cu_seqlens_q, md.cu_seqlens_k, md.max_seq_len_q, md.cache_seqlens_int32 = cu, None, m, l
        fb = state["forward_batch"]
        fb.batch_size, fb._attn_output = b, dest
        layer = state["layer"]
        layer.sliding_window_size, layer.tp_q_head_num, layer.tp_k_head_num = win, g*h, h
        layer.head_dim = layer.v_head_dim = 64
        layer.is_cross_attention, layer.attn_type = False, "decoder"
        state.update(is_cp_active=lambda _:False, AttentionType=NS(ENCODER_ONLY="enc", DECODER_BIDIRECTIONAL="bi"))
        exec(full_backend,state)
        op = state["self"]
        op.forward_metadata, op.use_sliding_window_kv_pool = md, True
        op.use_mla = op.fa_skip_kv_cache = op.kv_cache_is_mxfp8 = False
        op.local_attn_builder = None
        op.prepare_paged_mha_query = lambda q,qr,kr,*args,**kwargs: (q,qr,kr,None,None)
        op.get_paged_mha_kv_cache = lambda layer: (bank,bank)
        op._mxfp8_sf_kwargs = lambda layer,fb,descale: state["_mxfp8_sf_kwargs"](op,layer,fb,descale)
        out = state["forward_extend"](op,q,None,None,layer,fb)
        assert out.shape == (b*m,g*h*64) and out.dtype == "bf16"
        p = md.page_table if win < 0 else md.swa_page_table
    if native:
        assert len(native)==1 and not p_calls and not m_calls and native[0]["page_table"] is p
        return None
    assert len(p_calls)==len(m_calls)==1
    (pg, a, kw), (mg, z, _) = p_calls[0], m_calls[0]
    assert a[3] is p and a[4] is l and a[5] is cu
    rows = z[-1] if len(z)==19 else 1
    assert len(z) in (18,19) and mg==(b,h,cdiv(m*g,rows))
    assert pg==(pg[0],h) and a[7].shape==(pg[0]+b-1,h,1<<(m*g-1).bit_length(),64)
    assert a[8].shape==a[7].shape[:-1] and z[0] is a[7] and z[1] is a[8]
    assert z[5] is dest and z[15]==pg[0] and a[-2]==pg[0]
    return dict(partitions=pg[0], one_tile=False, slots=a[7].shape[0], partition_ctas=math.prod(pg),
                rows=rows, merge_ctas=math.prod(mg), merge_grid=mg[2], scratch_bytes=4*(math.prod(a[7].shape)+math.prod(a[8].shape)),
                direct="Origins" in kw)


def main(stage):
    old, new = parent_kernel(), (stage/KERNEL).read_bytes()
    backend=(stage/BACKEND).read_bytes()
    envs = [fixture["load"](src) for src in (old,new)]
    gate = backend_dispatch(backend)
    full_backend = backend_methods(backend)
    rng = random.Random(311)
    count = Counter()
    for _ in range(3000):
        b=rng.choice([1,2,3,4,7,16,32,64])
        m=rng.randrange(2,17); g=rng.choice([n for n in range(1,33) if 16<=m*n<=64])
        h=rng.choice([1,2,4,8]); page=rng.choice([1,2,16,64])
        cols=rng.choice([1,2,17,33,64,127,128,129,511,2051,8192,131088])
        w=rng.choice([-1,0,1,60,64,127,2047]); tn=128 if w<0 else 64
        sms=rng.choice([20,80,132])
        before,after=[call(e,gate,b,m,g,h,page,cols,w,sms) for e in envs]
        assert before["direct"]==after["direct"]==(page==1 and w<0)
        assert after["partition_ctas"]<=before["partition_ctas"] and after["scratch_bytes"]<=before["scratch_bytes"]
        assert after["merge_ctas"]<=before["merge_ctas"] and (after["rows"]==1 or after["merge_ctas"]>=sms)
        lengths=[rng.randrange(cols*page+1) for _ in range(b)]
        queries=[rng.randint(0,min(m,l)) for l in lengths]
        geometries=[[e["_tile_geometry"](l,q,w,tn) for l,q in zip(lengths,queries)] for e in envs]
        assert geometries[0]==geometries[1]
        ns=[n for _,n in geometries[0]]
        left,right=[fixture["segments"](ver,ns) for ver in (before,after)]
        assert right==left and sum(end-begin for _,_,begin,end in right)==sum(ns)
        assert all(0<=slot<after["slots"] for slot, *_ in right)
        for q in set(queries):
            row_count=after["rows"]
            owners=[]
            for tile in range(after["merge_grid"]):
                first=tile*row_count
                if first<q*g:
                    n=min(row_count,q*g-first) if g%row_count else row_count
                    owners.extend(range(first,first+n))
            assert owners==list(range(q*g))
            count["merged_query_length_checks"]+=1
        count["mid_tile_bound_witnesses"]+=bool(w>=0 and any(n>cdiv(w+m,tn) for n in ns))
        count["identical_partial_intersections"]+=len(left)
        count["empty_queries"]+=queries.count(0)
        count["dispatch_pairs"]+=1
    assert count["mid_tile_bound_witnesses"]>0
    table={}
    for b in (1,4,16,32):
        exposure={}
        saved_partition=saved_merge=0
        for w,layers in ((-1,16),(2047,46)):
            a,c=[call(e,gate,b,8,8,2,1,131088,w,132,full_backend=full_backend) for e in envs]
            count["full_forward_extend_pairs"]+=1
            exposure["full" if w<0 else "sliding"]=dict(parent=a,candidate=c,scratch_bytes_saved_per_call=a["scratch_bytes"]-c["scratch_bytes"])
            saved_partition += layers*(a["partition_ctas"]-c["partition_ctas"])
            saved_merge += layers*(a["merge_ctas"]-c["merge_ctas"])
            for mode in ("ragged_layout","softcap","wrong_dtype","missing_origin"):
                result=[call(e,gate,b,8,8,2,1,131088,w,132,mode,full_backend=full_backend) for e in envs]
                count["full_forward_extend_pairs"]+=1
                if mode=="missing_origin":
                    assert all(r and not r["direct"] for r in result)
                else:
                    assert result==[None,None]
                count["fallback_or_indexed_pairs"]+=1
        exposure["partition_ctas_saved_per_62_layers"]=saved_partition
        exposure["merge_ctas_saved_per_62_layers"]=saved_merge
        exposure["two_kernels_per_layer_before_and_after"]=True
        table[b]=exposure
    print(json.dumps(dict(sources_sha256={key:sha256(data).hexdigest() for key,data in [("v204",old),("v311",new),("backend",backend)]},
        counts=count,verify_block_8_per_load=table,
        scope="Full shipped forward_extend for published shapes and fallbacks, plus guarded branch across randomized geometry. Public host launch and scratch allocation run on modeled CUDA tensors. "
              "Padded layouts include per-request query lengths from zero through max_q. Explicit ragged-layout metadata continues through native FA. "
              "Public work counts assume complete eight-token verification blocks, two KV heads and 132 SMs. No physical latency or allocated peak is claimed."),indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))
