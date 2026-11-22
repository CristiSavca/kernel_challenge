"""Pair the ordinary RadixAttention and FA3 entry path against the pinned baseline on CPU."""

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import torch


PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
ROOT = Path(__file__).resolve().parents[2]
BACK = "srt/layers/attention/flashattention_backend.py"
KERNEL = "kernels/ops/attention/verify_fp8_cache.py"
MEM = "srt/mem_cache/memory_pool.py"
SWA = "srt/mem_cache/swa_memory_pool.py"
DTYPE = torch.float8_e4m3fn


def pinned(name):
    return subprocess.check_output(["git", "-C", str(ROOT / "sglang"), "show",
                                   f"{PIN}:python/sglang/{name}"]).decode()


def compile_defs(text, names, ns, owner=None):
    tree = ast.parse(text)
    body = tree.body if owner is None else next(n.body for n in tree.body if getattr(n, 'name', '') == owner)
    selected = [n for n in body if getattr(n, "name", None) in names]
    assert set(n.name for n in selected) == set(names)
    exec("from __future__ import annotations\n" + "\n".join(ast.unparse(n) for n in selected), ns)
    return {n.name: ns[n.name] for n in selected}


class Run:
    def __init__(self, stage, new):
        self.new = new
        self.log = []
        self.source = (stage / "python/sglang" / BACK).read_text() if new else pinned(BACK)
        self.kern = (stage / "python/sglang" / KERNEL).read_text()
        self.ns = ns = dict(torch=torch, dataclass=dataclass, debug_kernel_api=lambda f: f, is_cp_active=lambda fb: False,
                            AttentionType=NS(ENCODER_ONLY="enc", DECODER_BIDIRECTIONAL="bidi"),
                            is_in_full_prefill_graph=lambda: False, is_in_breakable_cuda_graph=lambda: False,
                            padded_extend_real_tokens=lambda *a: None, get_attn_backend=lambda: self.backend,
                            flash_attn_with_kvcache=self.fa, _is_cuda=True, _is_hip=False, _is_cpu=False,
                            can_use_store_cache=lambda *a: True, store_cache=self.store,
                            maybe_detect_oob=lambda *a: None,
                            MLATokenToKVPool=type("MLA", (), {}))
        pool = pinned(MEM)
        compile_defs(pool, {"KVWriteLoc", "unwrap_write_loc", "_set_kv_buffer_impl"}, ns)
        names = {"get_kv_buffer", "get_key_buffer", "_get_key_buffer", "get_value_buffer",
                 "_get_value_buffer", "_store_kv_layer", "set_kv_buffer"}
        ns["MHATokenToKVPool"] = type("MHA", (), dict(
            compile_defs(pool, names, ns, "MHATokenToKVPool"),
            **compile_defs(pool, {"_check_physical_write_loc"}, ns, "KVCache")))
        ns["SWAKVPool"] = type("SWA", (), compile_defs(
            pinned(SWA), {"get_kv_buffer", "_wait_for_layer", "set_kv_buffer"}, ns, "SWAKVPool"))
        compile_defs(pinned("srt/mem_cache/layout/paged_view.py"), {"paged_view", "paged_kv_view"}, ns)
        triton = NS(cdiv=lambda n,d: (n+d-1)//d, next_power_of_2=lambda n: 1 << (n-1).bit_length())
        runtime = dict(torch=torch, triton=triton, _encode_and_scatter=self)
        mod = ModuleType("sglang.kernels.ops.attention.verify_fp8_cache")
        mod.__dict__.update(compile_defs(self.kern, {"bf16_e4m3_encodings", "encode_verify_qkv"}, runtime))
        sys.modules[mod.__name__] = mod
        self.lut = mod.bf16_e4m3_encodings("cpu")
        smod = ModuleType("sglang.kernels.ops.kvcache.kvcache")
        smod.can_use_store_cache = ns["can_use_store_cache"]
        sys.modules[smod.__name__] = smod
        methods = {"forward_extend", "_mxfp8_sf_kwargs", "get_paged_mha_kv_cache", "prepare_paged_mha_query"}
        if new:
            methods.add("_encode_verify_qkv")
        backend = compile_defs(self.source, methods, ns, "FlashAttentionBackend")
        backend.update(compile_defs(pinned("srt/layers/attention/base_attn_backend.py"),
                                    {"forward"}, ns, "AttentionBackend"))
        self.backend = type("Backend", (), backend)()
        self.radix = type("Radix", (), compile_defs(pinned("srt/layers/radix_attention.py"),
                                                    {"forward"}, ns, "RadixAttention"))

    def __getitem__(self, grid):
        def emulate(q, k, v, out, kc, vc, loc, table, *args, **meta):
            *pitches, capacity = args
            assert pitches == [x.stride(0) for x in (q, k, v, kc, vc, loc)]
            assert (grid[0], meta["Q_WIDTH"], meta["KV_WIDTH"]) == (len(q), q.shape[1], k[0].numel())
            self.log.append("fused")
            out.copy_(table[q.to(torch.int32)])
            for i, slot in enumerate(loc.tolist()):
                assert 0 <= slot < capacity
                if slot:
                    kc[slot].copy_(table[k[i].to(torch.int32)])
                    vc[slot].copy_(table[v[i].to(torch.int32)])
        return emulate

    def store(self, k, v, kc, vc, loc, *, row_bytes, v_row_bytes, size_limit):
        self.log.append("stock_store")
        assert k.shape[1] == row_bytes and v.shape[1] == v_row_bytes
        for i, slot in enumerate(loc.tolist()):
            assert 0 <= slot < size_limit
            if slot:
                kc[slot].copy_(k[i])
                vc[slot].copy_(v[i])

    def fa(self, **kw):
        self.last_fa = kw
        self.log.append("fa")
        # Distinct finite BF16 output driven by the exact bytes actually handed to FA.
        # This recorder does not emulate FA3 floating-point attention.
        q = kw["q"].view(torch.uint8).to(torch.float32)
        term = sum(kw[x].view(torch.uint8).to(torch.float32).sum() % 5 for x in ("k_cache","v_cache"))
        return (q+term).to(torch.bfloat16)

    def pools(self, page, layout, sub, qheads, kheads, dim, seed):
        n = max(16, (sub + page - 1) // page) * page
        typ = self.ns["MHATokenToKVPool"]
        def child(seed_delta):
            owner = typ()
            owner.__dict__.update(size=n-page, page_size=page, is_quantized_kv_cache=False,
                                  kv_cache_layout="nhd", start_layer=0, store_dtype=torch.uint8,
                                  dtype=DTYPE, row_dim=kheads*dim, v_row_dim=kheads*dim,
                                  layer_transfer_counter=None, use_hnd=False, requires_physical_write_loc=False,
                                  device_module=None, alt_stream=None)
            # One extra allocation page models the VA reserve beyond finalized serving capacity.
            storage=[]
            for pad in (1, 2):
                raw = torch.randint(0, 256, (n + page, kheads + pad, dim), dtype=torch.uint8,
                                    generator=torch.Generator().manual_seed(seed + seed_delta + pad))
                storage.append(raw)
            owner.buffers = storage
            owner.k_buffer, owner.v_buffer = [[s[:, :kheads]] for s in storage]
            return owner
        full = child(1)
        if layout == "plain":
            return full, [full]
        sliding = child(17)
        hybrid = self.ns["SWAKVPool"]()
        hybrid.__dict__.update(layers_mapping={0:(0,False), 1:(0,True)},full_kv_pool=full,
                               swa_kv_pool=sliding, layer_transfer_counter=None, start_layer=0)
        return hybrid, [full, sliding]

    def case(self, rows, page, layout, scenario, dtype, seed):
        self.log.clear()
        d, qheads, heads = 64, 16, 2
        pool, owners = self.pools(page, layout, rows+40, qheads, heads, d, seed)
        ix = int(layout == "sliding")
        layer = self.radix()
        layer.__dict__.update(tp_q_head_num=qheads, tp_k_head_num=heads, tp_v_head_num=heads,
            head_dim=d, qk_head_dim=d, v_head_dim=d, layer_id=ix, is_cross_attention=False,
            attn_type="decoder", sliding_window_size=2047 if ix else -1,
            use_prefill_attention_wrapper=False, scaling=.125, logit_cap=0.,
            k_scale=torch.tensor(1.5) if scenario == "scale" else None,
            v_scale=torch.tensor(2.5) if scenario == "scale" else None)
        generator = torch.Generator().manual_seed(seed)
        raw = torch.randint(-32768,32768,(rows,qheads*d+2*heads*d+qheads*d),dtype=torch.int16,
                            generator=generator).view(torch.bfloat16)
        q, k, v, _ = torch.split(raw, [qheads*d,heads*d,heads*d,qheads*d], dim=-1)
        # Real fused QK norm yields 2D Q/K; V can retain a large QKV token pitch.
        if seed % 2:
            q, k = q.contiguous(), k.contiguous()
        if scenario == "bad_input":
            v = v.to(torch.float32)
        mode = NS(is_idle=lambda:False,is_decode=lambda:False,is_mixed=lambda:False,
                  is_target_verify=lambda:scenario!="prefill")
        loc = torch.arange(rows*2,dtype=dtype)[::2]//2 + 1
        slots = torch.empty(2*rows,dtype=dtype)
        slots[::2] = loc
        slots = slots[::2]
        slots[-1] = 0
        if rows>1: slots[0] = 0
        sloc = (loc.flip([0])+15).to(dtype)
        sloc[-1] = 0
        if rows>1: sloc[0] = 0
        fb = NS(batch_size=1, forward_mode=mode, dcp_kv_mask=torch.tensor([True]) if scenario=="dcp" else None,
                out_cache_loc=slots, out_cache_loc_is_physical=False)
        cuq = torch.tensor([0,rows],dtype=torch.int32)
        physical = torch.tensor([[1, 0]],dtype=torch.int32)
        md = NS(swa_out_cache_loc=torch.cat((sloc,sloc[:1])),image_token_ranges=None,swa_spec_metadata=None,
                max_seq_len_q=rows, page_table=physical, swa_page_table=physical.flip([1]),
                cu_seqlens_q=cuq, cache_seqlens_int32=cuq[1:], cu_seqlens_k=cuq)
        self.backend.__dict__.update(fa_impl_ver=3, _verify_fp8_encodings=self.lut if scenario!="disabled" else None,
            kv_cache_dtype=DTYPE, kv_cache_dtype_str="fp8_e4m3", kv_cache_is_mxfp8=False,
            page_size=page, forward_metadata=md, use_sliding_window_kv_pool=layout!="plain",
            use_mla=False, fa_skip_kv_cache=False, local_attn_builder=None, topk=1, num_splits=1,
            token_to_kv_pool=pool)
        result = layer.forward(q,k,v,fb,save_kv_cache=scenario!="no_store",**(
            {"q_rope":q.clone()} if scenario=="rope" else {}))
        fa = self.last_fa
        assert fa["q"].dtype == fa["k_cache"].dtype == fa["v_cache"].dtype == DTYPE
        metadata = {key: (tuple(x.shape),x.dtype,x.stride(),x.contiguous().view(torch.uint8).numpy().tobytes())
                    if isinstance(x,torch.Tensor) else x for key,x in fa.items()}
        buffers = [s.numpy().tobytes() for owner in owners for s in owner.buffers]
        return list(self.log),metadata,buffers,result.view(torch.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage",type=Path)
    stage = p.parse_args().stage
    stock, new = Run(stage,False),Run(stage,True)
    torch.set_num_threads(1)
    counts=Counter()
    for layout in ("plain","full","sliding"):
        for i,scenario in enumerate(("long","scale","no_store","rope","dcp","bad_input","prefill","disabled")):
            for rows, page, dt in ((8,1,torch.int64),(32,7,torch.int32),(128,160,torch.int64),(256,16,torch.int32)):
                args=(rows,page,layout,scenario,dt,4433+i+rows)
                a,ma,ba,ra=stock.case(*args)
                b,mb,bb,rb=new.case(*args)
                assert a == (["fa"] if scenario=="no_store" else ["stock_store","fa"])
                assert b == (["fused","fa"] if scenario=="long" else a), (args,a,b)
                assert ma==mb and ba==bb and torch.equal(ra,rb), (args,b)
                counts["fused" if scenario=="long" else "baseline_fallback"] += 1
                counts["FA_query_bytes"] += len(ma["q"][-1])
                counts["complete_pool_storage_bytes"] += sum(map(len,ba))
                counts["FA_output_bytes"] += ra.numel()
    print(json.dumps({"passed":True,"entry":"RadixAttention.forward -> AttentionBackend.forward -> FlashAttentionBackend.forward_extend",
        "device":"CPU with original Torch casts, emulated device launch/store and recording FA3",
        "torch":torch.__version__, "kernel_sha256":hashlib.sha256(new.kern.encode()).hexdigest(),
        "backend_sha256":hashlib.sha256(new.source.encode()).hexdigest(),"counts":counts},indent=2))


if __name__ == "__main__":
    main()
