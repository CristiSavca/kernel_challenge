"""Exercise the normal Muse -> Radix -> paged/FA3 verify dispatch on finite BF16 fixtures."""
import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS

import torch

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
BASE = "208b625ae43f820d9f06f36e97c2b1e00484d42c"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load("check178", HERE.parent / "v178_direct_fp8_prompt_kv/check.py")
contracts = load("check136", HERE.parent / "v136_fused_verify_qk_rope_cache/dispatch_check.py")
shared.V64 = BASE
SRT = "python/sglang/srt/"
MODEL = SRT + "models/muse_spark.py"
RADIX = SRT + "layers/radix_attention.py"
NORM = "python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py"
torch.set_num_threads(1)


def define(path, cls, name, env, old=False):
    shared.add(env, path, [name], cls=cls, old=old)
    return env[name]


class Launch:
    """Model each device instruction boundary in the enclosing production launcher."""
    def __init__(self, stats):
        self.stats = stats

    def __getitem__(self, grid):
        def launch(qkv, q, k, pos, trig, kc, vc, slots, eps, pitch, logical_rows,
                   bound, h, g, d, bd, block, **cfg):
            assert pitch == qkv.stride(0) and bd == d
            self.stats["qk_launches"] += 1
            raw_q, raw_k, raw_v = torch.split(qkv[:, :(h + 2*g)*d], [h*d, g*d, g*d], 1)
            def norm(x):
                t = x.reshape(len(qkv), -1, d).float()
                rounded = (t * torch.rsqrt(t.square().mean(-1, keepdim=True) + eps)).bfloat16()
                if cfg["HAS_ROPE"]:
                    cc, ss = trig[pos.long()].unsqueeze(1).chunk(2, -1)
                    a, b = rounded.float()[..., 0::2], rounded.float()[..., 1::2]
                    rounded = torch.stack((a*cc - b*ss, a*ss + b*cc), -1).flatten(-2).bfloat16()
                return rounded.reshape(len(qkv), -1)
            qn, kn = norm(raw_q), norm(raw_k)
            table = cfg.get("ENCODINGS")
            if table is None:
                assert not cfg["HAS_KV_STORE"] and k.dtype == torch.bfloat16
                q.copy_(qn)
                k.copy_(kn)
                return
            assert cfg["HAS_KV_STORE"] and cfg["FP8_KV_STORE"] and k is qkv
            assert logical_rows == len(qkv) * (h + 2*g)
            assert torch.all((slots >= 0) & (slots < bound))
            live = slots != 0
            def code(x):
                return table[x.view(torch.int16).int() & 0xffff]
            q.copy_(code(qn))
            kc[slots[live].long()] = code(kn[live])
            vc[slots[live].long()] = code(raw_v[live])
            self.stats["integrated_stores"] += 1
        return launch


def one(old, *, raw, positions, trig, caches, tables, gated, rotate, owner, policy, accept, stats):
    n = len(raw)
    h, g, d = 16, 2, 64
    kscale = torch.tensor([.75]) if policy == "scaled" else None
    vscale = torch.tensor([.625]) if policy == "scaled" else None
    window = 2047 if owner else -1
    full, swa = tables
    b,pool,_,writes = shared.mk(old, *caches, raw, full, swa, window, kscale, vscale)
    env = b.forward_extend.__func__.__globals__
    # The shared synthetic pool writes bytes on CPU. Match the native padding-slot contract.
    cls = type(pool.full_kv_pool)
    old_write = cls._store_kv_layer
    def store(self, layer_idx, slots, k, v):
        valid = slots != 0
        return old_write(self, layer_idx, slots[valid], k.view(torch.uint8)[valid], v.view(torch.uint8)[valid])
    cls._store_kv_layer = store
    if old:
        shared.add(env, shared.HELP, ["supports_direct_prompt_pool","prepare_direct_prompt_kv","complete_direct_prompt_kv"], old=False)
    cu = torch.tensor([0,n], dtype=torch.int32)
    lens = torch.tensor([23+n], dtype=torch.int32)
    full_slots, swa_slots = [x[0,-n:].long().clone() for x in tables]
    # Native graph padding is unaddressable as an owner value. Check the whole bank after both executions.
    full_slots[0] = swa_slots[0] = 0
    b.forward_metadata = NS(cache_seqlens_int32=lens, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seq_len_q=n,
                            page_table=full, swa_page_table=swa, swa_spec_metadata=None,
                            image_token_ranges=None, swa_out_cache_loc=swa_slots)
    b._hopper_verify_sms = 132
    b._disable_scheduler_metadata_precompute = False
    mode = NS(is_target_verify=lambda: True, is_extend=lambda: False)
    fb = NS(forward_mode=mode, batch_size=1, dcp_kv_mask=None, out_cache_loc=full_slots,
            spec_algorithm=NS(is_dflash=lambda:True), spec_info=NS(ragged_verify_layout=None),
            out_cache_loc_is_physical=False)
    enc = define(NORM, None, "bf16_e4m3_encodings", {"torch":torch}, False)("cpu")
    b._muse_verify_encodings = None if policy == "no_table" else enc
    for item in [pool.full_kv_pool, pool.swa_kv_pool]:
        item.row_dim = item.v_row_dim = g*d
        item._muse_verify_capacity = torch.tensor(item.size+1)
    nk = Launch(stats)
    nenv = dict(torch=torch, VerifyFP8Store=contracts.VerifyFP8Store, triton=NS(
        cdiv=lambda a,b:(a+b-1)//b, next_power_of_2=lambda a:1<<(a-1).bit_length()),
        _qk_rms_norm_fwd_kernel=nk, _triton_config=lambda rows:(4,4), is_arch_support_pdl=lambda:False)
    native_launch = define(NORM, "FusedQKNorm", "_forward_triton", nenv, old)
    radix_env = dict(is_in_breakable_cuda_graph=lambda:False, is_in_full_prefill_graph=lambda:False,
                     is_in_prefill_graph_scope=lambda:False, padded_extend_real_tokens=lambda *a:None, get_attn_backend=lambda:b)
    radix = define(RADIX, "RadixAttention", "forward", radix_env, True)
    norm_cls = type("FusedQKNorm", (contracts.FusedQKNorm,), {"forward":native_launch})
    layer_cls = type("RadixAttention", (contracts.RadixAttention,), {"forward":radix})
    norm, layer = norm_cls(), layer_cls(0)
    norm._eps = 1e-6
    layer.__dict__.update(sliding_window_size=window, k_scale=kscale, v_scale=vscale,
        tp_q_head_num=h, qk_head_dim=d, scaling=0.125, logit_cap=0., attn_type="decoder", use_prefill_attention_wrapper=True)
    guard_env = dict(torch=torch, BaseFusedOp=contracts.BaseFusedOp, FusedQKNorm=norm_cls, VerifyFP8Store=contracts.VerifyFP8Store,
            FlashAttentionBackend=type(b), RadixAttention=layer_cls, MHATokenToKVPool=cls, SWAKVPool=type(pool),
            KVWriteLoc=env["KVWriteLoc"], KernelBackend=NS(TRITON="triton"), can_use_store_cache=lambda a,b:True,
            is_cp_active=lambda _:False, is_in_prefill_graph_scope=lambda:False,
            envs=NS(SGLANG_ENABLE_ASYNC_ASSERT=NS(get=lambda:False)),get_fused_op_backend=lambda:None)
    if not old:
        prepare = contracts.load_function(STAGE/shared.BACK, "muse_verify_fp8_store", guard_env)
        b.muse_verify_fp8_store = MethodType(prepare,b)
    fused = not old and policy == "eligible"
    stage_out = {}
    def attention(q,k,v,l,fb_,save,**kw):
        assert save == (not fused) and (k is None) == fused and l is layer
        stage_out["q_before"] = q
        return b.forward_extend(q,k,v,l,fb_,save,**kw)
    b.forward = attention
    expected_table = tables[owner]
    original_pool = (pool.full_kv_pool,pool.swa_kv_pool)[owner].data
    def probe(q, k, v, pages, context, scale, ks, vs):
        assert q.is_contiguous() and q.dtype == torch.float8_e4m3fn and q.shape==(n,h,d)
        assert q.data_ptr() == stage_out["q_before"].data_ptr() if fused else q.data_ptr() != stage_out["q_before"].data_ptr()
        assert scale == .125 and pages is expected_table and context is lens
        assert k.data_ptr() == original_pool[0].data_ptr() and v.data_ptr() == original_pool[1].data_ptr()
        if policy=="scaled":
            assert torch.all(ks==kscale) and torch.all(vs==vscale)
        else:
            assert ks is None and vs is None
        return q.bfloat16()
    trace = []
    def paged(q,k,v,pages,lengths,cu_,**kw):
        trace.append("paged")
        output = probe(q,k,v,pages,lengths,kw["softmax_scale"],kw["k_descale"],kw["v_descale"])
        assert cu_ is cu and kw["max_q"] == n and kw["window_left"] == window
        return output if accept else None
    def native(**kw):
        trace.append("FA3")
        return probe(kw["q"], kw["k_cache"], kw["v_cache"], kw["page_table"], kw["cache_seqlens"],
                     kw["softmax_scale"],kw.get("k_descale"),kw.get("v_descale"))
    env.update(paged_verify_attention=paged,flash_attn_with_kvcache=native)
    env_model = dict(torch=torch,get_memory=lambda:NS(enable_unified_memory=False),get_attn_backend=lambda:b,
            get_token_to_kv_pool=lambda:pool, TRTLLMHAAttnBackend=type("Other",(),{}))
    first = define(MODEL,"MuseSpark1xAttention","_fused_kv_store",env_model,old)
    forward = define(MODEL,"MuseSpark1xAttention","forward",env_model,old)
    model=NS(qkv_proj=lambda _: (raw.clone(),None),q_size=h*d,kv_size=g*d,use_o_gate=gated,
          _fused_qk_norm=norm, rotary_emb=NS(cos_sin_cache=trig) if rotate else None, attn=layer,
          o_proj=lambda a:(a,None))
    model._fused_kv_store = MethodType(first,model)
    def output_gate(a, gate):
        assert (gate is not None) == gated
        if gate is not None: assert torch.equal(gate, raw[:,-h*d:])
        return a if gate is None else a + gate.reshape_as(a)
    model._fused_o_norm_gate = output_gate
    output = forward(model,positions,raw,fb)
    assert trace == (["paged"] if accept else ["paged","FA3"])
    assert len(writes) == int(not fused)
    stats["ordinary_stores"] += len(writes)
    stats["paged_calls"] += 1
    stats["native_FA3_calls"] += int(not accept)
    return output, [*pool.full_kv_pool.data,*pool.swa_kv_pool.data]


def main():
    counts = Counter()
    torch.manual_seed(182)
    for n,gated,rotate,owner,policy,accept in itertools.product((8,16),(False,True),(False,True),(0,1),
                                                             ("eligible","scaled","no_table"),(False,True)):
        cap = 60
        tables = [(torch.randperm(cap-1)[:n+23]+1).int()[None] for _ in range(2)]
        caches = [tuple(torch.randn(cap,2,64).to(torch.float8_e4m3fn) for _ in range(2)) for _ in range(2)]
        raw = torch.randn(n, (16+4+16*gated)*64).bfloat16()
        defaults = dict(raw=raw, positions=torch.randperm(2*n)[:n], trig=torch.randn(2*n,64),
                        caches=caches, tables=tables, gated=gated, rotate=rotate, owner=owner, policy=policy,
                        accept=accept, stats=counts)
        old, stock_cache = one(True, **defaults)
        new, current_cache = one(False, **defaults)
        assert torch.equal(old,new)
        for x,y in zip(stock_cache,current_cache,strict=True):
            assert torch.equal(x.view(torch.uint8),y.view(torch.uint8))
        counts["matched_pairs"] += 1
        counts["matched_BF16_output_values"] += new.numel()
        counts["matched_physical_cache_bytes"] += sum(x.numel() for x in current_cache)
    return dict(counts=counts, reference="V180", torch=torch.__version__,
                device_substitutes="RMS/RoPE/FP8, scatter and final attention dispatch use CPU fixtures.",
                hardware=False)


if __name__ == "__main__":
    print(json.dumps(main(),indent=2))
