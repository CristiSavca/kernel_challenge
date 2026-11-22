"""Oracle for causal window equivalence in FA3's physical-offset/used-length ABI."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import runpy

import numpy as np


def mask_equivalence():
    counts = Counter()
    for left, prefix, q_len in itertools.product(
        (0, 1, 2, 7, 63, 159, 160, 255, 256, 2047),
        (0, 1, 2, 63, 159, 160, 255, 256, 2046, 2047, 2048, 16383, 74160),
        (0, 1, 2, 7, 63, 159, 160, 255, 512, 753, 16384),
    ):
        full_k = prefix + q_len
        # The first valid raw index is derived by enumerating the union of
        # individual causal windows, independently of the packer's size formula.
        suffix_k = min(full_k, q_len + left) if q_len else 0
        start = full_k - suffix_k
        sample = range(q_len) if q_len <= 256 else sorted({0, 1, 2, q_len//2, q_len-2, q_len-1})
        for i in sample:
            abs_row = prefix + i
            row_length = min(abs_row + 1, left + 1)
            references = (abs_row - row_length + 1, abs_row + 1)
            local_row = i + suffix_k - q_len
            relative = (start + max(0, local_row - left), start + local_row + 1)
            assert references == relative
            counts['causal_window_queries'] += 1
        counts['length_triples'] += 1
    return counts


def physical_oracle(k, v, table, page_size, physical_positions):
    pages = table[physical_positions // page_size]
    assert np.all(pages >= 0)
    ofs = physical_positions % page_size
    return k[pages, ofs], v[pages, ofs]


def dense_attention(query, key, value, lut, kscale, vscale, softcap):
    # The two sides deliberately share only the scalar operator. Their page
    # selections, bounds and loads originate from independent index systems.
    qhead = np.arange(len(query)) // (len(query) // key.shape[1])
    k = lut[key][:, qhead, :] * kscale[qhead][None, :, None]
    v = lut[value][:, qhead, :] * vscale[qhead][None, :, None]
    score = np.einsum('hd,mhd->mh', query, k) / (query.shape[1] ** .5)
    if softcap is not None:
        score = softcap * np.tanh(score / softcap)
    score -= np.max(score, axis=0)
    exp_score = np.exp(score)
    prob = exp_score / np.sum(exp_score, axis=0)
    return np.einsum('mh,mhd->hd', prob, v)


def validate_packed(k, v, table, prefix, qlens, qcap, page_size, left, packed, counts):
    dk, dv, cuq, cuk, used = packed
    b = len(qlens)
    actual_q = np.cumsum([0, *qlens], dtype=np.int64)
    assert list(cuq) == [*actual_q, qcap]
    assert list(cuk[:-1]) == list(actual_q + np.arange(b+1) * (left + 256))
    assert int(cuk[-1]) == len(dk) == len(dv) == qcap + b*(left+256) + 256
    assert int(used[-1]) == 1
    # Unique FP8 values, including negative zero and tiny signed subnormals.
    ids = np.arange(256)
    ex = (ids >> 3) & 15
    man = ids & 7
    lut = np.where(ex == 0, man / 512., (1 + man / 8.) * (2. ** (ex - 7)))
    lut = np.where(ids & 128, -lut, lut)
    lut[(ex == 15) & (man == 7)] = np.nan
    rng = np.random.default_rng(7881)
    for i, (offset, qn) in enumerate(zip(prefix, qlens)):
        full_k = offset + qn
        n = int(used[i])
        lo = max(0, offset-left)
        assert n == (full_k-lo if qn else 0)
        if n:
            ek, ev = physical_oracle(k, v, table[i], page_size, np.arange(lo, full_k))
            assert np.array_equal(dk[cuk[i]:cuk[i]+n], ek)
            assert np.array_equal(dv[cuk[i]:cuk[i]+n], ev)
            counts['copied_fp8_bytes'] += ek.size + ev.size
        zero_hi = min(int(cuk[i+1]), int(cuk[i])+n+256)
        for dense in [dk, dv]:
            assert not np.any(dense[cuk[i]+n:zero_hi])
        for u in sorted({0, min(1, qn-1), qn//2, qn-1}):
            if u < 0 or u >= qn:
                continue
            pos = offset + u
            raw_ids = np.arange(max(0, pos-left), pos+1)
            raw_k, raw_v = physical_oracle(k, v, table[i], page_size, raw_ids)
            dense_q_pos = u + n - qn
            start = int(cuk[i]) + max(0, dense_q_pos - left)
            stop = int(cuk[i]) + dense_q_pos + 1
            heads, dim = k.shape[2:]
            qheads = min(4, 8*heads)
            if qheads < heads:
                qheads = heads
            q = rng.normal(size=(qheads, dim))
            ks = np.linspace(.01*(i+1), .12*(i+1), heads)
            vs = np.linspace(.125*(i+1), .8*(i+1), heads)
            for cap in (None, 2.0):
                a = dense_attention(q, raw_k, raw_v, lut, ks, vs, cap)
                z = dense_attention(q, dk[start:stop], dv[start:stop], lut, ks, vs, cap)
                np.testing.assert_array_equal(a, z)
                assert np.isfinite(z).all()
                counts['reference_attention_scalars'] += z.size
    for dense in [dk, dv]:
        assert not np.any(dense[cuk[-2]:min(cuk[-1], cuk[-2]+256)])
    counts['packed_batches'] += 1


def run_cpu(stage, counts):
    import ast
    import torch
    from types import SimpleNamespace

    here = Path(__file__).resolve().parent
    sim = runpy.run_path(str(here.parent / 'v70_dflash_selector_tree/cpu_check.py'))
    source = stage / 'python/sglang/kernels/ops/attention/swa_prefill_kv.py'
    pid, (kernel,) = sim['program'](source, ['_pack_swa_suffix'])
    cdiv = lambda x, d: (x + d - 1)//d
    current_grid = [0, 0]
    kernel.__globals__['tl'].num_programs = lambda axis: current_grid[axis]

    def ptr(tensor):
        n = tensor.untyped_storage().nbytes() // tensor.element_size()
        raw = tensor.as_strided((n,), (1,), storage_offset=0).numpy()
        return sim['Pointer'](raw, tensor.storage_offset())

    class Launch:
        def __getitem__(self, grid):
            def invoke(*args, **kwargs):
                current_grid[:] = grid
                lane_dtype = args[0].dtype
                assert all(args[i].dtype == lane_dtype for i in (1, 5, 6))
                assert lane_dtype in (torch.int32, torch.uint8)
                path = 'aligned_word' if lane_dtype == torch.int32 else 'unaligned_byte'
                counts[path + '_batches'] += 1
                assert lane_dtype == expected_lane_dtype
                for tensor in args[5:7]:
                    tensor.view(torch.uint8).fill_(127)  # FP8 NaN in every destination byte
                for tensor in args[7:10]:
                    tensor.fill_(-921)
                values = [ptr(t) for t in args[:10]]
                for x, y in itertools.product(range(grid[0]), range(grid[1])):
                    pid[:2] = [sim['array'](x), sim['array'](y)]
                    kernel(*values, *args[10:], **kwargs)
            return invoke

    names = {'pack_swa_prefill', 'can_pack_swa_prefill', 'with_padding_descale'}
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if getattr(n, 'name', '') in names]
    env = dict(torch=torch, triton=SimpleNamespace(cdiv=cdiv), _pack_swa_suffix=Launch())
    exec(compile(tree, str(source), 'exec'), env)
    rng = np.random.default_rng(25019)
    cases = [
        ([0], [512], 0, 0), ([0], [513], 7, 2047),
        ([74160], [753], 273, 2047), ([32768], [16384], 0, 2047),
        ([159, 0, 64, 512], [512, 0, 2, 253], 55, 63),
        ([71993, 0, 160, 2047], [753, 0, 1, 512], 2050, 2047),
        ([0, 11, 2048, 2049], [0, 0, 0, 0], 1024, 2047),
        ([0, 11, 2048, 2049], [0, 0, 0, 0], 0, 2047),
        ([0, 11, 2048, 2049], [1, 511, 0, 513], 0, 159),
    ]
    for case_idx, (prefix, qlens, qpad, left) in enumerate(cases):
        b, qcap = len(prefix), sum(qlens) + qpad
        dim = 64
        heads = 2 if case_idx % 3 else 1
        lens = np.add(prefix, qlens)
        columns = max(1, int(lens.max()))
        n_physical = min(columns, 8053) + 31
        def cache(gap, unaligned):
            shape = (n_physical+2, heads*dim+gap+4) if unaligned else (n_physical+2, 1, heads+gap, dim)
            bits = rng.integers(0, 256, size=shape, dtype=np.uint8)
            # Leave positive/negative zero and subnormals; avoid NaNs in valid rows.
            bits[bits == 127] = 1
            bits[bits == 255] = 129
            base = torch.from_numpy(bits)
            if unaligned:
                # Odd physical row pitch and a non-word offset. Head and dim
                # remain contiguous, exactly as required by the public packer.
                raw = base[:, 1:1+heads*dim].view(n_physical+2, 1, heads, dim)[:-2]
                assert raw.data_ptr() % 4 and raw.storage_offset() % 4 and raw.stride(0) % 4
            else:
                raw = base[1:-1, :, :heads]
                assert raw.data_ptr() % 4 == raw.stride(0) % 4 == 0
            return raw.view(torch.float8_e4m3fn)
        unaligned_k, unaligned_v = case_idx in (1, 4), case_idx in (1, 5)
        k, v = cache(1, unaligned_k), cache(3, unaligned_v)
        expected_lane_dtype = torch.uint8 if (unaligned_k or unaligned_v) else torch.int32
        if heads == 2 and dim == 64:
            layout = 'byte' if expected_lane_dtype == torch.uint8 else 'word'
            counts['muse_2x64_' + layout + '_batches'] += 1
        table = torch.full((b, columns+5), -2**24, dtype=(torch.int64 if case_idx % 2 else torch.int32))[:, 3:-2]
        for r, (pref, q) in enumerate(zip(prefix, qlens)):
            if q:
                lo = max(0, pref-left)
                # Model a genuinely evicted prefix. False reads before lo will
                # trigger the checked pointer rather than returning plausible K/V.
                n = int(lens[r] - lo)
                table[r, lo:int(lens[r])] = torch.tensor(rng.integers(0, n_physical, n))
        lens_t = torch.tensor(lens, dtype=torch.int32)
        cuq_t = torch.tensor(np.cumsum([0, *qlens]), dtype=torch.int32)
        assert env['can_pack_swa_prefill'](k,v,table,lens_t,cuq_t,qcap,left)
        max_q = max(512, max(qlens))
        result = env['pack_swa_prefill'](k,v,table,lens_t,cuq_t,qcap,max_q,left)
        dk, dv, rq, rk, used, max_k = result
        assert max_k >= int(used.max()) and max_k <= columns
        packed = (dk.view(torch.uint8).numpy(), dv.view(torch.uint8).numpy(), rq.numpy(), rk.numpy(), used.numpy())
        validate_packed(k.view(torch.uint8).numpy(),v.view(torch.uint8).numpy(),table.numpy(),prefix,qlens,qcap,1,left,packed,counts)

    assert counts['muse_2x64_byte_batches'] and counts['muse_2x64_word_batches']
    fn = env['with_padding_descale']
    for b, h in itertools.product((1, 2, 7), (1, 2, 4)):
        for scale in [torch.arange(b*h).reshape(b,h).float(), torch.arange(h).float()[None,:].expand(b,h),
                      torch.arange(b*h*2).reshape(b,2*h).float()[:,1::2]]:
            extended = fn(scale)
            assert extended.shape == (b+1,h) and torch.equal(extended[:-1],scale)
            assert torch.isfinite(extended).all()
            counts['descale_layouts'] += 1

    # Ordinary FA3 permits only tensors with unit last stride; the conservative
    # packer guard additionally demands contiguous head and token-row metadata.
    base = lambda: [k, v, table, lens_t, cuq_t, qcap, left]
    for column,value in [(0,k.transpose(2,3)), (1,v[:,:,:,:32]), (2,None),
                         (2,table.T), (3,lens_t.long()), (4,cuq_t[:-1]), (5,32*1024**2), (6,-1)]:
        args = base()
        args[column] = value
        assert not env['can_pack_swa_prefill'](*args)
        counts['fallback_shapes'] += 1
    backend = stage / 'python/sglang/srt/layers/attention/flashattention_backend.py'
    check_backend_call(backend, env, counts)
    return {'source':source.name, 'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest(),
            'torch_cpu':torch.__version__, 'numpy':np.__version__}


def check_backend_call(source, production, counts):
    import ast
    import torch
    from types import SimpleNamespace

    branch, = [node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.If)
               and any(isinstance(n, ast.Attribute) and n.attr == '_pack_swa_fp8_prefill'
                       for n in ast.walk(node.test))]
    condition = compile(ast.Expression(branch.test), str(source), 'eval')
    branch_body = compile(ast.Module(body=branch.body, type_ignores=[]), str(source), 'exec')
    mode = SimpleNamespace(EXTEND='prefill')
    def metadata():
        return SimpleNamespace(forward_mode=mode.EXTEND, dcp_kv_mask=None)
    key = torch.empty((17,1,2,64), dtype=torch.float8_e4m3fn)
    lens = torch.tensor([7312, 291], dtype=torch.int32)
    cu = torch.tensor([0, 512, 524], dtype=torch.int32)
    page = torch.zeros((2,7410), dtype=torch.int32)
    q = torch.empty((560,16,64),dtype=torch.float8_e4m3fn)
    out = torch.empty(q.shape,dtype=torch.bfloat16)
    now = dict(self=SimpleNamespace(_pack_swa_fp8_prefill=True, page_size=1,
               is_prefill_aware_swa=False, kv_index_translator=SimpleNamespace(is_translating=False),
               num_splits=3, flash_attn_varlen_func=lambda **kw:kw),
               forward_batch=metadata(), ForwardMode=mode, is_swa_layer=True, causal=True,
               window_size=(2047,0), max_seqlen_q=512, use_local_attn=False,
               return_lse=False, q=q, torch=torch, kwargs={}, can_pack_swa_prefill=production['can_pack_swa_prefill'],
               key_cache=key, value_cache=key.clone(), page_table=page, cache_seqlens=lens, cu_seqlens_q=cu,
               layer=SimpleNamespace(tp_q_head_num=16, head_dim=64, scaling=.117, logit_cap=13.),
               _fa_out=out, with_padding_descale=production['with_padding_descale'])
    sentinels = [object() for _ in range(5)]
    def capture(*args):
        assert all(a is b for a, b in zip(args[:5], (key, now['value_cache'], page, lens, cu)))
        assert args[5:] == (560, 512, 2047)
        return (*sentinels, 2559)
    now['pack_swa_prefill'] = capture
    assert eval(condition, now)
    for override in ({'is_swa_layer':False}, {'forward_batch':SimpleNamespace(forward_mode='verify',dcp_kv_mask=None)},
                     {'causal':False}, {'window_size':(-1,-1)}, {'window_size':(2047,1)}, {'max_seqlen_q':511},
                     {'use_local_attn':True}, {'return_lse':True}, {'q':q.float()}, {'kwargs':{'sinks':object()}},
                     {'forward_batch':SimpleNamespace(forward_mode=mode.EXTEND,dcp_kv_mask=object())}):
        assert not eval(condition, {**now, **override})
        counts['backend_guard_fallbacks'] += 1
    for field, value in [('_pack_swa_fp8_prefill',False), ('page_size',160), ('is_prefill_aware_swa',True),
                         ('kv_index_translator',SimpleNamespace(is_translating=True))]:
        owner = SimpleNamespace(**{**vars(now['self']),field:value})
        assert not eval(condition, {**now,'self':owner})
        counts['backend_guard_fallbacks'] += 1
    for with_scales in (False, True):
        scales = {'k_descale':torch.tensor([[1.,2.],[3.,4.]]), 'v_descale':torch.tensor([[7.,3.],[5.,4.]])} if with_scales else {}
        env = {**now, 'kwargs':scales}
        assert eval(condition,env)
        exec(branch_body,env)
        args = env['result']
        assert args['q'].data_ptr() == q.data_ptr()
        assert [args[n] for n in ('k','v','cu_seqlens_q','cu_seqlens_k','seqused_k')] == sentinels
        assert args['max_seqlen_q'] == 560 and args['max_seqlen_k'] == 2559
        assert args['window_size'] == (2047,0) and args['causal'] is True and args['out'] is out
        assert args['softmax_scale'] == .117 and args['softcap'] == 13. and args['num_splits'] == 3
        assert 'page_table' not in args and not args.get('return_softmax_lse')
        for n, scale in scales.items():
            assert torch.equal(args[n][:-1],scale) and args[n].shape == (3,2)
        counts['production_backend_abi'] += 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', type=Path)
    a = p.parse_args()
    counts = mask_equivalence()
    source = run_cpu(a.stage, counts)
    print(json.dumps({'scope':'Checked production packer and wrapper on CPU, with an independent physical-page and float64 local-attention oracle. No GPU FA3 execution or wall-time result.',
                      'counts':counts, 'checked_source':source}, indent=2))
