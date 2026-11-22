"""Run V81's Triton staging body with the existing checked-array interpreter."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

stage = Path(sys.argv[1]).resolve()
fixture = Path(__file__).resolve().parents[1] / 'v66_humming_grouped_decode/cpu_check.py'
scope = {}
exec(compile(fixture.read_text().split('if len(sys.argv) != 2:')[0], str(fixture), 'exec'), scope)
tensor, Pointer, bf16 = (scope[name] for name in ('tensor', 'Pointer', 'bf16'))


class Language(scope['Language']):
    minimum = staticmethod(np.minimum)
    sum = staticmethod(lambda x, axis: tensor(np.sum(x, axis, dtype=x.dtype)))
    max = staticmethod(lambda x, axis: tensor(np.max(x, axis)))
    gather = staticmethod(lambda x, indices, axis: tensor(np.take(x, indices, axis=axis)))


adapter = stage / 'python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py'
kernels = {'tl': Language}
scope['extract'](adapter, ('_stage_token_routes',), kernels)
rng = np.random.default_rng(81001)
power2 = lambda n: 1 << (n - 1).bit_length()
cases = checked_values = 0
small_cases = itertools.product(
    (1, 3, 16, 32), (1, 3, 8), (1, 8, 32), (13, 128), (False, True),
    ('random', 'remote', 'repeated', 'concentrated'),
)
target_cases = itertools.product((8, 32, 128, 256), (8,), (32,), (4096,),
                                 (True,), ('random', 'remote', 'repeated'))
for m, k, experts, hidden, mapped, mode in itertools.chain(small_cases, target_cases):
    slots = m * k
    n_experts = experts * 4 if mapped else experts
    ids = rng.integers(-2, n_experts + 2, size=slots, dtype=np.int64)
    mapping = np.full(n_experts, -1, np.int32)
    owned = rng.choice(n_experts, experts, replace=False)
    mapping[owned] = np.arange(experts)
    if mode == 'remote':
        ids[:] = -1
        ids[1::2] = 2**43
    elif mode == 'repeated':
        ids[:] = (owned if mapped else np.arange(experts))[0]
    elif mode == 'concentrated':
        ids[:] = rng.choice(owned if mapped else np.arange(experts), m).repeat(k)
    mask = (ids >= 0) & (ids < n_experts)
    expected_id = np.full(slots, -1, np.int32)
    expected_id[mask] = mapping[ids[mask]] if mapped else ids[mask]

    x = bf16(rng.normal(size=(m, hidden)).astype(np.float32))
    x[::5] = 0
    weight = bf16(rng.normal(size=hidden).astype(np.float32))
    eps = np.float32(1e-6)
    ref = bf16((x * (1 / np.sqrt(np.mean(x*x, axis=1, keepdims=True)+eps))) * weight)
    local, offsets, keys, rows = [
        np.full(n, -777, np.int32) for n in (slots, experts + 1, slots, slots)
    ]
    output = np.full((slots, hidden), np.nan, np.float32)
    args = [Pointer(tensor(a)) for a in (ids, mapping, x, weight, local, offsets, keys, rows)]
    args += [Pointer(tensor(output), dtype='bf16')]
    for token in rng.permutation(m):
        Language.pid = token
        kernels['_stage_token_routes'](*args, eps, slots, n_experts, experts, mapped,
                                      k, power2(k), hidden, power2(slots),
                                      power2(experts+1), power2(hidden))
    live = np.flatnonzero(expected_id >= 0)
    ordered = live[np.argsort(expected_id[live], kind='stable')]
    first = slots - len(live)
    row_map = np.full(slots, -1, np.int32)
    row_map[ordered] = np.arange(first, slots)
    np.testing.assert_array_equal(rows, row_map)
    np.testing.assert_array_equal(local, expected_id)
    np.testing.assert_array_equal(keys[first:], (expected_id[ordered]+1)*power2(slots)+ordered)
    np.testing.assert_array_equal(np.diff(offsets), np.bincount(expected_id[live], minlength=experts))
    assert offsets[0] == first and offsets[-1] == slots
    assert np.isnan(output[:first]).all()
    np.testing.assert_array_equal(output[first:], ref[ordered // k])
    cases += 1
    checked_values += len(live) * hidden


# Exercise production eligibility, including custom module hooks and cross-device weights.
from types import ModuleType, SimpleNamespace as NS

module_name = 'sglang.srt.layers.muse_spark.muse_spark_layers'
module = ModuleType(module_name)


class Norm:
    hidden_size = 13
    _forward_hooks = {}
    _forward_pre_hooks = {}


module.MuseSpark1xRMSNorm = Norm
sys.modules[module_name] = module
mod_hooks = NS(_global_forward_hooks={}, _global_forward_pre_hooks={})
flag = NS(get=lambda: True)
provider = NS(SGLANG_ENABLE_MUSE_SPARK_HUMMING_ROUTED_NORM=flag,
              SGLANG_ENABLE_MUSE_SPARK_HUMMING_FUSED_STAGE=flag)
runtime = {'torch': NS(bfloat16='bf16', float32='fp32', nn=NS(modules=NS(module=mod_hooks))),
           'envs': provider}
scope['extract'](adapter, ('routed_norm_weight',), runtime)
norm = Norm()
weight = NS(shape=(13,), dtype='bf16', device='cuda:0', is_contiguous=lambda: True)
norm._get_effective_weight = lambda: weight
hidden = NS(shape=(4, 13), device='cuda:0')
ids = NS(shape=(4, 8))
select = runtime['routed_norm_weight']
assert select(norm, hidden, ids) is weight
for owner, key, invalid in (
    (norm, '_forward_hooks', {0: object()}), (norm, '_forward_pre_hooks', {1: object()}),
    (mod_hooks, '_global_forward_hooks', {0: object()}),
    (mod_hooks, '_global_forward_pre_hooks', {0: object()}),
    (norm, '_get_effective_weight', lambda: None), (norm, 'hidden_size', 12),
    (hidden, 'shape', (5, 13)), (ids, 'shape', (4, 9)), (weight, 'shape', (1, 13)),
    (weight, 'device', 'cuda:1'), (weight, 'dtype', 'fp16'),
    (weight, 'is_contiguous', lambda: False), (flag, 'get', lambda: False),
):
    old = getattr(owner, key)
    setattr(owner, key, invalid)
    assert select(norm, hidden, ids) is None
    setattr(owner, key, old)
class SpecialNorm(Norm):
    pass
assert select(SpecialNorm(), hidden, ids) is None
assert select(None, hidden, ids) is None
print(json.dumps({'source_kernel_cases': cases, 'bf16_output_values': checked_values,
                  'source_guard_cases': 16}, indent=2))
