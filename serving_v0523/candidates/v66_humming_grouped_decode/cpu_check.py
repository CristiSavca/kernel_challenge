#!/usr/bin/env python3
"""Execute V66 source Python and Triton bodies through CPU tensor shims."""

import ast
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import numpy as np


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    finite = (bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    return (np.where(finite, rounded, bits) & np.uint32(0xFFFF0000)).view(np.float32)


class Tensor(np.ndarray):
    device = 'cpu'

    def to(self, dtype):
        return tensor(bf16(self) if dtype == 'bf16' else np.asarray(self, dtype=dtype))

    def new_empty(self, size):
        poison = np.nan if self.dtype == np.float32 else -32768
        return tensor(np.full(size, poison, dtype=self.dtype))

    def view(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], type):
            return super().view(dims[0])
        return self.reshape(*dims)

    def numel(self):
        return self.size

    def contiguous(self):
        return tensor(np.ascontiguousarray(self))

    def is_contiguous(self):
        return self.flags.c_contiguous


def tensor(values):
    return np.asarray(values).view(Tensor)


class Pointer:
    def __init__(self, data, offsets=0, dtype=None):
        self.data = data.reshape(-1)
        self.offsets = offsets
        self.dtype = NS(element_ty=dtype or data.dtype)

    def __add__(self, offsets):
        return Pointer(self.data, self.offsets + offsets, self.dtype.element_ty)


class Language:
    constexpr = object()
    float32 = np.float32
    int32 = np.int32
    int64 = np.int64
    pid = 0
    grid = 1

    @classmethod
    def program_id(cls, axis):
        assert axis == 0
        return cls.pid

    @classmethod
    def num_programs(cls, axis):
        assert axis == 0
        return cls.grid

    @staticmethod
    def arange(start, stop):
        return tensor(np.arange(start, stop))

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, masks = np.broadcast_arrays(pointer.offsets, mask)
        assert np.all((0 <= offsets[masks]) & (offsets[masks] < pointer.data.size))
        result = np.full(offsets.shape, other, dtype=pointer.data.dtype)
        result[masks] = pointer.data[offsets[masks]]
        return tensor(result)

    @staticmethod
    def store(pointer, value, mask=True):
        offsets, masks, values = np.broadcast_arrays(pointer.offsets, mask, value)
        assert np.all((0 <= offsets[masks]) & (offsets[masks] < pointer.data.size))
        assert np.unique(offsets[masks]).size == offsets[masks].size
        pointer.data[offsets[masks]] = values[masks]

    @staticmethod
    def histogram(values, bins):
        return tensor(np.bincount(values, minlength=bins))

    @staticmethod
    def cumsum(values, axis):
        return tensor(np.cumsum(values, axis=axis))

    @staticmethod
    def sort(values, descending=False):
        assert not descending
        return tensor(np.sort(values))

    @staticmethod
    def where(cond, yes, no):
        return tensor(np.where(cond, yes, no))

    @staticmethod
    def zeros(shape, dtype):
        return tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def rsqrt(values):
        return tensor(np.float32(1) / np.sqrt(values, dtype=np.float32))

    @staticmethod
    def sum(values, axis):
        return tensor(np.sum(values, axis=axis, dtype=np.float32))

    static_range = range


class Launch:
    def __init__(self, body, bf16_pointers=()):
        self.body = body
        self.bf16_pointers = bf16_pointers

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            opts = {k: v for k, v in kwargs.items() if not k.startswith('num_')}
            args = tuple(Pointer(x, dtype='bf16' if i in self.bf16_pointers else None)
                         if isinstance(x, Tensor) else x for i, x in enumerate(args))
            Language.grid = grid[0]
            for Language.pid in range(grid[0]):
                self.body(*args, **opts)
        return run


def extract(path, names, scope, klass=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if klass:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == klass)
    functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    for node in functions:
        node.decorator_list.clear()
    header = ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(header + functions, type_ignores=[]), str(path), 'exec'), scope)


if len(sys.argv) != 2:
    raise SystemExit('usage: cpu_check.py PATCHED_SGLANG_ROOT')
source = Path(sys.argv[1]) / 'python/sglang'
adapter = source / 'srt/layers/muse_spark/muse_spark_humming_moe.py'
finalizer = source / 'kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
activation = source / 'kernels/jit/csrc/muse_spark/expert_scaled_silu.cuh'
torch = NS(
    empty=lambda shape, dtype, device: tensor(np.empty(shape, dtype=dtype)),
    empty_like=lambda value, dtype=None, **kw: tensor(np.empty_like(value, dtype=dtype)),
    bfloat16=np.dtype(np.float32),
    int32=np.int32,
    int64=np.int64,
    compiler=NS(is_compiling=lambda: False),
)
scope = dict(torch=torch, triton=NS(next_power_of_2=lambda n: 1 << (n - 1).bit_length()), tl=Language,
             _GROUPED_COMPUTE='{"gemm_type":"grouped_contiguous","use_f16_accum":false}',
             _MAX_DECODE_SLOTS=2048, _MIN_TOKENS_FOR_FUSED=32)
extract(adapter, ('_plan_grouped_decode','_gather_grouped_decode','humming_decode_eligible','humming_grouped_decode_outputs'), scope)
extract(finalizer, ('_post_expert_norm_weighted_sum_kernel','fused_post_expert_norm_weighted_sum'), scope)
for name in ('_plan_grouped_decode', '_gather_grouped_decode', '_post_expert_norm_weighted_sum_kernel'):
    scope[name] = Launch(scope[name], (0, 3) if name.startswith('_post_expert') else ())

def silu_scaled(inputs, output, keys, offsets, scales, span):
    for row in range(offsets[0], inputs.shape[0]):
        expert = keys[row] // span - 1
        left, right = np.split(bf16(inputs[row] * scales[expert]), 2)
        output[row] = bf16(left / (np.float32(1) + np.exp(-left)) * right)


scope["_expert_scaled_silu_module"] = lambda: NS(run=silu_scaled)
src = activation.read_text()
for fragment in (
    'mul.rn.f32', 'cast<float>(cast<bf16_t>(product))',
    'params.offsets[0] + blockIdx.x', 'params.row_keys[row] / params.slot_span - 1',
    'cast<bf16_t>(apply_activation_f32<ActivationKind::kSiLU>(gate_f32) * up_f32)',
):
    assert fragment in src
assert 'extra_cuda_cflags=get_activation_cuda_cflags()' in adapter.read_text()


class Humming:
    @staticmethod
    def forward_layer(layer, *, inputs, outputs, expert_layout, compute_config, tuning_config, sublayer_name):
        assert not np.shares_memory(inputs, outputs)
        assert inputs.flags.c_contiguous and outputs.flags.c_contiguous
        assert expert_layout.dtype == np.int32 and expert_layout.size == layer.num_local_experts + 1
        assert 0 <= expert_layout[0] <= inputs.shape[0] == outputs.shape[0] == expert_layout[-1]
        assert np.all(np.diff(expert_layout) >= 0)
        if sublayer_name == 'w13':
            assert np.isnan(inputs[:expert_layout[0]]).all()
        assert np.isfinite(inputs[expert_layout[0]:]).all()
        assert json.loads(compute_config) == {'gemm_type': 'grouped_contiguous', 'use_f16_accum': False}
        assert json.loads(tuning_config) == {'use_stream_k': False, 'use_tma_c': False}
        layer.launches.append(sublayer_name)
        for expert, (first, last) in enumerate(zip(expert_layout, expert_layout[1:])):
            outputs[first:last] = gemm_bf16(inputs[first:last], getattr(layer, sublayer_name)[expert])


def gemm_bf16(inputs, weight):
    # Keep CPU BLAS tiling from changing the simulated BF16 GEMM boundary.
    return bf16(inputs.astype(np.float64) @ weight.astype(np.float64).T)


vendor = ModuleType('humming.layer')
vendor.HummingMethod = Humming
sys.modules['humming.layer'] = vendor

rng = np.random.default_rng(66)
routing_cases = kernel_cases = values_compared = 0
for experts, slots in itertools.product((1, 7, 16, 31, 32, 64), (1, 4, 7, 32, 65, 255, 512, 513, 1023, 2048)):
    for pattern in ('remote', 'first', 'last', 'mixed', 'full'):
        raw = np.full(slots, -1, dtype=np.int32)
        if pattern in ('first', 'last'):
            raw[::3] = 0 if pattern == 'first' else experts - 1
        if pattern in ('mixed', 'full'):
            raw[:] = rng.integers(0, experts, slots)
            if pattern == 'mixed':
                raw[::3] = -1
        ids = tensor(raw)
        span = scope['triton'].next_power_of_2(slots)
        order = sorted((int(raw[slot]), slot) for slot in range(slots) if raw[slot] >= 0)
        first = slots - len(order)
        expected_map = np.full(slots, -1, dtype=np.int32)
        for row, (_, slot) in enumerate(order, first):
            expected_map[slot] = row
        for has_map in (False, True):
            expert_map = tensor(np.full(8 * experts, -1, dtype=np.int32))
            rank_ids = rng.choice(expert_map.size, size=experts, replace=False)
            expert_map[rank_ids] = np.arange(experts)
            global_ids = tensor(np.where(raw < 0, -1, rank_ids[np.maximum(raw, 0)])) if has_map else ids
            local = ids.new_empty(slots)
            offsets, keys, rowmap = ids.new_empty(experts + 1), ids.new_empty(slots), ids.new_empty(slots)
            scope['_plan_grouped_decode'][(1,)](
                global_ids, expert_map, local, offsets, keys, rowmap, slots, experts,
                expert_map.size, has_map, span, scope['triton'].next_power_of_2(experts + 1),
            )
            assert np.array_equal(local, raw)
            assert offsets[0] == first and offsets[-1] == slots
            assert np.array_equal(np.diff(offsets), np.bincount(raw[raw >= 0], minlength=experts))
            assert np.array_equal(keys[first:] // span - 1, [expert for expert, _ in order])
            assert np.array_equal(keys[first:] % span, [slot for _, slot in order])
            assert np.array_equal(rowmap, expected_map)
            routing_cases += 1

for tokens, topk, experts, hidden, inner in (
    (1, 1, 1, 3, 5), (3, 3, 7, 7, 16), (8, 8, 4, 31, 32),
    (32, 8, 32, 16, 32), (128, 8, 32, 16, 16), (256, 8, 32, 8, 16),
):
    for mode in ('empty', 'mixed', 'skewed', 'dense'):
        ids = rng.integers(0, experts, (tokens, topk), dtype=np.int32)
        if mode == 'empty':
            ids.fill(-1)
        elif mode == 'mixed':
            ids[rng.random(ids.shape) < 0.7] = -1
        elif mode == 'skewed':
            ids[:, 1:] = -1
            ids[:, 0] = experts - 1
        ids = tensor(ids)
        inputs = tensor(bf16(rng.normal(0, 0.5, (tokens, hidden))))
        layer = NS(
            global_num_experts=8 * experts, num_local_experts=experts, launches=[],
            _expert_map=None,
            intermediate_size_per_partition=inner,
            humming_decode_tunings={name: [(tokens * topk, '{"use_stream_k":false,"use_tma_c":false}')]
                                    for name in ('w13','w2')},
            w13=tensor(bf16(rng.normal(0, 0.5, (experts, 2*inner, hidden)))),
            w2=tensor(bf16(rng.normal(0, 0.5, (experts, hidden, inner)))),
            w13_global_scale=tensor(bf16(rng.uniform(0.25, 1.75, experts))),
            w2_global_scale=tensor(bf16(rng.uniform(0.25, 1.75, experts))),
        )
        assert scope['humming_decode_eligible'](layer, inputs, ids)
        values, row_map, local_ids = scope['humming_grouped_decode_outputs'](layer, inputs, ids)
        assert np.array_equal(local_ids, ids)
        assert layer.launches == ['w13', 'w2']
        scores = tensor(rng.random(ids.shape, dtype=np.float32))
        norm_weight = tensor(bf16(rng.uniform(0.75, 1.75, hidden)))
        reference = inputs.new_empty((tokens,topk,hidden))
        for row, slot in np.argwhere(ids >= 0):
            expert = ids[row, slot]
            gate_up = gemm_bf16(inputs[row], layer.w13[expert])
            left, right = np.split(bf16(gate_up * layer.w13_global_scale[expert]), 2)
            act = bf16((left / (np.float32(1) + np.exp(-left))) * right)
            result = bf16(gemm_bf16(act, layer.w2[expert]) * layer.w2_global_scale[expert])
            reference[row, slot] = result
            assert np.array_equal(result, bf16(values[row_map[row,slot]] * layer.w2_global_scale[expert]))
            values_compared += hidden
        for weight_first in (True, False):
            fixed_output = inputs.new_empty(inputs.shape)
            actual = scope['fused_post_expert_norm_weighted_sum'](
                values, scores, norm_weight, weight_first=weight_first,
                slot_expert_ids=ids, slot_row_indices=row_map, out=fixed_output,
                expert_output_scales=layer.w2_global_scale,
            )
            expected = scope['fused_post_expert_norm_weighted_sum'](
                reference, scores, norm_weight, weight_first=weight_first, slot_expert_ids=ids,
            )
            assert actual is fixed_output and np.array_equal(actual, expected)
            if mode == 'empty':
                assert not actual.any()
            kernel_cases += 1

model_path = source / 'srt/models/muse_spark.py'
extract(model_path, ('_forward_tp',), scope, klass='MuseSpark1xMoE')
activation_dispatch_cases = 0
for mode in ('decode', 'verify', 'extend', 'idle', 'mixed', 'draft', None):
    visited = []
    layer = NS(
        finalize_ar_workspace=None,
        _forward_experts=lambda inputs, router_input, ar_workspace, grouped_decode: (
            visited.append(grouped_decode) or inputs
        ),
        _reduce_and_unpool=lambda inputs, already_reduced: inputs,
    )
    batch = None if mode is None else NS(forward_mode=NS(
        is_decode=lambda: mode == 'decode', is_target_verify=lambda: mode == 'verify'))
    inputs = object()
    assert scope['_forward_tp'](layer, inputs, forward_batch=batch) is inputs
    assert visited == [mode in ('decode', 'verify')]
    activation_dispatch_cases += 1

for hidden, inner, slots, alternate_dtype, compile_enabled in itertools.product(
    (4, 8), (4, 8), (0, 1, 2048, 2049), (False, True), (False, True)
):
    torch.compiler.is_compiling = lambda: compile_enabled
    layer = NS(intermediate_size_per_partition=inner, humming_decode_tunings={'w13': [1]})
    x = tensor(np.empty((1, hidden), dtype=np.float32))
    ids = tensor(np.empty(slots, dtype=np.float32 if alternate_dtype else np.int32))
    assert scope['humming_decode_eligible'](layer, x, ids) == (
        hidden <= inner and 0 < slots <= 2048 and not alternate_dtype and not compile_enabled
    )
    activation_dispatch_cases += 1

print(json.dumps(dict(planner_cases=routing_cases, end_to_end_cpu_cases=kernel_cases,
                      rounded_scale_output_values=values_compared,
                      mode_and_geometry_dispatch_cases=activation_dispatch_cases,
                      numpy=np.__version__), indent=2))
