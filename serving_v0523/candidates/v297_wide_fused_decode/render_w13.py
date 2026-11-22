"""Render native and four-band fused W13 through the production Humming generator."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

here = Path(__file__).resolve().parent
driver = here.parent / 'v169_grouped_expert_tile_cohorts/vendor_codegen.py'
exec(compile(driver.read_text().split('sp=importlib.util.spec_from_file_location', 1)[0], str(driver), 'exec'))
layers = stage / 'python/sglang/srt/layers/muse_spark'
for name in ('decode_mma_pair', 'residual_prompt_mma', 'live_decode_reduce'):
    module = ModuleType('sglang.srt.layers.muse_spark.' + name)
    module.__file__ = str(layers / (name + '.py'))
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, 'exec'), vars(module))
    sys.modules[module.__name__] = module
jit = ModuleType('sglang.kernels.jit')
jit.__file__ = str(stage / 'python/sglang/kernels/jit/__init__.py')
kernels = ModuleType('sglang.kernels')
kernels.jit = jit
sys.modules[kernels.__name__] = kernels
utils = ModuleType('sglang.kernels.jit.utils')
utils.load_jit = lambda *a, **kw: SimpleNamespace(configure=lambda handle, shared: True)
sys.modules[utils.__name__] = utils
compiler = ModuleType('humming.jit.compiler')
compiler.NVCCCompiler = type('NVCCCompiler', (), {})
sys.modules[compiler.__name__] = compiler
path = layers / 'decode_gate_up.py'
gate = {}
exec(compile(path.read_text(), str(path), 'exec'), gate)
sources = {}
def emit(kernel):
    m, n, _ = kernel.block_shape
    assert kernel.both_pairs and kernel.live_reduction
    stem = f'w13_m{m}_n{n}'
    kernel.func = m * 1000 + n
    source = kernel.code + '\nauto ptr = reinterpret_cast<void*>(&' + kernel.kernel_expr + ');\n'
    (target / (stem + '.cu')).write_text(source)
    sources[stem] = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         block=kernel.block_shape, warp=kernel.warp_shape,
                         stages=kernel.num_stages, smem=kernel.estimated_smem_size)
OfflineRuntime.prepare = emit
OfflineRuntime.check_context = lambda self: None
with patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(multi_processor_count=132)):
    obj = LayerConfig(shape_n=8192, shape_k=4096, num_experts=32, a_dtype=dtypes.bfloat16,
                      b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3,
                      c_dtype=dtypes.bfloat16, weight_scale_group_size=16)
    configs = scope['Sm90Heuristics'].get_configs(layer_config=obj, gemm_type=GemmType.GROUPED_CONTIGUOUS)
    compute = json.dumps({'gemm_type': 'grouped_contiguous', 'use_f16_accum': False})
    for lo, hi, config in configs:
        if lo >= 256:
            continue
        m = config['block_shape'][0]
        config = config | dict(use_stream_k=False, use_tma_c=False)
        for n in ([256] if m == 8 else [256, 512]):
            tuned = config | {'block_shape': [m, n, 128], 'warp_shape': [m, n // 8, 64]}
            result = gate['_compile'](0, obj.to_str(), compute, json.dumps(tuned))
            assert result is not None and result[1] == m * 1000 + n
print(json.dumps(sources, indent=2))
