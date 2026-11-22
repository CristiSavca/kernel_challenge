"""Emit real Humming 0.1.12 grouped BF16/FP4 W2 images for wider row tiles."""
import hashlib
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
vendor, stage, images = map(Path, sys.argv[1:])
driver = here.parent / "v169_grouped_expert_tile_cohorts/vendor_codegen.py"
preamble = driver.read_text().split("sp=importlib.util.spec_from_file_location", 1)[0]
exec(compile(preamble, str(driver), "exec"))
configs = {}


def emit(kernel):
    m, n, _ = kernel.block_shape
    warp_m = kernel.warp_shape[0]
    name = f"m{m}_n{n}_w{warp_m}_a{kernel.multi_cast_size_a}"
    source = kernel.code + "\nauto ptr = reinterpret_cast<void*>(&" + kernel.kernel_expr + ");\n"
    (target / (name + ".cu")).write_text(source)
    configs[name] = dict(block=kernel.block_shape, warp=kernel.warp_shape, math_threads=kernel.num_math_threads,
                         shared=kernel.estimated_smem_size, threads=kernel.num_threads, stages=kernel.num_stages,
                         sha256=hashlib.sha256(source.encode()).hexdigest())


OfflineRuntime.prepare = emit
with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), patch.object(
    torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=132)
):
    op = LayerConfig(shape_n=4096, shape_k=4096, num_experts=32, a_dtype=dtypes.bfloat16,
                     b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3, c_dtype=dtypes.bfloat16,
                     weight_scale_group_size=16)
    configs_for_op = scope["Sm90Heuristics"].get_config(op, 65408, gemm_type=GemmType.GROUPED_CONTIGUOUS)
    for m,n,wm,cast_a in (
        (176,256,176,1), (192,256,192,1), (208,256,208,1), (224,256,224,1),
        (240,128,240,2), (248,128,248,2), (256,128,256,2), (256,128,128,2),
        (352,128,176,2), (512,128,128,2)
    ):
        config = configs_for_op | dict(block_shape=(m,n,64), warp_shape=(wm,32,64), multi_cast_size_a=cast_a,
                                      use_stream_k=False, use_tma_c=False)
        HummingKernel(shape_n=4096, shape_k=4096, num_experts=32, a_dtype=dtypes.bfloat16, b_dtype=dtypes.float4e2m1,
                      bs_dtype=dtypes.float8e4m3, c_dtype=dtypes.bfloat16, weight_scale_group_size=16,
                      gemm_type=GemmType.GROUPED_CONTIGUOUS, **config)
print(json.dumps(configs, indent=2))

