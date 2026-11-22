"""Render a diagnostic eight-CTA multicast image from the same frozen specialization."""
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

here=Path(__file__).resolve().parent
ns=runpy.run_path(str(here/"render.py"))
scope=ns["ns"]
with patch.object(scope["torch"].cuda,"get_device_capability",return_value=(9,0)), patch.object(scope["torch"].cuda,"get_device_properties",return_value=ns["SimpleNamespace"](multi_processor_count=132)):
    klass=type(ns["kernels"]["WidePromptWeightPipeline2"])
    args=scope["json"].loads(ns["obj"].to_str())|dict(gemm_type="grouped_contiguous",use_f16_accum=False)|ns["tuned"]
    args.pop("sublayer_name",None)
    args.update(shape_n=4096,block_shape=(256,128,64),warp_shape=(256,16,64),multi_cast_size_a=8,use_tma_bs=False,use_cp_async=False)
    runtime=ns["runtime"]
    def record(kernel):
        code=kernel.code+"\nauto ptr = reinterpret_cast<void*>(&"+kernel.kernel_expr+");\n"
        assert "Ctx::kMultiCastSizeA == 4);" in code
        code=code.replace("Ctx::kMultiCastSizeA == 4);","Ctx::kMultiCastSizeA == 4 || Ctx::kMultiCastSizeA == 8);")
        (Path(sys.argv[3])/"DiagnosticGroup8.cu").write_text(code)
    with patch.object(runtime,"prepare",record):
        klass(**args)

