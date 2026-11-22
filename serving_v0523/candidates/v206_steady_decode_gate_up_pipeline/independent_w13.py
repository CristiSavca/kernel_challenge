"""Run with python3 independent_w13.py <V206 scratch containing sglang, humming, full>.

The CPU schedule and decoder routing tests emulate hardware. SASS is from the
actual offline-built W13 machines; the H200 figures are separate archived runs.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from random import Random
import re
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent
ROOT = CANDIDATES.parents[1]


def source_function(path, name, **scope):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, 'name', '') == name)
    exec(compile(ast.Module([node], []), str(path), 'exec'), scope)
    return scope[name]


def check_include(scratch, helper):
    enable = source_function(helper, "enable_full_pair", Path=Path, hashlib=hashlib)
    include = "#include <humming/kernel/humming_ws.cuh>"
    rejected = 0
    accepted = 0
    with patch.dict(sys.modules, humming=SimpleNamespace(__path__=[str(scratch / 'humming')])):
        for width in (8,16,24):
            for content in (include, "", include+"\n"+include, '#include "humming/kernel/humming_ws.cuh"'):
                kernel = SimpleNamespace(block_shape=(width,256,128), use_tma_a=True, use_tma_b=True, use_tma_bs=True,
                                         use_mbarrier=True, multi_cast_size_a=1, multi_cast_size_b=1, code=content)
                try:
                    enable(kernel)
                except ValueError:
                    assert content != include and kernel.code == content
                    rejected += 1
                else:
                    assert content == include
                    body, macro = kernel.code.split("\n#define MUSE_PAIR_LAST_K_SLICES\n")
                    assert not macro
                    assembled = (scratch / "full" / f"decode_{width}.cu").read_text()
                    assert body in assembled and "if (warp_iter_id + 1 == Ctx::kWarpIters) consumer.arrive(stage_id)" in body
                    accepted += 1
    return dict(source_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                matches_actual_compiled_loops=accepted, rejects_missing_duplicated_or_altered_include=rejected)


def check_runtime(gate):
    run_gate_up = source_function(gate, "run_gate_up")

    class Tensor:
        def __init__(self, shape, pointer):
            self.shape = shape
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

        def new_empty(self, shape):
            return Tensor(shape, 55)

        def int(self):
            return self

    class Launcher:
        def __init__(self):
            self.calls = []
            self.success = True

        def run(self, *args):
            self.calls.append(("decode", args))
            return self.success

        def run_prompt(self, *args):
            self.calls.append(("prompt", args))
            return self.success

    layer = SimpleNamespace(w13_weight=Tensor((32,256,16384), 64), w13_weight_scale=Tensor((32,256,8192),128),
                            w13_global_scale=Tensor((32,), 256), intermediate_size_per_partition=4096)
    launcher = Launcher()
    consume = (
        Tensor((256,4096), 512), Tensor((256,), 1024), Tensor((33,), 2048),
    )
    checks = 0
    for width in (8, 16, 24, 176):
        prompt = width == 176
        key = "humming_prompt_gate_up" if prompt else "humming_decode_gate_up"
        setattr(layer, key, None)
        assert run_gate_up(layer, *consume, width, prompt=prompt) is None
        bank = tuple((t.data_ptr(), t.shape) for t in (layer.w13_weight, layer.w13_weight_scale))
        maps = object()
        kernel = (object(), 867, 12345, width)
        setattr(layer, key, (launcher, maps, bank, {width: kernel}, 132))
        before = len(launcher.calls)
        assert run_gate_up(layer, *consume, 999, prompt=prompt) is None
        for weight in (layer.w13_weight, layer.w13_weight_scale):
            weight.pointer += 16
            assert run_gate_up(layer, *consume, width, prompt=prompt) is None
            weight.pointer -= 16
        assert len(launcher.calls) == before
        out = run_gate_up(layer, *consume, width, prompt=prompt)
        assert out.shape == (256,4096)
        expected = (consume[0], out, consume[1], consume[2], layer.w13_global_scale, maps, *kernel[1:], 132)
        assert launcher.calls[-1] == ("prompt" if prompt else "decode", expected)
        launcher.success = False
        assert run_gate_up(layer, *consume, width, prompt=prompt) is None
        launcher.success = True
        checks += 6
    return checks


def disassembly(path, width):
    pending_regs = set()
    drains = []
    arrivals = []
    transfers = Counter()
    writes = 0
    mmas = 0
    for m in re.finditer(r"/\*([0-9a-f]+)\*/\s*(?:@[!\w]+\s+)?([A-Z][.A-Za-z0-9_]+)\s+([^;\n]*);", path.read_text()):
        _, op, args = m.groups()
        if op.startswith("UTMALDG."):
            transfers[op] += 1
        if op.startswith("HGMMA."):
            assert op == f"HGMMA.64x{width}x16.F32.BF16"
            dst, source, desc, accum, *_ = (s.strip() for s in args.split(','))
            assert dst == accum and desc.startswith("gdesc[")
            first = int(source[1:])
            pending_regs.update(range(first, first+4))
            mmas += 1
        elif op.startswith("WARPGROUP.DEPBAR") and mmas:
            assert "0x0" in args
            pending_regs.clear()
            drains.append(mmas)
        elif op.startswith("SYNCS.ARRIVE") and mmas:
            assert not pending_regs
            arrivals.append(mmas)
        elif pending_regs:
            target = re.match(r"R(\d+)(?:,|\.)", args)
            if target:
                assert op.split(".")[0] in {"LDS","IMAD","PRMT","LOP3","SHF","HMUL2","VIADD","SEL","MOV","S2R"}
                first = int(target[1])
                size = 4 if ".128" in op else 2 if ".64" in op else 1
                assert pending_regs.isdisjoint(range(first, first+size))
                writes += size
    assert drains == list(range(4,33,4)) and arrivals == [8,16,24,32,32]
    assert transfers == {"UTMALDG.2D":7, "UTMALDG.4D":14}
    return dict(cubin_sha256=hashlib.sha256(path.with_suffix(".cubin").read_bytes()).hexdigest(),
                arrivals_after_mma=arrivals, drains_after_mma=drains, outstanding_register_writes_checked=writes)


def main(scratch):
    source = scratch / "sglang/python/sglang"
    vendor = scratch / "humming/include/humming"
    gate = source / "srt/layers/muse_spark/decode_gate_up.py"
    helper = gate.with_name("decode_mma_pair.py")
    native = (vendor / "memory/g2s_pipeline.cuh").read_text()
    for code in ("smem.stages[stage_id].a, mbar_ptr", "smem.stages[stage_id].b, mbar_ptr",
                 "smem.stages[stage_id].bs, mbar_ptr", "expect_tma_load<kHasTmaMBarrier>(mbar_ptr, load_bytes.x)"):
        assert code in native
    s2r = (vendor / "memory/s2r_pipeline.cuh").read_text()
    for code in ("loader_b.load(smem.stages[stage_id].b", "loader_bs.load(smem.stages[stage_id].bs",
                 "if constexpr (!kUseWgmma)"):
        assert code in s2r
    fused = (source / "kernels/jit/csrc/muse_spark/decode_gate_up.cuh").read_text()
    assert "const auto mbar = cast_smem_ptr_to_uint(bar)" in fused
    assert "struct MusePackedGateLoader" in fused and "ctx.smem.reduce[" in fused
    parent = CANDIDATES / "v205_steady_decode_weight_pipeline"
    replay = runpy.run_path(str(parent / "independent_multijob.py"))["replay"]
    rng = Random(206)
    shapes = [(32,)*2, (32,)*5, (32,)*32] + [(i,32,33-i) for i in range(1,33)]
    events = sum(replay(s,rng) for s in shapes for _ in range(9))
    images = {str(w): disassembly(scratch/"full"/f"decode_{w}.sass",w) for w in (8,16,24)}
    geometry = {}
    for role, n, block_n, block_k in (("W13",8192,256,128), ("W2",4096,128,256)):
        k = 4096
        geometry[role] = dict(full_k=k, output_channels=n, cta_n=block_n, cta_k=block_k, output_ctas=n//block_n,
                              k_stages=k//block_k, k_warps=block_k//64, matrix_warpgroups=4,
                              packed_weights_and_group_scales=k*n//2+k*n//16,
                              paired_parent_waits_per_warpgroup_per_job=3*(k//block_k),
                              full_pair_waits_per_warpgroup_per_job=2*(k//block_k))
    experiments = ROOT/"experiments/modal_serving"
    archive = json.loads((experiments/"20261113T065259Z/result.json").read_text())
    assert archive["gpu"] == "NVIDIA H200"
    quant = archive["dequant_us"]
    standalone = json.loads((experiments/"20261112T185954Z/result.json").read_text())
    assert standalone["gpu"] == "NVIDIA H200"
    physical = {c['T']: c['stock_us'] for c in standalone["cases"] if c['T'] in (8,32,128,256)}
    control = json.loads((ROOT/"serving_v0523/runs/04b01f958aa5487c.json").read_text())
    assert control['result']['passed'] and control['result']['rungs'][0]['concurrency']==16
    return dict(patch_sha256=hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(), images=images,
                full_pair_include=check_include(scratch, helper),
                run_gate_up_host_cases=check_runtime(gate), ring_job_sequences=len(shapes), ring_schedules=9*len(shapes),
                ring_events=events, geometry=geometry,
                separate_single_H200_dequant_graph_us=dict(w13=quant["w13_8192x4096"]["graph_us"],w2=quant["w2_4096x4096"]["graph_us"]),
                separate_single_H200_whole_Marlin_call_us=physical, V68_16_user_official_tok_s_per_gpu=control['result']['rungs'][0]['tok_s_per_gpu'],
                matched_V206_H200_measurement=False)


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
