"""One-H200 A/B of the Muse Marlin expert path: stock SGLang v0.5.23 against a patched module file.

Runs inside the judge's image (SGLang at /sgl-workspace/sglang). Synthetic NVFP4 weights in checkpoint layout are
prepared by SGLang's own Marlin preparation; both variants then run on identical weights and routes for rank 3 of EP8.
Writes one JSON result. Nothing here reads the competition workload.
"""

import importlib.util
import json
import os
import sys
import time
import traceback

import torch

RESULT = sys.argv[1]
_ARGS = [a for a in sys.argv[2:] if not a.startswith(("tokens:", "install:", "layout", "import:"))]
# import:<module>: after installing files, import these modules and stop (a smoke test of patched sources)
_IMPORTS = [a[7:] for a in sys.argv[2:] if a.startswith("import:")]
LAYOUT_BENCH = "layout" in sys.argv[2:]
# install:<path under the sglang package>=<uploaded file>: new source files a variant needs.
_INSTALL = dict(a[8:].split("=", 1) for a in sys.argv[2:] if a.startswith("install:"))
_TOK = [a for a in sys.argv[2:] if a.startswith("tokens:")]
VARIANTS = dict(a.split("=", 1) for a in _ARGS if not a.startswith("kern:"))  # name=path of a patched fp4 file
# kern:name=path.json -> {"fp4": path of the fp4 file to use, "cuh": [[old, new], ...]} applied to a copy of the Marlin
# MoE kernel source, built as its own JIT module so stock stays untouched.
KERNELS = dict(a[5:].split("=", 1) for a in _ARGS if a.startswith("kern:"))


def load_kernel_variant(name, spec_path):
    import os
    import types
    import sglang.kernels.jit as jit_pkg
    import sglang.kernels.ops.moe.moe_wna16_marlin as base_wrapper
    spec = json.load(open(spec_path))
    csrc = os.path.join(os.path.dirname(jit_pkg.__file__), "csrc")
    src = open(os.path.join(csrc, "gemm/marlin_moe/moe_wna16_marlin.cuh")).read()
    for old, new in spec["cuh"]:
        if src.count(old) != 1:
            raise RuntimeError(f"kernel variant {name}: patch target count {src.count(old)}: {old[:80]}")
        src = src.replace(old, new)
    rel = f"gemm/marlin_moe/moe_wna16_marlin_{name}.cuh"
    with open(os.path.join(csrc, rel), "w") as f:
        f.write(src)
    wsrc = open(base_wrapper.__file__).read()
    for old, new in (('"moe_wna16_marlin",', f'"moe_wna16_marlin_{name}",'),
                     ('"gemm/marlin_moe/moe_wna16_marlin.cuh"', f'"{rel}"')):
        if wsrc.count(old) != 1:
            raise RuntimeError(f"wrapper patch target count {wsrc.count(old)}: {old}")
        wsrc = wsrc.replace(old, new)
    modname = f"sglang.kernels.ops.moe.moe_wna16_marlin_{name}"
    mod = types.ModuleType(modname)
    mod.__file__ = base_wrapper.__file__
    exec(compile(wsrc, f"<{modname}>", "exec"), mod.__dict__)
    sys.modules[modname] = mod
    fp4 = open(spec["fp4"]).read()
    imp = "from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm"
    if fp4.count(imp) != 1:
        raise RuntimeError("fp4 import target not unique")
    fp4 = fp4.replace(imp, f"from {modname} import moe_wna16_marlin_gemm")
    fpath = f"/tmp/w/kvariant_{name}.py"
    with open(fpath, "w") as f:
        f.write(fp4)
    return fpath
H = I = 4096
E_GLOBAL, E_LOCAL, RANK, TOPK = 256, 32, 3, 8
TOKENS = tuple(int(t) for t in _TOK[0][7:].split(",")) if _TOK else (8, 128, 256)
REPS = 30


def dense_ceiling(T, local, dev, flush, sink):
    """cuBLAS BF16 GEMM time for the same expert work with dense weights (no dequantization): an upper bound."""
    n = max(1, int(local.sum()) // E_LOCAL)
    g = torch.Generator(device=dev).manual_seed(5)
    w13 = (torch.randn((E_LOCAL, 2 * I, H), generator=g, device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((E_LOCAL, H, I), generator=g, device=dev) * 0.02).to(torch.bfloat16)
    x = (torch.randn((E_LOCAL, n, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)

    def fn():
        outs = []
        for e in range(E_LOCAL):
            h = torch.matmul(x[e], w13[e].t())
            a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
            outs.append(torch.matmul(a, w2[e].t()))
        return outs

    fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        keep = fn()
    ts = []
    for _ in range(9):
        sink.copy_(flush.sum())
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    del gr, keep, w13, w2, x
    return {"rows_per_expert": n, "us": round(sorted(ts)[4], 1)}
out = {"status": "started", "cases": [], "t0": time.time()}


class Bound:
    """A layer seen through one variant's class, so methods the variant adds resolve to its own code."""

    def __init__(self, layer, cls):
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_cls", cls)

    def __getattr__(self, name):
        cls = object.__getattribute__(self, "_cls")
        for klass in cls.__mro__:
            if name in klass.__dict__ and klass is not torch.nn.Module:
                attr = klass.__dict__[name]
                return attr.__get__(self, cls) if hasattr(attr, "__get__") else attr
        return getattr(object.__getattribute__(self, "_layer"), name)


def e2m1_reference(codes, scales):
    """Dense BF16 [N, K] from checkpoint-layout codes [N, K / 2] and E4M3 block scales [N, K / 16]."""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    nib = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(codes.shape[0], -1).long()
    return (lut[nib] * scales.float().repeat_interleave(16, dim=1)).to(torch.bfloat16)


def save():
    out["elapsed_s"] = round(time.time() - out["t0"], 1)
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=1)


def build_layer(dev):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

    g = torch.Generator(device=dev).manual_seed(1234)
    layer = torch.nn.Module()

    def codes(*shape):
        return torch.randint(0, 256, shape, generator=g, device=dev, dtype=torch.uint8)

    def scales(*shape):
        # positive normal E4M3 bytes around 1.0 (exponent field 6..8)
        return torch.randint(0x30, 0x48, shape, generator=g, device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)

    for name, val in (
        ("w13_weight", codes(E_LOCAL, 2 * I, H // 2)),
        ("w2_weight", codes(E_LOCAL, H, I // 2)),
        ("w13_weight_scale", scales(E_LOCAL, 2 * I, H // 16)),
        ("w2_weight_scale", scales(E_LOCAL, H, I // 16)),
        ("w13_weight_scale_2", (torch.rand(E_LOCAL, generator=g, device=dev) * 0.5 + 0.75).float()),
        ("w2_weight_scale_2", (torch.rand(E_LOCAL, generator=g, device=dev) * 0.5 + 0.75).float()),
    ):
        layer.register_parameter(name, torch.nn.Parameter(val, requires_grad=False))
    layer.raw = {e: tuple(getattr(layer, n)[e].clone() for n in
                          ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")) for e in (0, 5)}
    layer.moe_runner_config = type("Cfg", (), {"is_gated": True})()
    layer.intermediate_size_per_partition = I
    layer.params_dtype = torch.bfloat16
    prepare_moe_nvfp4_layer_for_marlin(layer, group_size=16)
    expert_map = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
    expert_map[RANK * E_LOCAL:(RANK + 1) * E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=dev)
    layer.register_buffer("_expert_map", expert_map, persistent=False)
    layer.num_local_experts = E_LOCAL
    layer.global_num_experts = E_GLOBAL
    return layer


def routes(T, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.rand((T, E_GLOBAL), generator=g, device=dev).argsort(dim=1)[:, :TOPK].to(torch.int32)
    scores = torch.rand((T, TOPK), generator=g, device=dev)
    scores = (scores / scores.sum(-1, keepdim=True)).to(torch.float32)
    x = (torch.randn((T, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    return x, scores, ids


def timed(fn, flush, sink, reps):
    ts = []
    for _ in range(reps):
        sink.copy_(flush.sum())
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    return ts


def main():
    dev = torch.device("cuda")
    out["gpu"] = torch.cuda.get_device_name(0)
    import shutil
    import sglang
    out["sglang_file"] = sglang.__file__
    pkg = os.path.dirname(sglang.__file__)
    for rel, src in _INSTALL.items():
        os.makedirs(os.path.dirname(os.path.join(pkg, rel)), exist_ok=True)
        shutil.copyfile(src, os.path.join(pkg, rel))
    if _IMPORTS:
        out["imports"] = {}
        for name in _IMPORTS:
            try:
                mod = importlib.import_module(name)
                out["imports"][name] = "ok: " + str(getattr(mod, "__file__", ""))
            except Exception as ex:
                out["imports"][name] = f"{type(ex).__name__}: {ex}"[-600:] + traceback.format_exc()[-1500:]
        out["status"] = "complete"
        save()
        return
    from sglang.srt.layers.muse_spark import muse_spark_fp4_moe as stock
    classes = {"stock": stock.MuseSpark1xFp4Experts}
    for name, spec_path in KERNELS.items():
        VARIANTS[name] = load_kernel_variant(name, spec_path)
    for name, path in VARIANTS.items():
        spec = importlib.util.spec_from_file_location(f"variant_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        classes[name] = mod.MuseSpark1xFp4Experts
    # Build every Marlin module up front, in parallel (stock: EP; variants: sorted local with inactive prefix).
    from concurrent.futures import ThreadPoolExecutor
    import sglang.kernels.ops.moe.moe_wna16_marlin as base_wrapper
    wrappers = [(base_wrapper, (torch.bfloat16, True, False, False, False, False)),
                (base_wrapper, (torch.bfloat16, False, False, True, False, True))]
    wrappers += [(sys.modules[f"sglang.kernels.ops.moe.moe_wna16_marlin_{n}"], (torch.bfloat16, False, False, True, False, True))
                 for n in KERNELS]
    tb = time.time()
    with ThreadPoolExecutor(len(wrappers)) as ex:
        list(ex.map(lambda wa: wa[0]._jit_moe_wna16_marlin_module(*wa[1]), wrappers))
    out["build_s"] = round(time.time() - tb, 1)
    save()
    layer = build_layer(dev)
    out["prepared"] = True
    save()
    fns = {n: (lambda x, sc, ids, c=c: c._marlin_expert_outputs(Bound(layer, c), x, sc, ids)) for n, c in classes.items()}

    # The dequantization kernel against the checkpoint-layout weights, bit for bit.
    try:
        from sglang.kernels.ops.moe.nvfp4_marlin_dequant import nvfp4_marlin_dequant
        checks = {}
        for e, (c13, s13, c2, s2) in layer.raw.items():
            for tag, codes, scales, qw, qs in (("w13", c13, s13, layer.w13_weight[e], layer.w13_weight_scale[e]),
                                               ("w2", c2, s2, layer.w2_weight[e], layer.w2_weight_scale[e])):
                ref = e2m1_reference(codes, scales)
                got = nvfp4_marlin_dequant(qw, qs, torch.empty_like(ref))
                torch.cuda.synchronize()
                checks[f"e{e}_{tag}"] = {"bitwise": bool(torch.equal(got.view(torch.int16), ref.view(torch.int16))),
                                         "mismatch_frac": float((got.view(torch.int16) != ref.view(torch.int16)).float().mean()),
                                         "ref_rms": float(ref.float().pow(2).mean().sqrt())}
        c13, s13 = layer.raw[5][0], layer.raw[5][1]
        up = nvfp4_marlin_dequant(layer.w13_weight[5], layer.w13_weight_scale[5],
                                  torch.empty((I, H), dtype=torch.bfloat16, device=dev), I)
        torch.cuda.synchronize()
        checks["e5_w13_up_half"] = {"bitwise": bool(torch.equal(up.view(torch.int16),
                                                                 e2m1_reference(c13, s13)[I:].view(torch.int16)))}
        out["dequant_vs_checkpoint"] = checks
        w13 = torch.empty((I, H), dtype=torch.bfloat16, device=dev)
        w2 = torch.empty((H, I), dtype=torch.bfloat16, device=dev)
        flush0 = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
        sink0 = torch.zeros((), device=dev)
        full13 = torch.empty((2 * I, H), dtype=torch.bfloat16, device=dev)
        dq = {}
        for tag, fn in (("w13_8192x4096", lambda: nvfp4_marlin_dequant(layer.w13_weight[7], layer.w13_weight_scale[7], full13)),
                        ("w13_half_4096x4096", lambda: nvfp4_marlin_dequant(layer.w13_weight[7], layer.w13_weight_scale[7], w13, I)),
                        ("w2_4096x4096", lambda: nvfp4_marlin_dequant(layer.w2_weight[7], layer.w2_weight_scale[7], w2))):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            dq[tag] = {"eager_us": round(sorted(timed(fn, flush0, sink0, 15))[7], 1),
                       "graph_us": round(sorted(timed(gr.replay, flush0, sink0, 15))[7], 1)}
            del gr
        out["dequant_us"] = dq
        del full13

        if not LAYOUT_BENCH:
            raise StopIteration
        # GPU-side cost of the two buffer layouts over all 32 experts at 512 rows each, hot buffers, one graph:
        # "pieces": gate, up and down as three 4096x4096 dequants into reused 33.5 MB buffers;
        # "full": gate+up as one 8192x4096 dequant (67 MB), then down.
        rows_n = 512
        g = torch.Generator(device=dev).manual_seed(9)
        xr = (torch.randn((E_LOCAL, rows_n, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)
        wa = torch.empty((I, H), dtype=torch.bfloat16, device=dev)
        wb = torch.empty((H, I), dtype=torch.bfloat16, device=dev)
        wf = torch.empty((2 * I, H), dtype=torch.bfloat16, device=dev)
        ya = torch.empty((rows_n, I), dtype=torch.bfloat16, device=dev)
        yf = torch.empty((rows_n, 2 * I), dtype=torch.bfloat16, device=dev)
        yo = torch.empty((rows_n, H), dtype=torch.bfloat16, device=dev)

        def pieces(gemm=True):
            for e in range(E_LOCAL):
                for half in range(2):
                    nvfp4_marlin_dequant(layer.w13_weight[e], layer.w13_weight_scale[e], wa, half * I)
                    if gemm:
                        torch.mm(xr[e], wa.t(), out=ya)
                nvfp4_marlin_dequant(layer.w2_weight[e], layer.w2_weight_scale[e], wb)
                if gemm:
                    torch.mm(ya, wb.t(), out=yo)

        def full(gemm=True):
            for e in range(E_LOCAL):
                nvfp4_marlin_dequant(layer.w13_weight[e], layer.w13_weight_scale[e], wf)
                if gemm:
                    torch.mm(xr[e], wf.t(), out=yf)
                nvfp4_marlin_dequant(layer.w2_weight[e], layer.w2_weight_scale[e], wb)
                if gemm:
                    torch.mm(ya, wb.t(), out=yo)

        def one_buffer(gemm=True):
            # every dequant lands in the same 33.5 MB buffer (valid here because I == H)
            for e in range(E_LOCAL):
                for half in range(2):
                    nvfp4_marlin_dequant(layer.w13_weight[e], layer.w13_weight_scale[e], wa, half * I)
                    if gemm:
                        torch.mm(xr[e], wa.t(), out=ya)
                nvfp4_marlin_dequant(layer.w2_weight[e], layer.w2_weight_scale[e], wa)
                if gemm:
                    torch.mm(ya, wa.t(), out=yo)

        def split(piece_rows, gemm=True):
            # gate, up and down in pieces of `piece_rows` output rows, all dequantized into one small buffer
            wp = wsmall[:piece_rows]
            for e in range(E_LOCAL):
                for r0 in range(0, 2 * I, piece_rows):
                    nvfp4_marlin_dequant(layer.w13_weight[e], layer.w13_weight_scale[e], wp, r0)
                    if gemm:
                        torch.mm(xr[e], wp.t(), out=youts[piece_rows])
                for r0 in range(0, H, piece_rows):
                    nvfp4_marlin_dequant(layer.w2_weight[e], layer.w2_weight_scale[e], wp, r0)
                    if gemm:
                        torch.mm(ya, wp.t(), out=youts[piece_rows])

        wsmall = torch.empty((2048, H), dtype=torch.bfloat16, device=dev)
        ysmall = None
        youts = {r: torch.empty((rows_n, r), dtype=torch.bfloat16, device=dev) for r in (2048, 1024, 512)}
        layout = {}
        for rows_p in (2048, 1024, 512):
            for tag, fn in ((f"split{rows_p}", lambda r=rows_p: split(r)), (f"split{rows_p}_dequant_only", lambda r=rows_p: split(r, False))):
                fn()
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                layout[tag] = round(sorted(timed(gr.replay, flush0, sink0, 9))[4], 1)
                del gr
        for tag, fn in (("pieces", pieces), ("full", full), ("one_buffer", one_buffer),
                        ("pieces_dequant_only", lambda: pieces(False)), ("full_dequant_only", lambda: full(False)),
                        ("one_buffer_dequant_only", lambda: one_buffer(False))):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            layout[tag] = round(sorted(timed(gr.replay, flush0, sink0, 9))[4], 1)
            del gr
        out["layout_us_32_experts_512_rows"] = layout
        del xr, wa, wb, wf, ya, yf, yo, wsmall, ysmall
        del flush0, w13, w2
    except StopIteration:
        pass
    except Exception as ex:
        out["dequant_error"] = f"{type(ex).__name__}: {ex}"[-1500:]
        out["dequant_trace"] = traceback.format_exc()[-2500:]
    save()

    # The dense path as a break inside SGLang's breakable CUDA graph: capture once, replay with new inputs and
    # new routing, and compare with a plain eager call bit for bit.
    for name, cls in classes.items():
        if not name.startswith("dense"):
            continue
        try:
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
                BreakableCUDAGraph, BreakableCUDAGraphCapture)
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
                enable_breakable_cuda_graph)
            T = 16384
            xb, sb, ib = routes(T, 1, dev)
            # the clone runs in the segment after the break and reads what the break wrote
            run = lambda: cls._marlin_expert_outputs(Bound(layer, cls), xb, sb, ib)[0].clone()
            graph = BreakableCUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with enable_breakable_cuda_graph():
                with BreakableCUDAGraphCapture(graph, pool=torch.cuda.graph_pool_handle(), stream=side):
                    captured = run()
            torch.cuda.current_stream().wait_stream(side)
            res = {"segments": len(graph._segments), "breaks": len(graph._break_fns)}
            for trial, seed in enumerate((11, 12, 13)):
                x2, s2, i2 = routes(T, seed, dev)
                if trial == 2:  # skewed routing: most local routes on four experts, some experts unused
                    hot = RANK * E_LOCAL + torch.randint(0, 4, i2[:, :1].shape, device=dev, dtype=torch.int32)
                    i2[:, :1] = hot
                xb.copy_(x2)
                sb.copy_(s2)
                ib.copy_(i2)
                with enable_breakable_cuda_graph():
                    graph.replay()
                torch.cuda.synchronize()
                eager = cls._marlin_expert_outputs(Bound(layer, cls), xb, sb, ib)[0]
                torch.cuda.synchronize()
                mask = ((ib >= RANK * E_LOCAL) & (ib < (RANK + 1) * E_LOCAL)).unsqueeze(-1).expand(T, TOPK, H)
                res[f"trial{trial}"] = {"local_routes": int(mask[:, :, 0].sum()),
                                        "replay_equals_eager": bool(torch.equal(captured[mask], eager[mask])),
                                        "finite": bool(torch.isfinite(captured[mask]).all())}
                del eager, mask
            del captured, graph
            out[f"bcg_{name}"] = res
        except Exception as ex:
            out[f"bcg_{name}"] = {"error": f"{type(ex).__name__}: {ex}"[-900:], "trace": traceback.format_exc()[-2500:]}
        save()
        break

    # The startup profiler's expert check and log lines, driven by hand on one layer.
    for name, cls in classes.items():
        if not name.startswith("dense"):
            continue
        try:
            import logging
            from sglang.srt.layers.muse_spark.prefill_profile import PrefillStageProfile as P
            lines = []
            handler = logging.Handler()
            handler.emit = lambda record: lines.append(record.getMessage())
            plog = logging.getLogger("sglang.srt.layers.muse_spark.prefill_profile")
            plog.addHandler(handler)
            plog.setLevel(logging.INFO)
            x, sc, ids = routes(4096, 5, dev)
            assert P.begin(16384, True) and P.forces_marlin() and not P.wants_expert_check()
            P.lap("stage_a")
            P.end(16384, 62)
            assert P.begin(16384, True) and not P.forces_marlin() and P.wants_expert_check()
            cls._check_dense_experts(Bound(layer, cls), x, sc, ids)
            assert not P.wants_expert_check()
            P.lap("stage_b")
            P.end(16384, 62)
            assert not P.begin(4096, True) and not P.begin(16384, False)
            assert P.begin(8192, True) and not P.forces_marlin()
            P.lap("stage_c")
            P.end(8192, 62)
            P.large_done = P.small_done = 99  # leave the profiler off for the timing cases below
            assert not P.begin(16384, True) and not P.begin(8192, True)
            plog.removeHandler(handler)
            out["profile_lines"] = lines
        except Exception as ex:
            out["profile_error"] = f"{type(ex).__name__}: {ex}"[-900:] + traceback.format_exc()[-1500:]
        save()
        break

    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    sink = torch.zeros((), device=dev)
    for T in TOKENS:
        case = {"T": T}
        try:
            x, scores, ids = routes(T, 77 + T, dev)
            local = ((ids >= RANK * E_LOCAL) & (ids < (RANK + 1) * E_LOCAL))
            case["local_routes"] = int(local.sum())
            case["gflop"] = round(2 * case["local_routes"] * (2 * I * H + H * I) / 1e9, 1)
            reps = REPS if T <= 1024 else 9
            m = local.unsqueeze(-1).expand(T, TOPK, H)
            ys = fns["stock"](x, scores, ids)[0]
            torch.cuda.synchronize()
            ref = ys[m].clone()
            del ys
            case["stock_rms"] = float(ref.float().pow(2).mean().sqrt()) if ref.numel() else 0.0
            graphs, keep = {}, {}
            for n, fn in fns.items():
                yv, lv = fn(x, scores, ids)
                torch.cuda.synchronize()
                if n != "stock":
                    got = yv[m]
                    d = (ref.float() - got.float()).abs()
                    case[f"{n}_bitwise"] = bool(torch.equal(ref, got))
                    case[f"{n}_max_abs_diff"] = float(d.max()) if d.numel() else 0.0
                    case[f"{n}_max_diff_over_rms"] = case[f"{n}_max_abs_diff"] / max(case["stock_rms"], 1e-30)
                    case[f"{n}_frac_diff_elems"] = float((d > 0).float().mean()) if d.numel() else 0.0
                    case[f"{n}_finite"] = bool(torch.isfinite(got).all())
                    del got, d
                del yv, lv
                if n.startswith("dense"):
                    continue  # routing-dependent shapes: timed eagerly below, as a graph break would run it
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fn(x, scores, ids)
                torch.cuda.current_stream().wait_stream(s)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    keep[n] = fn(x, scores, ids)
                graphs[n] = gr
            torch.cuda.synchronize()
            for n, fn in fns.items():
                run = graphs[n].replay if n in graphs else (lambda fn=fn: fn(x, scores, ids))
                ts = timed(run, flush, sink, reps)
                case[f"{n}_us"] = round(sorted(ts)[reps // 2], 2)
                case[f"{n}_mode"] = "graph" if n in graphs else "eager"
            for n in fns:
                if n != "stock":
                    case[f"{n}_speedup"] = round(case["stock_us"] / case[f"{n}_us"], 4)
            del graphs, keep, ref
            if T >= 512:
                case["dense_ceiling"] = dense_ceiling(T, local, dev, flush, sink)
        except Exception as ex:  # record and continue to the next size
            case["error"] = f"{type(ex).__name__}: {ex}"[-900:]
            case["trace"] = traceback.format_exc()[-2500:]
        out["cases"].append(case)
        save()
        torch.cuda.empty_cache()
    out["status"] = "complete"
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        out["status"] = "failed"
        out["error"] = f"{type(ex).__name__}: {ex}"[-800:]
        out["trace"] = traceback.format_exc()[-3000:]
        save()
        raise
