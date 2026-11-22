"""Compile selected decode Triton specializations offline from an applied SGLang tree.

Run on Linux with Triton installed. The generated modules contain the checkout's
verbatim JIT definitions, allowing compilation without importing the serving runtime.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback


def fixtures():
    cases = []

    def add(file, kernel, label, types, constants, warps=4, stages=3):
        cases.append(dict(file=file, kernel=kernel, label=label, types=types.split(),
                          constants=constants, num_warps=warps, num_stages=stages))

    tree = "kernels/ops/speculative/dflash_tree.py"
    attention = "kernels/ops/attention/paged_verify_attention.py"
    staged = "srt/layers/muse_spark/muse_spark_humming_moe.py"
    for width in (8, 12):
        n = 1 << (width - 1).bit_length()
        for k in (4, 8, 16):
            add(tree, "_propose_tree", f"width{width}-k{k}", "*i64 *fp32 " + "*i64 " * 5,
                dict(ANCHOR_STRIDE=1, WIDTH=width, K=k, BLOCK=n*k))
        add(tree, "_accept_tree", f"width{width}", "*i64 *i64 *i64 *i32 *i32 *i64 *i32 *i32 *i64",
            dict(WIDTH=width, BLOCK=n), warps=1)
        add(tree, "_reorder_slots", f"width{width}", "*i32 *i64 *i32 *i64 *i64 *i64",
            dict(ROW_STRIDE=131072, WIDTH=width, BLOCK=n), warps=1)
        for window, heads, gqa in ((-1, 2, 4), (512, 4, 4)):
            m = 1 << (width * gqa - 1).bit_length()
            partitions = math.ceil((264 if window < 0 else 132) / heads)
            tile_n = 128 if window < 0 else 64
            common = dict(B=16, H_KV=heads, GQA=gqa, D=64, WINDOW_LEFT=window,
                          TILE_M=m, TILE_N=tile_n, PARTITIONS=partitions, BLOCK_B=16)
            for tree_mask in (True, False):
                case = f"width{width}-window{window}-tree{int(tree_mask)}"
                opt = "*i64 *i64 " if tree_mask else ""
                types = "*fp8e4nv *fp8e4nv *fp8e4nv *i32 *i32 *i32 *fp32 " + opt + "*fp32 *fp32"
                const = dict(common, MAX_Q=width, PAGE_ROW_STRIDE=131072, PAGE_SIZE=1,
                             K_PAGE=heads*64, K_TOKEN=heads*64, K_HEAD=64,
                             V_PAGE=heads*64, V_TOKEN=heads*64, V_HEAD=64,
                             KS_ROW=0, KS_HEAD=0, SCALE_LOG2=math.log2(math.e)/8)
                if not tree_mask:
                    const.update(TreeDepths=None, TreeAncestors=None)
                add(attention, "_partition_attention", case, types, const)
            add(attention, "_merge_partitions", f"width{width}-window{window}",
                "*fp32 *fp32 *i32 *i32 *fp32 *bf16", dict(common, VS_ROW=0, VS_HEAD=0, BLOCK_SPLITS=16))
    add("kernels/ops/communication/muse_spark_router_gather.py", "_unpack", "tp8",
        "*i32 *i32 *i32 i32", dict(TP=8, POOL_WORDS=256, EXPERTS=32, BLOCK=128))
    ctx = "kernels/ops/speculative/dflash_committed_projection.py"
    add(ctx, "_plan_rows", "width8", "*i64 *i32 *i32 i32", dict(WIDTH=8, BLOCK=256))
    add(ctx, "_project", "16x32x64", "*bf16 *bf16 *i32 *bf16 i32",
        dict(N=1024, K=40960, BM=16, BN=32, BK=64))
    for slots in (64, 1024, 2048):
        common = dict(TOPK=8, HIDDEN=4096)
        add(staged, "_stage_token_routes", f"slots{slots}",
            "*i32 *i32 *bf16 *fp32 *i32 *i32 *i32 *i32 *bf16 fp32",
            dict(common, SLOTS=slots, GLOBAL=256, EXPERTS=32, HAS_MAP=True, K_TILE=8,
                 SPAN=slots, BUCKETS=64, WIDTH=4096), warps=8)
        add(staged, "_stage_grouped_decode", f"slots{slots}",
            "*i32 *i32 *bf16 *i32 *i32 *i32 *i32 *bf16 i32",
            dict(common, GLOBAL_EXPERTS=256, EXPERTS=32, HAS_MAP=True,
                 SLOT_SPAN=slots, TILE=min(128, slots), ROWS=4, WIDTH=4096), warps=8)
        add(staged, "_plan_grouped_decode", f"slots{slots}", "*i32 " * 6 + "i32",
            dict(EXPERTS=32, GLOBAL_EXPERTS=256, HAS_EXPERT_MAP=True,
                 SLOT_SPAN=slots, EXPERT_SPAN=64), warps=8 if slots > 256 else 4)
    return cases


def load_jit_module(path, work):
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    jit_defs = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and any(
            ast.unparse(dec.func if isinstance(dec, ast.Call) else dec) == "triton.jit"
            for dec in node.decorator_list
        ):
            jit_defs.extend(lines[node.decorator_list[0].lineno - 1:node.end_lineno])
            jit_defs.append("\n")
    name = "offline_" + path.stem
    module_path = work / (name + ".py")
    module_path.write_text("import triton\nimport triton.language as tl\n" + "".join(jit_defs))
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--extra", type=Path, help="JSON array of additional compile fixtures")
    parser.add_argument("--only-extra", action="store_true")
    args = parser.parse_args()
    cases = [] if args.only_extra else fixtures()
    if args.extra:
        extra = json.loads(args.extra.read_text())
        if isinstance(extra, dict):
            extra = extra["cases"]
        keys = ("file", "kernel", "label", "types", "constants", "num_warps", "num_stages", "aligned")
        cases.extend({key: row[key] for key in keys if key in row} for row in extra)
    ptxas = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    receipt = dict(triton=triton.__version__, target="sm90", cuda_execution=False,
                   assembler=subprocess.check_output([ptxas, "--version"], text=True).splitlines()[-2],
                   cases=[])
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        modules = {}
        for case in cases:
            path = args.checkout / "python/sglang" / case["file"]
            row = {**case, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            receipt["cases"].append(row)
            try:
                if path not in modules:
                    modules[path] = load_jit_module(path, work)
                fn = getattr(modules[path], case["kernel"])
                const = case["constants"]
                assert set(const) <= set(fn.arg_names)
                dynamic = [arg for arg in fn.arg_names if arg not in const]
                signature = dict(zip(dynamic, case["types"], strict=True))
                signature.update({key: "constexpr" for key in const})
                options = {key: case[key] for key in ("num_warps", "num_stages")}
                attrs = {(fn.arg_names.index(arg),): [["tt.divisibility", 16]]
                         for arg in case.get("aligned", [])}
                binary = triton.compile(ASTSource(fn, signature, constexprs=const, attrs=attrs),
                                        target=GPUTarget("cuda", 90, 32), options=options)
                cubin = binary.asm["cubin"]
                row.update(cubin_sha256=hashlib.sha256(cubin).hexdigest(), cubin_bytes=len(cubin),
                           shared_bytes=binary.metadata.shared)
                ptx = work / "kernel.ptx"
                ptx.write_text(binary.asm["ptx"])
                asm = subprocess.run([ptxas, "-arch=sm_90a", "-v", str(ptx), "-o", str(work / "probe.cubin")],
                                     capture_output=True, text=True, check=True)
                row["assembler_stats"] = [line.strip() for line in asm.stderr.splitlines()
                                          if re.search("spill stores|Used [0-9]+ registers", line)]
            except Exception:
                row["error"] = traceback.format_exc()[-3500:]
            print(case["kernel"], case["label"], "failed" if "error" in row else "compiled",
                  file=sys.stderr, flush=True)
    print(json.dumps(receipt, indent=2))
    return int(any("error" in row for row in receipt["cases"]))


if __name__ == "__main__":
    sys.exit(main())
