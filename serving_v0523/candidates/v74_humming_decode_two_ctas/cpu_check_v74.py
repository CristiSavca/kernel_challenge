"""Check the exact V74 helper with pinned Humming configuration/resource bodies."""

import argparse
import ast
from collections import Counter
import dataclasses
import itertools
import json
import math
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np


def selected(path, names, scope):
    nodes = ast.parse(path.read_text()).body
    found = [node for node in nodes if getattr(node, "name", None) in names]
    assert len(found) == len(names)
    exec(compile(ast.Module(found, type_ignores=[]), str(path), "exec"), scope)


def import_vendor_bodies(vendor):
    enums = {}
    exec((vendor / "config/enum.py").read_text(), enums)
    dtype_values = {
        "bfloat16": 16, "float16": 16, "int8": 8, "float8e4m3": 8,
        "float8e5m2": 8, "float8e3m4": 8, "float4e2m1": 4,
    }
    dtypes = SimpleNamespace(**{
        name: SimpleNamespace(num_bits=bits, name=name)
        for name, bits in dtype_values.items()
    })
    dtypes.DataType = type(dtypes.bfloat16)
    scope = dict(
        dataclasses=dataclasses, ClassVar=ClassVar, math=math, dtypes=dtypes,
        BaseHummingConfig=object,
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (9, 0))),
        **{name: enums[name] for name in ("GemmType", "MmaType")},
    )
    selected(vendor / "config/config.py", {"ComputeConfig", "TuningConfig"}, scope)
    scope.update(LayerConfig=SimpleNamespace, np=np)
    selected(vendor / "tune/base.py", {"DeviceHeuristics"}, scope)
    selected(vendor / "tune/sm90.py", {"Sm90Heuristics"}, scope)
    functions = ast.parse((vendor / "utils/smem.py").read_text())
    functions.body = [node for node in functions.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(functions, str(vendor / "utils/smem.py"), "exec"), scope)

    config_module = ModuleType("humming.config")
    for name in ("ComputeConfig", "TuningConfig", "GemmType"):
        setattr(config_module, name, scope[name])
    smem_module = ModuleType("humming.utils.smem")
    smem_module.estimate_smem_size_config = scope["estimate_smem_size_config"]
    sys.modules["humming.config"] = config_module
    sys.modules["humming.utils.smem"] = smem_module
    return scope


def metadata(dtypes, mma_type, experts, n):
    return SimpleNamespace(
        a_dtype=dtypes.bfloat16, b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16, bs_dtype=dtypes.float8e4m3,
        num_experts=experts, shape_k=4096, shape_n=n, use_packed_k_layout=False,
        weight_scale_group_size=16, input_scale_group_size=0,
        is_group_weight_scale=True, is_block_weight_scale=False,
        is_channel_weight_scale=False, is_channel_weight_scale_2=False,
        has_zero_point=False, is_fp_zero_point=False, has_bias=False,
        mma_type=mma_type,
    )


def device(**overrides):
    return SimpleNamespace(**dict(
        max_threads_per_multi_processor=2048,
        shared_memory_per_multiprocessor=233472,
        regs_per_multiprocessor=65536,
    ) | overrides)


def resources(config, meta, scope):
    names = {field.name for field in dataclasses.fields(scope["TuningConfig"])}
    params = {key: value for key, value in config.items() if key in names}
    tuning = scope["TuningConfig"](**params)
    compute = scope["ComputeConfig"](
        gemm_type=scope["GemmType"].GROUPED_CONTIGUOUS, use_f16_accum=False
    )
    smem = scope["estimate_smem_size_config"](meta, compute, tuning)
    ctas = tuning.num_ctas_per_sm
    warps = tuning.num_threads // 32
    loader_regs = 40 if tuning.num_math_threads > 256 else 24
    budget = ((65536 // (warps * ctas)) // 256 * 256) * warps
    available = budget - tuning.num_load_threads * loader_regs
    preferred = 96 if tuning.num_math_threads > 256 else 232
    math_regs = min(preferred, max(24, available // tuning.num_math_threads // 8 * 8))
    return {
        "block": tuning.block_shape,
        "warp": tuning.warp_shape,
        "ctas_per_sm": ctas,
        "threads_per_cta": tuning.num_threads,
        "shared_bytes_per_cta": smem,
        "loader_registers": loader_regs,
        "math_registers": math_regs,
        "cta_registers": tuning.num_load_threads * loader_regs
        + tuning.num_math_threads * math_regs,
    }


def schedules(counts, block_m, n_tiles, ctas, multicast):
    tiles = []
    for expert, count in enumerate(counts):
        tiles.extend((expert, start, min(block_m, count - start))
                     for start in range(0, count, block_m))
    groups = ctas // multicast
    columns = n_tiles // multicast
    seen = Counter()
    for cta in range(ctas):
        for index in range(cta // multicast, len(tiles) * columns, groups):
            row, column = divmod(index, columns)
            seen[(*tiles[row], multicast * column + cta % multicast)] += 1
    return seen


def prepare_guards(source):
    nodes = ast.parse(source.read_text()).body
    prepare = next(node for node in nodes if getattr(node, "name", None) == "prepare_humming_nvfp4")
    selected_nodes = []
    for node in prepare.body:
        if isinstance(node, ast.For):
            break
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {
                "prepare_grouped_decode", "multicast_a", "resident_device",
            } for target in node.targets
        ):
            selected_nodes.append(node)
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Name) and child.id == "resident_device"
            and isinstance(child.ctx, ast.Store) for child in ast.walk(node)
        ):
            selected_nodes.append(node)
    code = compile(ast.Module(selected_nodes, type_ignores=[]), str(source), "exec")
    checked = 0
    for grouped, multicast, two_ctas, deterministic, experts, capability, sms in itertools.product(
        (False, True), (False, True), (False, True), (False, True),
        (32, 64, 65), ((8, 0), (9, 0), (9, 1), (10, 0)), (1, 2, 131, 132),
    ):
        flags = SimpleNamespace()
        for name, value in (
            ("SGLANG_ENABLE_MUSE_SPARK_HUMMING_GROUPED_DECODE", grouped),
            ("SGLANG_ENABLE_MUSE_SPARK_HUMMING_DECODE_A_MULTICAST", multicast),
            ("SGLANG_ENABLE_MUSE_SPARK_HUMMING_DECODE_TWO_CTAS", two_ctas),
            ("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", deterministic),
        ):
            setattr(flags, name, SimpleNamespace(get=lambda value=value: value))
        calls = []
        props = SimpleNamespace(major=capability[0], minor=capability[1],
                                multi_processor_count=sms)

        def get_device(device_id):
            calls.append(device_id)
            return props

        scope = dict(
            envs=flags, _MAX_DECODE_EXPERTS=64,
            layer=SimpleNamespace(num_local_experts=experts,
                                  gate_up_proj=SimpleNamespace(device="cuda:5")),
            torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=get_device)),
        )
        exec(code, scope)
        enabled = grouped and not deterministic and experts <= 64
        assert scope["prepare_grouped_decode"] == enabled
        assert scope["multicast_a"] == (
            enabled and multicast and capability == (9, 0) and sms >= 2 and sms % 2 == 0
        )
        assert (scope["resident_device"] is props) == (
            enabled and two_ctas and capability == (9, 0)
        )
        assert calls == (["cuda:5"] if enabled and (multicast or two_ctas) else [])
        checked += 1
    return checked


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("source", type=Path)
    args.add_argument("vendor", type=Path)
    options = args.parse_args()
    scope = import_vendor_bodies(options.vendor)
    helper_scope = {"json": json}
    selected(options.source, {"_decode_tuning_config"}, helper_scope)
    choose = helper_scope["_decode_tuning_config"]
    count = Counter()
    count["prepare_guard_cases"] = prepare_guards(options.source)
    resource_rows = {}
    rng = random.Random(74)
    for experts, n in itertools.product((32, 64), (4096, 8192)):
        meta = metadata(scope["dtypes"], scope["MmaType"].WGMMA, experts, n)
        configs = scope["Sm90Heuristics"].get_configs(
            meta, gemm_type=scope["GemmType"].GROUPED_CONTIGUOUS
        )
        for rows in range(1, 2048 * experts // 256 + 1):
            config = next(config for _, stop, config in configs if rows <= stop)
            for multicast in (False, True):
                previous = json.loads(choose(config, n, multicast))
                selected_config = json.loads(choose(
                    config, n, multicast, layer_meta=meta, resident_device=device()
                ))
                before = resources(previous, meta, scope)
                after = resources(selected_config, meta, scope)
                eligible = before["shared_bytes_per_cta"] * 2 <= 233472
                expected = previous | {"num_ctas_per_sm": 2} if eligible else previous
                assert selected_config == expected, (
                    experts, n, rows, before, selected_config,
                )
                assert after["block"] == before["block"] and after["warp"] == before["warp"]
                assert after["ctas_per_sm"] * after["cta_registers"] <= 65536
                assert after["ctas_per_sm"] * after["threads_per_cta"] <= 2048
                assert after["ctas_per_sm"] * after["shared_bytes_per_cta"] <= 233472
                assert selected_config["use_stream_k"] is False
                assert selected_config.get("use_f16_accum", False) is False
                key = (experts, n, before["block"][0], bool(multicast))
                resource_rows[key] = dict(experts=experts, n=n, multicast=multicast,
                                          before=before, after=after)
                count["config_comparisons"] += 1
                for limits in (
                    {"max_threads_per_multi_processor": 1279},
                    {"shared_memory_per_multiprocessor": before["shared_bytes_per_cta"] * 2 - 1},
                    {"regs_per_multiprocessor": 65535},
                ):
                    assert json.loads(choose(config, n, multicast, layer_meta=meta,
                                             resident_device=device(**limits))) == previous
                    count["resource_rejections"] += 1
        for _, stop, config in configs:
            if stop > 1024:
                break
            previous = json.loads(choose(config, n, True))
            block_m, block_n, _ = previous["block_shape"]
            for _ in range(80):
                counts = [rng.randrange(0, 2 * stop + 1) for _ in range(experts)]
                a = schedules(counts, block_m, n // block_n, 132, 2)
                b = schedules(counts, block_m, n // block_n, 264, 2)
                assert a == b and set(a.values()) <= {1}
                count["tile_ownership_comparisons"] += 1
    print(json.dumps({"checks": dict(count), "resources": list(resource_rows.values()),
                      "gpu_executed": False}, indent=2))


if __name__ == "__main__":
    main()
