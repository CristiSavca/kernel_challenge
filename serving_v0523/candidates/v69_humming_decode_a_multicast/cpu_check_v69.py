"""Exercise V69 CPU config and model the pinned Humming grouped scheduler."""

import ast
from collections import Counter
from enum import Enum
import itertools
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np


def use_source(path, names, scope):
    nodes = ast.parse(path.read_text()).body
    selected = [node for node in nodes if getattr(node, "name", None) in names]
    assert len(selected) == len(names), names
    exec(compile(ast.Module(selected, type_ignores=[]), str(path), "exec"), scope)


def pick(configs, rows):
    return next(config for _, end, config in configs if rows <= end)


class GemmType(Enum):
    DENSE = "dense"
    INDEXED = "indexed"
    GROUPED_CONTIGUOUS = "grouped_contiguous"


def load_sm90(vendor):
    dtype_sizes = {
        "bfloat16": 16, "float16": 16, "int8": 8, "float8e4m3": 8,
        "float8e5m2": 8, "float8e3m4": 8, "float4e2m1": 4,
    }
    dtypes = SimpleNamespace(**{
        name: SimpleNamespace(num_bits=bits, name=name)
        for name, bits in dtype_sizes.items()
    })
    dtypes.DataType = type(dtypes.bfloat16)
    scope = dict(dtypes=dtypes, GemmType=GemmType, LayerConfig=SimpleNamespace,
                 np=np, math=math)
    use_source(vendor / "tune/base.py", {"DeviceHeuristics"}, scope)
    use_source(vendor / "tune/sm90.py", {"Sm90Heuristics"}, scope)
    return scope["Sm90Heuristics"], dtypes


def config_checks(select, sm90, dtypes):
    configurations = {}
    checks = 0
    for experts, n in itertools.product((32, 64), (4096, 8192)):
        layer = SimpleNamespace(a_dtype=dtypes.bfloat16, b_dtype=dtypes.float4e2m1,
                                num_experts=experts, shape_k=4096, shape_n=n,
                                use_packed_k_layout=False)
        configs = sm90.get_configs(layer, gemm_type=GemmType.GROUPED_CONTIGUOUS)
        configurations[(experts, n)] = configs
        for rows in range(1, 2048 * experts // 256 + 1):
            config = pick(configs, rows)
            original = json.loads(json.dumps(
                config | {"use_stream_k": False, "use_tma_c": False}
            ))
            assert json.loads(select(config, n, False)) == original
            assert json.loads(select(config, n, True)) == original | {"multi_cast_size_a": 2}
            assert "multi_cast_size_a" not in config
            checks += 3

    config = pick(configurations[(32, 8192)], 256)
    for override in (
        {"use_tma": False}, {"use_tma_a": False}, {"use_mbarrier": False},
        {"use_warp_spec": False}, {"num_sms": 131}, {"multi_cast_size_a": 2},
        {"multi_cast_size_b": 2}, {"block_shape": (24, 512, 128)},
    ):
        original = config | override
        updated = json.loads(select(original, 768, True))
        assert updated == json.loads(json.dumps(
            original | {"use_stream_k": False, "use_tma_c": False}
        ))
        checks += 1
    assert json.loads(select(config, 768, True)) == json.loads(json.dumps(
        config | {"use_stream_k": False, "use_tma_c": False}
    ))
    return configurations, checks + 1


def configure_device(path):
    """Execute exact prepare-time guard up to the first packing loop."""
    prepare = next(node for node in ast.parse(path.read_text()).body
                   if getattr(node, "name", None) == "prepare_humming_nvfp4")
    declarations = []
    for node in prepare.body:
        if isinstance(node, ast.For):
            break
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {
                "prepare_grouped_decode", "multicast_a",
            } for target in node.targets
        ):
            declarations.append(node)
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Name) and child.id == "multicast_a" and
            isinstance(child.ctx, ast.Store) for child in ast.walk(node)
        ):
            declarations.append(node)
    code = compile(ast.Module(declarations, type_ignores=[]), str(path), "exec")
    checks = 0
    for group, opt, det, experts, device in itertools.product(
        (False, True), (False, True), (False, True), (32, 64, 65),
        ((9, 0, 132), (9, 0, 131), (9, 0, 1), (9, 0, 2),
         (9, 1, 132), (10, 0, 132), (8, 0, 132)),
    ):
        queries = []
        major, minor, sms = device

        def props(ordinal):
            queries.append(ordinal)
            return SimpleNamespace(major=major, minor=minor, multi_processor_count=sms)

        flags = SimpleNamespace()
        for name, value in (
            ("SGLANG_ENABLE_MUSE_SPARK_HUMMING_GROUPED_DECODE", group),
            ("SGLANG_ENABLE_MUSE_SPARK_HUMMING_DECODE_A_MULTICAST", opt),
            ("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", det),
        ):
            setattr(flags, name, SimpleNamespace(get=lambda value=value: value))
        scope = dict(
            layer=SimpleNamespace(num_local_experts=experts,
                                  gate_up_proj=SimpleNamespace(device="cuda:3")),
            envs=flags, _MAX_DECODE_EXPERTS=64,
            torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=props)),
        )
        exec(code, scope)
        valid = group and not det and experts <= 64
        assert scope["prepare_grouped_decode"] == valid
        assert scope["multicast_a"] == (valid and opt and device in ((9, 0, 132), (9, 0, 2)))
        assert queries == (["cuda:3"] if valid and opt else [])
        checks += 1
    return checks


def work_schedule(counts, slots, block_m, n_tiles, group_size, ctas):
    """Mirror get_next_block and fetch_moe_group_block without Stream-K."""
    first = slots - sum(counts)
    offsets = tuple(itertools.accumulate(counts, initial=first))
    m_tiles = sum((count + block_m - 1) // block_m for count in counts)
    group_columns = n_tiles // group_size
    workers = ctas // group_size
    jobs = Counter()
    loads = 0
    for cta in range(ctas):
        expert = offset = old_m = 0
        for index in range(cta // group_size, m_tiles * group_columns, workers):
            m_tile, n_tile = divmod(index, group_columns)
            n_tile = n_tile * group_size + cta % group_size
            offset += (m_tile - old_m) * block_m
            while offset >= counts[expert]:
                offset -= ((counts[expert] + block_m - 1) // block_m) * block_m
                expert += 1
            old_m = m_tile
            start = offsets[expert] + offset
            valid = min(block_m, offsets[expert + 1] - start)
            assert 0 <= start < start + valid <= slots
            jobs[(expert, start, valid, n_tile)] += 1
            loads += int(cta % group_size == 0)
    return jobs, loads


def scheduler_checks(configs):
    rng = random.Random(69)
    checks = 0
    for experts in (32, 64):
        for rows in (1, 2, 7, 8, 17, 32, 64, 127, 128, 203, 256):
            slots = 8 * rows
            counts = [
                [0] * experts,
                [rows] + [0] * (experts - 1),
                [0] * (experts - 1) + [rows],
                [rows] * 8 + [0] * (experts - 8),
            ]
            for _ in range(16):
                uniform = [0] * experts
                for _ in range(rows):
                    for expert in rng.sample(range(256), 8):
                        if expert < experts:
                            uniform[expert] += 1
                counts.append(uniform)
            for n in (4096, 8192):
                tile_m, tile_n, tile_k = pick(configs[(experts, n)], slots * experts // 256)[
                    "block_shape"
                ]
                assert 4096 % tile_k == 0
                for distribution in counts:
                    base, loads_base = work_schedule(distribution, slots, tile_m, n // tile_n, 1, 132)
                    pair, loads_pair = work_schedule(distribution, slots, tile_m, n // tile_n, 2, 132)
                    assert pair == base and set(pair.values()) <= {1}
                    assert loads_pair * 2 == loads_base
                    for expert, start, valid, column in pair:
                        assert (expert, start, valid, column ^ 1) in pair
                    checks += 1
    return checks


def main():
    source, vendor = map(Path, sys.argv[1:])
    scope = {"json": json}
    use_source(source, {"_decode_tuning_config"}, scope)
    sm90, dtypes = load_sm90(vendor)
    configs, config_count = config_checks(scope["_decode_tuning_config"], sm90, dtypes)
    print(json.dumps({
        "config_assertions": config_count,
        "device_flag_combinations": configure_device(source),
        "paired_grouped_schedule_cases": scheduler_checks(configs),
        "cuda_executed": False,
    }))


if __name__ == "__main__":
    main()
