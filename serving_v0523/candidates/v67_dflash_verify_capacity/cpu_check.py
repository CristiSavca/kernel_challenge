"""Exercise the patched DFLASH allocation and FA3/FA4 graph buffer geometry."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace


def extract(path, names, env, class_name=None):
    tree = ast.parse(path.read_text(), str(path))
    scope = tree.body
    if class_name:
        scope = next(node.body for node in scope if getattr(node, "name", None) == class_name)
    tree.body = [node for node in scope if getattr(node, "name", None) in names]
    tree.body.insert(0, ast.ImportFrom("__future__", [ast.alias("annotations")], 0))
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), env)
    return [env[name] for name in names]


class SpeculativeAlgorithm:
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_string(cls, name):
        return cls(name)

    def is_dflash(self):
        return self.name == "DFLASH"

    def is_uno(self):
        return self.name == "UNO"

    def has_draft_kv(self):
        return self.name != "NGRAM"


@dataclass
class Tensor:
    shape: tuple[int, ...]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        result = tuple(
            len(range(*index.indices(size)))
            for index, size in zip(key, self.shape)
            if isinstance(index, slice)
        )
        return Tensor(result + self.shape[len(key):])


def ceil_div(n, d):
    return (n + d - 1) // d


def verify_builder_calls(path):
    builders = {
        "_init_unified_mamba_pools",
        "_init_unified_mamba_swa_pools",
        "_build_req_to_token_pool",
    }
    seen = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef) or node.name not in builders:
            continue
        matches = [
            statement
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and [ast.unparse(target) for target in statement.targets] == ["extra_max_context_len"]
        ]
        assert len(matches) == 1
        shared = node.name == "_build_req_to_token_pool"
        if not shared:
            matches.extend(
                statement
                for statement in node.body
                if isinstance(statement, ast.If)
                and ast.unparse(statement.test) == "self.spec_algorithm.is_dflash()"
            )
            assert len(matches) == 2
        program = compile(ast.Module(body=matches, type_ignores=[]), str(path), "exec")
        for algorithm, expected in (("DFLASH", 99), ("EAGLE", 99 if shared else 12)):
            namespace = {
                "self": SimpleNamespace(spec_algorithm=SpeculativeAlgorithm(algorithm)),
                "get_spec": lambda: SimpleNamespace(speculative_num_draft_tokens=8),
                "get_req_to_token_extra_context_len": lambda: 99,
            }
            exec(program, namespace)
            assert namespace["extra_max_context_len"] == expected
        assert any(
            isinstance(child, ast.keyword)
            and child.arg == "extra_max_context_len"
            and ast.unparse(child.value) == "extra_max_context_len"
            for child in ast.walk(node)
        )
        seen.add(node.name)
    assert seen == builders


def run(source_root):
    root = source_root / "python/sglang/srt"
    spec_module = ModuleType("sglang.srt.speculative.spec_info")
    spec_module.SpeculativeAlgorithm = SpeculativeAlgorithm
    sys.modules[spec_module.__name__] = spec_module
    spec = SimpleNamespace(speculative_algorithm="DFLASH", speculative_eagle_topk=1,
                           speculative_num_steps=1)
    schedule = SimpleNamespace()
    parallel = SimpleNamespace()
    alloc_names = (
        "get_alloc_page_size", "get_alloc_len_per_decode", "get_alloc_reserve_per_decode",
        "get_req_to_token_extra_context_len", "page_aligned_decode_alloc_lens",
    )
    _, _, _, headroom, allocate = extract(
        root / "mem_cache/allocation_sizing.py", alloc_names,
        dict(
            get_spec=lambda: spec, get_schedule=lambda: schedule, get_parallel=lambda: parallel,
            max_speculative_num_draft_tokens=lambda: max(
                spec.speculative_num_draft_tokens, spec.capacity
            ),
        ),
    )
    verify_builder_calls(root / "mem_cache/kv_cache_configurator.py")

    def zeros(*shape, **_):
        return Tensor(shape[0] if isinstance(shape[0], tuple) else shape)

    def fill_tables(*args, **kwargs):
        page_table, max_columns = args[2], args[5]
        seq_lens, delta, page_size, swa_table = args[6:10]
        assert max_columns == page_table.shape[1]
        assert len(seq_lens) == page_table.shape[0]
        assert all(ceil_div(length + delta, page_size) <= max_columns for length in seq_lens)
        if swa_table is not None:
            assert swa_table.shape == page_table.shape

    torch = SimpleNamespace(
        zeros=zeros, arange=lambda start, stop, step=1, **_: Tensor((len(range(start, stop, step)),)),
        int32="int32", int64="int64",
    )
    graph_init, graph_bind, graph_replay, graph_capture = extract(
        root / "layers/attention/flashattention_backend.py",
        ("init_cuda_graph_state", "_bind_metadata_buffers", "_set_decode_page_metadata",
         "init_forward_metadata_in_graph"),
        dict(
            torch=torch, get_spec=lambda: spec, SpeculativeAlgorithm=SpeculativeAlgorithm,
            FlashAttentionMetadata=lambda: SimpleNamespace(swa_page_table=None),
            maybe_create_verify_mask=lambda **_: None, normal_decode_set_metadata=fill_tables,
            IdSpaceKind=SimpleNamespace(FULL="full", SLIDING_WINDOW="swa"),
        ),
        "FlashAttentionBackend",
    )
    residues = 0
    graph_cases = 0
    mode = SimpleNamespace(is_decode_or_idle=lambda: False, is_target_verify=lambda: True)
    for context in (32, 2047, 2048, 2049, 131_071, 131_072, 131_073, 1_048_576):
        for block in range(1, 33):
            spec.speculative_num_draft_tokens = spec.capacity = block
            for page_size in (1, 2, 3, 8, 16, 32, 64, 128, 160, 256, 640):
                schedule.page_size = page_size
                for parallel.attn_dcp_size in (1, 2, 4):
                    alloc_page = page_size * parallel.attn_dcp_size
                    row_width = context + headroom()
                    assert row_width == context + max(4 + block, 2 * block + alloc_page - 1)
                    # Every allocation-page residue in the context tail. The
                    # previous verify may publish B extra tokens on device.
                    prefixes = range(max(0, context - alloc_page - 2 * block), context + 1)
                    reqs = [
                        SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=0, kv_committed_len=x))
                        for x in prefixes
                    ]
                    _, ends, _ = allocate(reqs, reserve=2 * block, page_size=alloc_page)
                    for prefix, alloc_end in zip(prefixes, ends):
                        read_end = prefix + 2 * block
                        assert read_end <= alloc_end <= row_width
                        last_page = (read_end - 1) // page_size
                        assert last_page * page_size < alloc_end
                        assert last_page < ceil_div(row_width, page_size)
                        residues += 1
                    for swa in (False, True):
                        legacy_pages = ceil_div(context, page_size)
                        backend = SimpleNamespace(
                            max_context_len=context, page_size=page_size, max_num_pages=legacy_pages,
                            device="cpu", req_to_token=Tensor((32, row_width)),
                            _get_scheduler_metadata=None, local_attn_builder=None,
                            use_sliding_window_kv_pool=swa, use_mla=False, topk=1,
                            speculative_num_draft_tokens=block, is_draft_runner=False,
                            skip_prefill=False, is_encoder_decoder=False,
                        )
                        graph_init(backend, 32, 32 * block)
                        verify_pages = ceil_div(row_width, page_size)
                        assert backend.target_verify_metadata["page_table"].shape == (32, verify_pages)
                        assert backend.target_verify_metadata["strided_indices"].shape == (verify_pages,)
                        assert backend.decode_cuda_graph_metadata["page_table"].shape == (32, legacy_pages)
                        assert backend.draft_extend_metadata["page_table"].shape == (32, legacy_pages)
                        backend.token_to_kv_pool = SimpleNamespace(full_to_swa_index_mapping=None)
                        backend.kv_index_translator = SimpleNamespace(
                            reads_are_translated=False, space=lambda _: SimpleNamespace(read_v2p=None)
                        )
                        backend._capture_unified_verify_metadata = lambda _: True
                        # Check an unpadded graph and a 32-slot graph with
                        # one real request, in the ordinary and unified builders.
                        for bucket in (1, 32):
                            meta, _ = graph_bind(backend, bucket, bucket * block, None, mode, None, "cpu")
                            assert meta.page_table.shape == (bucket, verify_pages)
                            lengths = [context + block] + [1] * (bucket - 1)
                            graph_replay(backend, None, meta, None, lengths, block)
                            graph_capture(
                                backend, SimpleNamespace(batch_size=bucket,
                                                         req_pool_indices=None, seq_lens=lengths)
                            )
                            graph_cases += 1
    spec.speculative_num_draft_tokens = 8
    spec.capacity = 16
    schedule.page_size = parallel.attn_dcp_size = 1
    assert headroom() == 32
    for algorithm in ("EAGLE", "NGRAM", "UNO"):
        spec.speculative_algorithm = algorithm
        graph_init(backend, 32, 256)
        assert backend.target_verify_metadata["page_table"].shape == (32, legacy_pages)
    assert headroom() == 34  # UNO holds one additional position in both blocks.
    print(f"Passed {residues:,} allocator tail checks and {graph_cases:,} verified graph shapes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="SGLang checkout with V67 applied")
    run(parser.parse_args().source_root)
