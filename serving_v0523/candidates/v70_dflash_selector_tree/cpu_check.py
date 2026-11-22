"""CPU execution of V70's Triton bodies with masked array pointers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Array(np.ndarray):
    def to(self, dtype):
        if dtype == "bfloat16":
            floats = self.astype(np.float32)
            words = floats.view(np.uint32)
            rounded = words + ((words >> 16) & 1) + 0x7FFF
            return array((rounded & 0xFFFF0000).view(np.float32))
        return array(self.astype(dtype))

    def copy_(self, other):
        np.copyto(self, other, casting="unsafe")


def array(value):
    return np.asarray(value).view(Array)


class Pointer:
    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset

    def __add__(self, delta):
        return Pointer(self.data, self.offset + delta)

    def __sub__(self, delta):
        return Pointer(self.data, self.offset - delta)


def load(pointer, mask=True, other=0):
    offset, mask = np.broadcast_arrays(pointer.offset, mask)
    active = offset[mask]
    assert np.all((active >= 0) & (active < pointer.data.size))
    out = np.full(offset.shape, other, dtype=pointer.data.dtype)
    out[mask] = pointer.data.flat[active]
    return array(out)


def store(pointer, value, mask=True):
    offset, value, mask = np.broadcast_arrays(pointer.offset, value, mask)
    active = offset[mask]
    assert np.all((active >= 0) & (active < pointer.data.size))
    pointer.data.flat[active] = value[mask]


def operation(function):
    def call(*args, **kwargs):
        return array(function(*args, **kwargs))
    return call


def dot(left, right, acc=0, **_):
    return array(left @ right + acc)


def program(path, names, scope_name=None, dependencies=None):
    module = ast.parse(path.read_text(), str(path))
    scope = module.body
    if scope_name is not None:
        scope = next(node.body for node in scope if getattr(node, "name", None) == scope_name)
    module.body = [node for node in scope if getattr(node, "name", None) in names]
    assert len(module.body) == len(names)
    module.body.insert(0, ast.ImportFrom("__future__", [ast.alias("annotations")], 0))
    ast.fix_missing_locations(module)
    pid = [0, 0, 0]
    tl = SimpleNamespace(
        program_id=lambda axis: pid[axis],
        int32=np.int32,
        int64=np.int64,
        float32=np.float32,
        bfloat16="bfloat16",
        arange=operation(np.arange),
        full=operation(np.full),
        zeros=operation(np.zeros),
        where=operation(np.where),
        sum=operation(np.sum),
        max=operation(np.max),
        min=operation(np.min),
        argmax=lambda arr, axis, **_: np.argmax(arr, axis=axis),
        maximum=operation(np.maximum),
        minimum=operation(np.minimum),
        cdiv=lambda x, d: (x + d - 1) // d,
        cumsum=operation(np.cumsum),
        exp=operation(np.exp),
        exp2=operation(np.exp2),
        log2=operation(np.log2),
        load=load,
        store=store,
        dot=dot,
        range=lambda a, b, **_: range(a, b),
    )
    namespace = {"triton": SimpleNamespace(jit=lambda function: function), "tl": tl}
    if dependencies is not None:
        namespace.update(dependencies)
    exec(compile(module, str(path), "exec"), namespace)
    return pid, [namespace[name] for name in names]


def proposal_oracle(candidates, scores, anchor):
    depth_count, k = candidates.shape
    heap = []
    tokens, depths, parents, ancestors = [int(anchor)], [0], [-1], [1]

    def add_edges(parent, depth, rank, mass):
        row = np.nan_to_num(scores[depth, rank], nan=-np.inf, neginf=-np.inf, posinf=np.inf)
        peak = row.max()
        weights = np.exp(row - peak) if np.isfinite(peak) else row == peak
        for index, probability in enumerate(weights / weights.sum()):
            heapq.heappush(heap, (-mass * probability, parent * k + index, depth))

    add_edges(0, 0, 0, 1.0)
    for slot in range(1, depth_count + 1):
        mass, edge, depth = heapq.heappop(heap)
        parent, rank = divmod(edge, k)
        tokens.append(candidates[depth, rank])
        depths.append(depth + 1)
        parents.append(parent)
        ancestors.append(ancestors[parent] | (1 << slot))
        if depth + 1 < depth_count:
            add_edges(slot, depth + 1, rank, -mass)
    return np.array([tokens, depths, parents, ancestors], dtype=np.int64)


def walk_oracle(tree, predictions, prefix):
    tokens, _, parents, _ = tree
    accepted = [0]
    out = []
    while True:
        leader = accepted[-1]
        predicted = predictions[leader]
        out.append(predicted)
        child = [i for i, p in enumerate(parents) if p == leader and tokens[i] == predicted]
        if not child:
            break
        accepted.append(child[0])
    order = accepted + [i for i in range(len(tokens)) if i not in accepted]
    return len(out), out[-1], prefix + len(out), out, order


def verify_tree_kernels(path, rng):
    pid, (_, propose, accept, reorder) = program(
        path, ("_edge_probabilities", "_propose_tree", "_accept_tree", "_reorder_slots")
    )
    plans, walks = 0, 0
    for width in range(2, 9):
        for k in (2, 3, 7, 16, 32):
            for bs in (1, 4):
                for case in range(8):
                    candidate = rng.integers(1, 1024, size=(bs, width - 1, k))
                    scores = rng.normal(size=(bs, width - 1, k, k)).astype(np.float32)
                    if case == 0:
                        scores.fill(0)
                    elif case == 1:
                        scores.fill(-np.inf)
                    elif case == 2:
                        scores[..., 0] = np.inf
                    elif case == 3:
                        scores[..., -1] = np.nan
                    anchors = rng.integers(1, 1024, size=(bs, width))
                    tree = np.zeros((4, bs, width), dtype=np.int64)
                    for row in range(bs):
                        pid[0] = row
                        propose(
                            Pointer(candidate), Pointer(scores), Pointer(anchors),
                            *[Pointer(x) for x in tree], width, width, k,
                            1 << (width * k - 1).bit_length(),
                        )
                        expected = proposal_oracle(candidate[row], scores[row], anchors[row, 0])
                        np.testing.assert_array_equal(tree[:, row], expected)
                        plans += 1
                    for mode in ("root_mismatch", "every_leaf", "duplicate_siblings"):
                        predict = np.zeros((bs, width), dtype=np.int64)
                        accept_tree = tree.copy()
                        if mode == "duplicate_siblings":
                            accept_tree[0].fill(12)
                            predict.fill(12)
                        num_walks = width if mode == "every_leaf" else 1
                        for leaf in range(num_walks):
                            prefix = rng.integers(1, 39, size=bs)
                            if mode == "every_leaf":
                                predict.fill(0)
                                for row in range(bs):
                                    node = leaf
                                    while node:
                                        parent = tree[2, row, node]
                                        predict[row, parent] = tree[0, row, node]
                                        node = parent
                            commit = np.zeros(bs, np.int32)
                            bonus = np.zeros(bs, np.int64)
                            new_lens = np.zeros(bs, np.int64)
                            output = np.full((bs, width), -7, np.int64)
                            order = np.full((bs, width), -7, np.int64)
                            slots = rng.permutation(bs * width).reshape(bs, width) + 1
                            request_order = rng.permutation(bs)
                            req_to_token = np.full((bs, 53), -1, np.int32)
                            remap = np.full_like(slots, -1)
                            for row in range(bs):
                                pid[0] = row
                                block = 1 << (width - 1).bit_length()
                                accept(*[Pointer(x) for x in (accept_tree[0], accept_tree[2], predict,
                                       prefix, commit, bonus, output, new_lens, order)], width, block)
                                n, next_id, size, tokens, slot_order = walk_oracle(
                                    accept_tree[:, row], predict[row], prefix[row]
                                )
                                assert (commit[row], bonus[row], new_lens[row]) == (n, next_id, size)
                                np.testing.assert_array_equal(output[row], tokens + [0] * (width - n))
                                np.testing.assert_array_equal(order[row], slot_order)
                                reorder(*[Pointer(x) for x in (req_to_token, request_order, prefix,
                                        slots, order, remap)], req_to_token.shape[1], width, block)
                                expected = slots[row, slot_order]
                                np.testing.assert_array_equal(remap[row], expected)
                                table_row = req_to_token[request_order[row]]
                                np.testing.assert_array_equal(table_row[prefix[row]:prefix[row]+width], expected)
                                assert (table_row[:prefix[row]] == -1).all()
                                assert (table_row[prefix[row]+width:] == -1).all()
                                walks += 1
    return plans, walks


def verify_attention(path, rng):
    pid, (_, _, partition, merge) = program(
        path, ("_tile_geometry", "_plan", "_partition_attention", "_merge_partitions")
    )
    checked, max_l2 = 0, 0.0
    digest = hashlib.sha256()
    for width in (4, 8):
        for bs in (1, 2, 4):
            for window in (-1, 7, 64, 2047):
                for splits in (1, 3, 13):
                    groups, kv_heads, dim, tile_n = 8, 1, 64, 64
                    tile_m = groups * width
                    q = rng.normal(size=(bs * width, groups, dim)).astype(np.float32)
                    prefixes = [0, 1, 63, 64, 65, 128, 129, 2049]
                    lengths = rng.choice(prefixes, size=bs) + width
                    table = rng.permutation(bs * lengths.max()).reshape(bs, -1).astype(np.int32)
                    k = rng.normal(size=(table.size, dim)).astype(np.float32)
                    v = array(rng.normal(size=k.shape)).to("bfloat16")
                    cuq = np.arange(0, (bs+1)*width, width, dtype=np.int32)
                    probs = rng.normal(size=(width-1, 4, 4)).astype(np.float32)
                    candidate = np.arange((width-1)*4).reshape(width-1, 4)
                    tree = np.stack([proposal_oracle(candidate, probs + i * 0.17, 999)
                                     for i in range(bs)])
                    depths = np.ascontiguousarray(tree[:, 1], dtype=np.int32)
                    ancestry = np.ascontiguousarray(tree[:, 3], dtype=np.int32)
                    partial = np.full((splits+bs-1, kv_heads, tile_m, dim), np.nan, np.float32)
                    lse = np.full(partial.shape[:-1], np.nan, np.float32)
                    out = np.full(q.shape, np.nan, np.float32)
                    scale = 0.125 * np.log2(np.e)
                    bpad = 1 << (bs - 1).bit_length()
                    for part in range(splits):
                        pid[:] = [part, 0, 0]
                        partition(
                            *[Pointer(x) for x in (q, k, v, table, lengths, cuq)],
                            None, Pointer(depths), Pointer(ancestry), Pointer(partial), Pointer(lse),
                            table.shape[1], 1, dim, dim, dim, dim, dim, dim, 0, 0,
                            bs, kv_heads, groups, dim, scale, window, width, tile_m, tile_n, splits, bpad,
                        )
                    for row in range(bs):
                        for packed in range(tile_m):
                            pid[:] = [row, 0, packed]
                            merge(
                                *[Pointer(x) for x in (partial, lse, lengths, cuq)], None,
                                Pointer(out), 0, 0, bs, kv_heads, groups, dim,
                                window, tile_m, tile_n, splits, bpad, 16,
                            )
                    reference = np.empty_like(out)
                    for row, length in enumerate(lengths):
                        prefix = length - width
                        for node in range(width):
                            position = prefix + depths[row, node]
                            left = max(position - window, 0) if window >= 0 else 0
                            visible = list(range(min(prefix, left), prefix))
                            visible += [prefix + i for i in range(width)
                                        if ancestry[row, node] >> i & 1
                                        and prefix + depths[row, i] >= left]
                            kv = table[row, visible]
                            h = row * width + node
                            logits = q[h] @ k[kv].T * 0.125
                            probability = np.exp(logits - logits.max(axis=1, keepdims=True))
                            reference[h] = (probability @ v[kv]) / probability.sum(axis=1, keepdims=True)
                    error = np.linalg.norm(out-reference) / np.linalg.norm(reference)
                    assert np.isfinite(error) and error < 0.004, error
                    max_l2 = max(max_l2, float(error))
                    digest.update(out.tobytes())
                    checked += 1
    return checked, max_l2, digest.hexdigest()


def verify_graph_stage(path):
    _, (stage,) = program(path, ("stage_dflash_tree",), "FlashAttentionBackend")
    checked = 0
    for width in range(2, 9):
        for graph_size in (1, 4, 32, 64):
            chain = array(np.broadcast_to(np.arange(width), (graph_size, width)).copy())
            chain_ancestors = array((1 << (chain + 1)) - 1)
            buffers = {"tree_linear_depths": chain, "tree_linear_ancestors": chain_ancestors,
                       "tree_depths": chain.copy(), "tree_ancestors": chain_ancestors.copy()}
            pointers = {k: v.ctypes.data for k, v in buffers.items()}
            backend = SimpleNamespace(target_verify_metadata=buffers)
            for active in range(1, graph_size+1):
                tree = array(np.zeros((active, width), dtype=np.int64))
                stage(backend, tree, tree+1)
                np.testing.assert_array_equal(buffers["tree_depths"][:active], tree)
                np.testing.assert_array_equal(buffers["tree_ancestors"][:active], tree+1)
                np.testing.assert_array_equal(buffers["tree_depths"][active:], chain[active:])
                stage(backend, None, None)
                np.testing.assert_array_equal(buffers["tree_depths"], chain)
                np.testing.assert_array_equal(buffers["tree_ancestors"], chain_ancestors)
                assert pointers == {k: v.ctypes.data for k, v in buffers.items()}
                checked += 1
    return checked


def verify_backend_capability(path):
    class Attention(SimpleNamespace):
        pass

    decoder = object()
    spec = SimpleNamespace(speculative_dflash_tree=True)
    config = SimpleNamespace(bs=[1, 4, 32])
    execution = SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=config)))
    parallel = SimpleNamespace(pp_size=1)
    _, (supported,) = program(
        path, ("_supports_dflash_tree",), "FlashAttentionBackend",
        {"get_spec": lambda: spec, "get_parallel": lambda: parallel,
         "get_exec": lambda: execution, "RadixAttention": Attention,
         "AttentionType": SimpleNamespace(DECODER=decoder)},
    )
    attention = Attention(
        head_dim=64, v_head_dim=64, tp_q_head_num=16, tp_k_head_num=2,
        attn_type=decoder, is_cross_attention=False, logit_cap=0.0,
    )
    hf = SimpleNamespace(model_type="muse_spark")
    translator = SimpleNamespace(is_translating=False)
    runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        max_running_requests=32, model_config=SimpleNamespace(hf_config=hf),
        model=SimpleNamespace(modules=lambda: [object(), attention]),
    )
    backend = SimpleNamespace(
        is_draft_runner=False, _hopper_verify_sms=132, kv_cache_dtype_str="fp8_e4m3",
        page_size=1, topk=1, speculative_num_draft_tokens=8,
        _disable_scheduler_metadata_precompute=False,
        kv_index_translator=translator, is_encoder_decoder=False, use_mla=False,
        is_prefill_aware_swa=False, local_attn_builder=None, fa_skip_kv_cache=False,
    )
    assert supported(backend, runner)
    rejections = (
        (spec, "speculative_dflash_tree", False),
        (runner.spec_algorithm, "is_dflash", lambda: False),
        (backend, "is_draft_runner", True),
        (backend, "_hopper_verify_sms", 0),
        (backend, "kv_cache_dtype_str", "auto"),
        (backend, "topk", 2),
        (backend, "speculative_num_draft_tokens", 16),
        (runner, "max_running_requests", 128),
        (config, "bs", [1, 4, 128]),
        (backend, "_disable_scheduler_metadata_precompute", True),
        (translator, "is_translating", True),
        (backend, "is_encoder_decoder", True),
        (backend, "use_mla", True),
        (backend, "is_prefill_aware_swa", True),
        (backend, "local_attn_builder", object()),
        (backend, "fa_skip_kv_cache", True),
        (parallel, "pp_size", 2),
        (hf, "model_type", "other"),
        (runner.model, "modules", lambda: [object()]),
        (attention, "head_dim", 128),
        (attention, "tp_q_head_num", 15),
        (attention, "tp_k_head_num", 1),
        (attention, "attn_type", object()),
        (attention, "is_cross_attention", True),
        (attention, "logit_cap", 1.0),
        *((backend, "page_size", page) for page in (2, 16, 64, 128, 160, 256)),
    )
    for owner, name, value in rejections:
        old = getattr(owner, name)
        setattr(owner, name, value)
        assert not supported(backend, runner), name
        setattr(owner, name, old)
    return 1 + len(rejections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--attention-baseline", type=Path)
    args = parser.parse_args()
    repo = args.source_root / "python/sglang"
    rng = np.random.default_rng(870)
    plans, walks = verify_tree_kernels(repo / "kernels/ops/speculative/dflash_tree.py", rng)
    cases, error, digest = verify_attention(
        repo / "kernels/ops/attention/paged_verify_attention.py", np.random.default_rng(872)
    )
    backend_file = repo / "srt/layers/attention/flashattention_backend.py"
    graph_cases = verify_graph_stage(backend_file)
    capability = verify_backend_capability(backend_file)
    print(f"{plans:,} proposals, {walks:,} accept/slot permutations, {graph_cases:,} replay resets passed")
    print(f"{cases} causal and SWA attention cases passed; maximum relative L2 {error:.6f}")
    print(f"{capability} backend capability and fallback cases passed")
    if args.attention_baseline is not None:
        baseline = verify_attention(args.attention_baseline, np.random.default_rng(872))
        assert baseline == (cases, error, digest)
        print(f"All {cases} attention outputs match the baseline bit for bit on the CPU.")
