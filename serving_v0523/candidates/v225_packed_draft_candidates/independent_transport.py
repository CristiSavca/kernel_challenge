"""Compare V225's word path with V224's rank-ordered TP merge on real CPU tensors."""
import ast
from contextlib import nullcontext
from functools import cache
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


DIR = Path(__file__).resolve().parent
PARENT = DIR.parent / "v224_packed_prefill_serving/patch.diff"
STAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
loader = importlib.util.spec_from_file_location(
    "independent_source", DIR.parent / "v217_captured_verify_context/independent_model_entry.py"
)
util = importlib.util.module_from_spec(loader)
loader.loader.exec_module(util)
util.PARENT = PARENT.read_text()
if (DIR / "patch.diff").exists():
    util.CHILD = (DIR / "patch.diff").read_text()
    util.STAGE = STAGE
else:
    assert STAGE


@cache
def source(path, child=True):
    if child and not (DIR / "patch.diff").exists():
        raw = (STAGE / util.SRT / path).read_text()
    else:
        raw = util.read_source(path, child)
    return raw


@cache
def function_code(path, method, child, cls):
    tree = ast.parse(source(path, child))
    container = tree if cls is None else next(n for n in tree.body if getattr(n, "name", None) == cls)
    node = next(n for n in container.body if getattr(n, "name", None) == method)
    node.decorator_list = []
    return compile("from __future__ import annotations\n" + ast.unparse(node), path, "exec")


def bind(path, method, env, child=True, cls=None):
    ns = dict(env)
    exec(function_code(path, method, child, cls), ns)
    return ns[method]


MODEL = "models/dflash.py"
PARALLEL = "distributed/parallel_state.py"
COMM = "distributed/communication_op.py"
words_env = {"torch": torch}
for fn in ("_pack_candidate_bits", "_rank_ordered_candidate_bits", "_pack_candidate_words",
           "_candidate_word_scores", "_candidate_word_ids"):
    words_env[fn] = bind(MODEL, fn, words_env)


def exhaustive():
    pack = words_env["_pack_candidate_words"]
    unpack_scores = words_env["_candidate_word_scores"]
    unpack_ids = words_env["_candidate_word_ids"]
    peers, rows, k = 8, 512, 16
    patterns = torch.arange(65536).reshape(peers, rows, k)
    values = patterns.to(torch.int16).view(torch.bfloat16)
    offsets = torch.tensor([0, 70000, 123456, 310000, 919999, 24680, 0, (1 << 63) - 65536])
    slots = torch.arange(peers * k).expand(rows, -1)
    def rank_major(x):
        return x.transpose(0, 1).reshape(rows, -1)
    digest = sha256()
    for shift in (0, 1, 32767, 32768, 65535):
        ids = (patterns * 32771 + shift) % 65536
        actual = pack(values, ids.to(torch.int32))
        assert actual.dtype == torch.int32
        assert torch.equal(actual.long() & 0xffffffff, (patterns << 16) | ids)
        scores = unpack_scores(actual, torch.bfloat16)
        assert scores.is_contiguous()
        assert torch.equal(scores.view(torch.int32), rank_major(values.float()).view(torch.int32))
        recovered = unpack_ids(actual, slots, offsets)
        assert torch.equal(recovered, rank_major(ids + offsets[:, None, None]))
        digest.update(actual.numpy().tobytes())
    return {"bf16_patterns": 65536, "uint16_indices": 65536, "paired_words": 5 * 65536,
            "patterns_sha256": digest.hexdigest()}


def native_wire(peers):
    """Use the actual SGLang coordinator. Only the communication primitive is emulated."""
    calls = []
    def feed(output, tensor, group):
        shards = group(tensor.dtype)
        assert torch.equal(tensor.view(torch.uint8), shards[0].view(torch.uint8))
        assert len(shards) == len(peers)
        output.copy_(torch.cat(shards, dim=0))
    coordinator = SimpleNamespace(world_size=len(peers), local_size=len(peers), hpu_communicator=None,
                                  use_symmetric_memory=lambda *a, **k: nullcontext(),
                                  is_allocation_symmetric=lambda: False)
    env = dict(torch=torch, is_shm_available=lambda *args: False, all_gather_single=feed)
    all_gather = bind(PARALLEL, "all_gather", env, cls="GroupCoordinator")
    def gather(input_, dim):
        calls.append((str(input_.dtype), dim))
        return all_gather(coordinator, input_, dim)
    coordinator.all_gather = gather
    wrapped = bind(COMM, "tensor_model_parallel_all_gather", {"get_tp_group": lambda: coordinator})
    return coordinator, wrapped, calls


def integration():
    torch.manual_seed(9601)
    cases, decisions, payload_bytes = 0, 0, []
    digest = sha256()
    for tp in (2, 4, 8):
        for k in (1, 7, 16):
            for rows in (1, 7, 31, 112):
                for kind in ("tied", "unique", "exceptional"):
                    # Build local projection and local native PyTorch top-k separately, from actual BF16 weights.
                    n = 67
                    hidden = torch.ones(rows, 1, dtype=torch.bfloat16)
                    offsets = torch.arange(tp, dtype=torch.long) * 25256
                    weights = []
                    for rank in range(tp):
                        vector = torch.randperm(n).to(torch.bfloat16) * tp + rank
                        if kind == "tied":
                            vector.remainder_(4)
                        if kind == "exceptional":
                            bits = [0x0000, 0x8000, 0xff80, 0x7f80, 0xffc1, 0x7f81, 0x0001, 0x807f]
                            vector[:len(bits)] = torch.tensor(bits).to(torch.int16).view(torch.bfloat16)
                        weights.append(vector[:, None])
                    local = [(hidden @ weight.T).topk(k) for weight in weights]
                    vals = [v for v, _ in local]
                    ids = [i for _, i in local]
                    coordinator, collect, calls = native_wire(local)
                    def provide_shards(dtype):
                        if dtype == torch.float32:
                            return [v.float() for v in vals]
                        if dtype == torch.int64:
                            return [i + o for i, o in zip(ids, offsets)]
                        assert dtype == torch.int32
                        return [words_env["_pack_candidate_words"](v, i) for v, i in local]
                    coordinator.device_group = provide_shards
                    head = SimpleNamespace(weight=weights[0], shard_indices=SimpleNamespace(
                        num_org_elements=n, org_vocab_start_index=0))
                    env = dict(words_env, _flashinfer_top_k=None, tensor_model_parallel_all_gather=collect,
                               get_parallel=lambda: SimpleNamespace(tp_size=tp))
                    env["is_dense_head_weight"] = bind("speculative/dflash_utils.py", "is_dense_head_weight",
                        {"_DENSE_HEAD_DTYPES": (torch.float16, torch.bfloat16, torch.float32)})
                    env["should_apply_lm_head_quant_method"] = bind("layers/logits_processor.py",
                        "should_apply_lm_head_quant_method", {})
                    for fn in ("_radix_topk", "_project_candidate_logits", "_merge_candidate_words"):
                        env[fn] = bind(MODEL, fn, env)
                    parent = bind(MODEL, "candidate_topk", env, child=False)
                    child = bind(MODEL, "candidate_topk", env)
                    expected = parent(hidden, head, k)
                    assert calls == [("torch.float32", -1), ("torch.int64", -1)]
                    calls.clear()
                    actual = child(hidden, head, k, candidate_shard_starts=offsets)
                    assert calls == [("torch.int32", 0)]
                    assert expected[2] is actual[2] is None
                    assert torch.equal(actual[0], expected[0])
                    assert torch.equal(actual[1].view(torch.int32), expected[1].view(torch.int32))
                    digest.update(actual[0].numpy().tobytes())
                    digest.update(actual[1].view(torch.int32).numpy().tobytes())
                    cases += 1
                    decisions += actual[0].numel()
                    if tp == 8 and k == 16 and kind == "unique":
                        payload_bytes.append({"rows": rows, "old_sent_bytes_per_rank": 12*rows*k,
                                              "new_sent_bytes_per_rank": 4*rows*k})
    return {"tp_sizes": [2,4,8], "cases": cases, "compared_topk_ids_and_fp32_words": decisions,
            "result_sha256": digest.hexdigest(), "tp8_k16_input_payloads": payload_bytes}


if __name__ == "__main__":
    torch.set_num_threads(1)
    output = {"passed": True, "platform": "CPU, uncompiled production helper bodies and real TP all-gather reshape; peer bytes supplied locally.",
              "all_bit_patterns": exhaustive(), "model_entry": integration(), "model_source_sha256":
                  sha256(source(MODEL).encode()).hexdigest(), "patch_sha256": sha256((DIR / "patch.diff").read_bytes()).hexdigest()
                      if (DIR / "patch.diff").exists() else None}
    (DIR / "independent_transport.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output))
