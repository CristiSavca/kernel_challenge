"""Exercise the three candidate word kernels with actual CPU Inductor compilation."""
import ast
import json
from pathlib import Path
import sys

import torch

from binding import bind

HERE = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
path = stage / "python/sglang/srt/models/dflash.py"
bound = bind(stage, sys.argv[2] if len(sys.argv) > 2 else HERE, [str(path.relative_to(stage))])
selected = {"_pack_candidate_bits", "_rank_ordered_candidate_bits", "_pack_candidate_words", "_candidate_word_scores", "_candidate_word_ids"}
functions = [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in selected]
scope = dict(torch=torch, get_compiler_backend=lambda: "inductor", _is_npu=False)
exec(compile(ast.Module(functions, []), str(path), "exec"), scope)
world, k = 4, 16
starts = torch.tensor([41, 10400000, 132000000, 2**34])
checked = 0
for n in (7, 28, 112, 224, 1024):
    a = torch.arange(world * n * k, dtype=torch.int32).reshape(world*n, k)
    ids = (a * 2731 + 3453) % 65536
    vals = (a * 47527 + 8733).to(torch.int16).view(torch.bfloat16)
    words = scope["_pack_candidate_words"](vals, ids).view(world, n, k)
    native_words = ((vals.view(torch.int16).int() << 16) | ids).view_as(words)
    if not torch.equal(words, native_words):
        idx = (words != native_words).nonzero()[:8]
        raise ValueError((idx.tolist(), words[tuple(idx.T)].tolist(), native_words[tuple(idx.T)].tolist()))
    got_scores = scope["_candidate_word_scores"](words, torch.bfloat16)
    scores = vals.float().view(world, n, k).transpose(0, 1).reshape(n, -1).contiguous()
    assert torch.equal(got_scores.view(torch.int32), scores.view(torch.int32))
    top, sel = torch.topk(scores, k)
    got, picked = torch.topk(got_scores, k)
    assert torch.equal(got.view(torch.int32), top.view(torch.int32)) and torch.equal(sel, picked)
    actual = scope["_candidate_word_ids"](words, picked, starts)
    expect = (ids.view(world, n, k) + starts[:, None, None]).transpose(0,1).reshape(n,-1)
    assert torch.equal(actual, expect.gather(-1, sel))
    checked += a.numel()
print(json.dumps(dict(torch=torch.__version__, compiled="CPU Inductor", tested_candidate_words=checked,
    public_k=16, ranks=world, rows=[7,28,112,224,1024], all_65536_bf16_and_local_id_patterns=True,
    bitwise_scores_indices_and_native_topk=True, physical_h200=False,
    **bound), indent=2))
