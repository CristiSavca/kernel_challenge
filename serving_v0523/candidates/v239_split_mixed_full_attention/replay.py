"""Replay exact V235 and V239 model/graph entry points and compare split attention operands."""
import argparse
import ctypes
from hashlib import sha256
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "v235_peer_mixed_sliding_prefill"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", type=Path)
parser.add_argument("--candidate", type=Path, default=HERE)
args = parser.parse_args()
child_folder = args.candidate.resolve()
helper = HERE.parent / "v231_prompt_query_encoding/independent_reach.py"
spec = importlib.util.spec_from_file_location("graph_replay", helper)
f = importlib.util.module_from_spec(spec)
with patch.object(sys, "argv", [str(helper), "--candidate", str(child_folder)]):
    spec.loader.exec_module(f)
H, D, QH = f.fx.H, f.fx.D, f.fx.QH
torch.set_num_threads(1)


def digest(tensor):
    raw = tensor.contiguous().view(torch.uint8)
    return sha256(ctypes.string_at(raw.data_ptr(), raw.numel())).hexdigest()


def role(fb, prompts, mode, child):
    fb.capture_hidden_mode = "full"
    fb.return_logprob = fb.is_prefill_only = False
    fb.tbo_split_seq_index = fb.token_indices_to_pool = fb.multi_item_delimiter_indices = None
    fb.spec_info = None
    fb.contains_mm_inputs = lambda: False
    length = sum(fb.extend_seq_lens_cpu)
    fb.input_ids = torch.arange(length)
    pos, slots = fb.positions, fb.out_cache_loc
    fb.positions, fb.out_cache_loc = pos[:length], slots[:length]
    env = dict(torch=torch, ForwardMode=f.harness.Mode, CaptureHiddenMode=SimpleNamespace(FULL="full"))
    fn = f.define(f.fx.SR + "model_executor/forward_batch_info.py", "set_mixed_verify_rows", env, True, "ForwardBatch")
    fn(fb, prompts)
    fb.positions, fb.out_cache_loc = pos, slots
    assert fb.mixed_verify_logits_indices.numel() == prompts + sum(fb.extend_seq_lens_cpu[prompts:])
    if child:
        assert fb.mixed_verify_prompt_count == prompts
    fb.mixed_verify_is_causal = mode != "uncertified"
    if mode == "masked":
        fb.spec_info = SimpleNamespace(custom_mask=torch.tensor([False, True]))
    if mode == "wrong_count":
        fb.mixed_verify_prompt_count = 0
    if mode == "ordinary":
        fb.mixed_verify_logits_indices = None


def check_attention(s):
    rows, prefixes, left = s["rows"], s["prefixes"], s["left"]
    count, cut = sum(rows), 0
    dense, paged = s["dense"], s["paged"]
    assert len(s["stores"]) == 1
    assert [(len(dense), len(paged))] == [expected], (expected, len(dense), len(paged))
    calls = [(c, True) for c in dense] + [(c, False) for c in paged]
    logical, outputs = [], []
    rq = 0
    for kw, is_dense in calls:
        lengths = (kw["cu_seqlens_q"][1:] - kw["cu_seqlens_q"][:-1]).tolist()
        end = rq + len(lengths)
        assert kw["cu_seqlens_q"][0] == 0 and rows[rq:end] == lengths
        n = sum(lengths)
        assert kw["q"].shape == (n, QH, D) and kw["q"].dtype == torch.float8_e4m3fn
        assert kw["out"].shape == (n, QH, D) and kw["out"].data_ptr() == s["out"][cut:].data_ptr()
        assert kw["causal"] and kw["max_seqlen_q"] == max(lengths)
        assert tuple(kw["window_size"]) == ((-1,-1) if left < 0 else (left,0))
        label = s["sentinel"] if not is_dense else -s["sentinel"]
        assert (s["out"][cut:cut+n] == label).all()
        if s["scale"]:
            for key, val in (("k_descale", .75), ("v_descale", 1.625)):
                assert torch.equal(kw[key], torch.full((end-rq,H),val))
        else:
            assert "k_descale" not in kw and "v_descale" not in kw
        if not is_dense and kw.get("cu_seqlens_k_new") is not None:
            cu = kw["cu_seqlens_k_new"]
            assert cu[0] == 0 and (cu[1:] - cu[:-1]).tolist() == [r+p for r,p in zip(lengths,prefixes[rq:end])]
        for i, r in enumerate(lengths):
            p = prefixes[rq+i]
            drop = max(0,p-left) if left >= 0 else 0
            if is_dense:
                start, stop = kw["cu_seqlens_k"][i:i+2].tolist()
                used = int(kw["seqused_k"][i])
                assert used == p+r-drop and stop-start == used+256
                kv = [kw[k][start:start+used] for k in ("k","v")]
                assert all(not kw[k][start+used:stop].view(torch.uint8).any() for k in ("k","v"))
            else:
                assert int(kw["cache_seqlens"][i]) == p+r
                slots = kw["page_table"][i,drop:p+r].long()
                kv = [kw[k][:,0].view(torch.uint8)[slots].view(torch.float8_e4m3fn) for k in ("k_cache","v_cache")]
            a,b = kw["cu_seqlens_q"][i:i+2].tolist()
            query = kw["q"][a:b]
            logical.append([digest(t) for t in (query,*kv)])
            for j in sorted({0,r//2,r-1}):
                hi=p+j+1
                lo=max(0,hi-1-left) if left >= 0 else 0
                for h in (0,QH-1):
                    k,v = [t[lo-drop:hi-drop,h//(QH//H)].double() for t in kv]
                    scale_k = float(kw["k_descale"][i,h//(QH//H)]) if s["scale"] else 1.
                    scale_v = float(kw["v_descale"][i,h//(QH//H)]) if s["scale"] else 1.
                    scores = (k @ query[j,h].double()) * kw["softmax_scale"] * scale_k
                    if kw["softcap"]:
                        scores = (scores/kw["softcap"]).tanh()*kw["softcap"]
                    value = scores.softmax(0) @ v * scale_v
                    assert value.isfinite().all()
                    outputs.append(value)
        rq, cut = end, cut+n
    assert rq == len(rows) and cut == count and s["out"][count:].count_nonzero() == 0
    banks = [digest(t) for bank in (s["pool"].full_kv_pool,s["pool"].swa_kv_pool) for t in bank.data]
    writes = [digest(t) for t in s["stores"][0]]
    return dict(logical=logical, outputs=digest(torch.stack(outputs)), banks=banks, writes=writes,
                output_scalars=len(outputs)*D, visible_KV_bytes=sum((p+r-(max(0,p-left) if left>=0 else 0))*2*H*D
                for p,r in zip(prefixes,rows)))


def run_graph():
    src = inspect.getsource(f.run)
    a = 'assert len(stores)==1 and len(dense)+len(paged)==1'
    assert src.count(a)==2
    src=src[:src.rindex(a)] + 'return check_attention(locals())\n'
    src=src.replace('dense.clear(); paged.clear(); stores.clear()', 'dense.clear(); paged.clear(); stores.clear()\n        role(fb,prompt_count,cert,version)')
    src=src.replace('return result.fill_(sentinel)', 'return result.fill_(-sentinel)')
    env=dict(vars(f), role=role, check_attention=check_attention)
    exec(src,env)
    return env

# The pinned pack kernel accepts a 1-sequence grid. Extend the old CPU-only
# witness (whose original caller required a batch of two) to cover that grid.
import textwrap
meth=textwrap.dedent(inspect.getsource(f.fx.KernelBoundary.__getitem__))
assert meth.count('grid[1] >= 2')==1
scope=dict(vars(f.fx))
exec(meth.replace('grid[1] >= 2','grid[1] >= 1'),scope)
f.fx.KernelBoundary.__getitem__=scope['__getitem__']
env = run_graph()
cases = [
    ([4096,8,8], [507,1931,2039], 1, -1, "approved", True),
    ([16384,8,8], [49152,65536,49159], 1, -1, "approved", False),
    ([3280,1820,8,8], [1259,1581,4224,3199], 2, -1, "approved", True),
    ([4096]+[8]*31, [2233]+[4129]*31, 1, -1, "approved", False),
    ([4096,8,8], [126900,4300,9000], 1, -1, "oversize", True),
    ([4096,8,8], [101,1462,2212], 1, -1, "uncertified", True),
    ([4096,8,8], [321,1226,1116], 1, -1, "masked", True),
    ([4096,8,8], [193,1644,1673], 1, -1, "wrong_count", True),
    ([4096,8,8], [306,1198,1616], 1, -1, "ordinary", True),
    ([4096,8,8], [554,4200,1231], 1, 2047, "approved", True),
]
results=[]; manifests={}
for rows,prefixes,prompt_count,left,cert,scale in cases:
    paired=[]
    for version,folder in enumerate((PARENT,child_folder)):
        m=json.loads((folder/'validation.json').read_bytes())
        f.reader.CHILD,manifests[m["candidate"]]=f.checked_patch(folder,m)
        f.STAGE=f.reader.STAGE=args.stage if version else None
        f.source.cache_clear()
        expected = (1,1) if version and cert=="approved" and left<0 else (1,0) if left>=0 or cert=="ordinary" else (0,1)
        env.update(prompt_count=prompt_count,cert=cert,version=version)
        paired.append(env["run"](True,rows,prefixes,left,None,scale))
    assert paired[0] == paired[1]
    results.append(dict(rows=rows,prefixes=prefixes,prompts=prompt_count,window=left,guard=cert,
                        scales=scale,candidate_launches=expected,**{k:paired[0][k] for k in ("output_scalars","visible_KV_bytes")}))
print(json.dumps(dict(candidates=manifests,cases=results,paired_model_forwards=len(cases),visible_KV_bytes=sum(r["visible_KV_bytes"] for r in results),
                      sampled_attention_scalars=sum(r["output_scalars"] for r in results),physical_H200=False,
                      coverage="CPU Muse producer, Breakable callback replay, scaled native physical cache writer, real host dispatch, independent sampled float64 attention and two disjoint BF16 output slices. FA3 hardware execution pending."),indent=2))

