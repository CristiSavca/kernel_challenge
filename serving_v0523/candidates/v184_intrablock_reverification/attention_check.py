"""Synthetic two-layer causal target with explicit prefix KV, fragmented physical slots and transient taps."""
import json
from pathlib import Path

import torch

from check import W, accepted

B, H, V = 64, 12, 31
DT = torch.float64
torch.manual_seed(184041)
torch.set_num_threads(1)


def rnd(*shape):
    return (torch.randn(shape, dtype=DT) / H**.5).bfloat16().to(DT)


embed, head = rnd(V, H), rnd(H, V)
weights = [(rnd(H, H), rnd(H, H), rnd(H, H), rnd(H, H)) for _ in range(2)]
round_bf = lambda x: x.bfloat16().to(DT)


def forward(ids, past):
    # Each row's full physical prefix is gathered in logical order. New K/V are appended per layer.
    x = embed[ids]
    n = ids.size(1)
    taps, writes = [], []
    for layer, (wq, wk, wv, wo) in enumerate(weights):
        prior_k, prior_v = past[layer]
        length = prior_k.shape[1]
        q, k, v = (round_bf(x @ w) for w in (wq, wk, wv))
        k_all, v_all = torch.cat((prior_k, k), 1), torch.cat((prior_v, v), 1)
        mask = torch.arange(length+n)[None] > (length + torch.arange(n)[:, None])
        logits = (q @ k_all.transpose(1, 2) / H**.5).masked_fill(mask, -torch.inf)
        x = round_bf(x + (logits.softmax(-1) @ v_all) @ wo)
        taps.append(x);writes.append((k, v))
    return (x @ head).argmax(-1), torch.cat(taps, dim=-1), writes


def chain(root, prefix):
    input_ids = root
    for _ in range(W):
        preds, _, _ = forward(input_ids, prefix)
        input_ids = torch.cat((input_ids, preds[:, -1:]), 1)
    _, taps, writes = forward(input_ids[:, :-1], prefix)
    return input_ids, taps, writes


def test(prefix_len):
    blank = torch.empty(B, 0, H, dtype=DT)
    past = [(blank, blank) for _ in weights]
    if prefix_len:
        prior = torch.randint(V, (B, prefix_len))
        _, _, past = forward(prior, past)
    oracle, oracle_taps, oracle_writes = chain(torch.randint(V, (B, 1)), past)
    input_ids = torch.randint(V, (B, W))
    target_c = (torch.arange(B) % W) + 1
    for i, c in enumerate(target_c):
        input_ids[i, :c] = oracle[i, :c]
        if c < W:
            input_ids[i, c] = (oracle[i, c] + 1) % V
    pred1, taps, writes1 = forward(input_ids, past)
    tokens, lengths = accepted(input_ids, pred1)
    assert torch.equal(target_c, lengths)
    nslots = B * W * 2
    slots = torch.randperm(nslots-1)[:B*W].reshape(B,W) + 1
    bank = torch.full((2, 2, nslots, H), -9999., dtype=DT)
    for l, vals in enumerate(writes1):
        for k, val in enumerate(vals):
            bank[l,k,slots] = val
    committed_taps = taps.clone()
    overwrite_changes = 0
    lens = lengths.clone()
    for c in range(1,W):
        group = torch.where(lengths == c)[0]
        pos = slots[group]
        before = bank.clone()
        old = [(torch.cat((past[l][0][group], bank[l,0,pos[:,:c]]),1),
                torch.cat((past[l][1][group], bank[l,1,pos[:,:c]]),1)) for l in range(len(weights))]
        repair = pred1[group,c-1:W-1]
        pred2, correct_taps, new = forward(repair,old)
        repaired, n = accepted(repair,pred2)
        tokens[group,c:] = repaired
        lens[group] += n
        overwrite_changes += int((committed_taps[group,c:]!=correct_taps).sum())
        committed_taps[group,c:] = correct_taps
        for l, vals in enumerate(new):
            for k, val in enumerate(vals):
                bank[l,k,pos[:,c:]] = val
        untouched = torch.ones(nslots,dtype=torch.bool)
        untouched[pos[:,c:]] = False
        assert torch.equal(before[:,:,untouched], bank[:,:,untouched])
    mask = torch.arange(W)[None] < lens[:,None]
    assert torch.equal(tokens[mask],oracle[:,1:][mask])
    assert torch.equal(committed_taps[mask],oracle_taps[mask])
    for l, vals in enumerate(oracle_writes):
        for k, val in enumerate(vals):
            assert torch.equal(bank[l,k,slots[mask]],val[mask])
    assert overwrite_changes>0
    return dict(prefix=prefix_len, cases=B, committed_rows=int(mask.sum()), different_offpath_tap_values=overwrite_changes,
                reserved_physical_slots=B*W, scalar_committed_KV_checks=int(mask.sum())*len(weights)*2*H)


if __name__=="__main__":
    print(json.dumps(dict(synthetic_causal_model=True,target_width=W,layer_count=2,hidden=H,vocab=V,
                          loads=[test(n) for n in (0,7,32,129)],physical_GPU=False),indent=2))
