"""Bound W2 traffic and cluster alternatives using the frozen V181 row-tile planner."""
from collections import Counter
import ast
import hashlib
from itertools import accumulate
import json
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent
ROOT = HERE.parents[2]
PATCH = CANDIDATES / "v181_peer_inputs_paired_gate_up/patch.diff"
SOURCE = "python/sglang/srt/layers/muse_spark/prefill_expert_tiles.py"
H = 4096
N_BANDS = 16
MIB = 1 << 20


def planner():
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == "52976e3b3ac177de2d4e2a1ebaeb5c0f7a86c0215bb0be6130b19f3d7dee4c3e"
    header = f"diff --git a/{SOURCE} b/{SOURCE}\n"
    section = PATCH.read_text().split(header, 1)[1].split("diff --git ", 1)[0]
    lines = [line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++")]
    definition = next(n for n in ast.parse("\n".join(lines)).body if getattr(n, "name", None) == "choose_tiles")
    ns = dict(accumulate=accumulate)
    exec(compile(ast.Module([definition], type_ignores=[]), SOURCE, "exec"), ns)
    return ns["choose_tiles"]


def census(counts, choose):
    state = ({w:{} for w in [136,152,176]}, (32,16), 132)
    choice = choose(state,counts)
    if choice is None:
        groups = [(176,counts)]
    else:
        groups = [(item[0], [b-a for a,b in zip(off,off[1:])])
                  for item,off in zip(choice[0],choice[1][-len(choice[0]):],strict=True)]
    out = Counter()
    for m, hist in groups:
        for rows in hist:
            if not rows:
                continue
            t = (rows+m-1)//m
            out["routes"] += rows
            out["active"] += 1
            out["mtiles"] += t
            out["padded_rows"] += t*m
            out["pairable_b_groups"] += t//2
            out["odd_m_tails"] += t%2
            # The last tile can use a neighboring N tile for A instead of running a dead M CTA.
            out["tail_live_rows"] += (rows-(t-1)*m) if t%2 else 0
    return out


def model():
    choose=planner()
    records=[]
    for trial in range(48):
        rng=random.Random(183000 + trial)
        counts=Counter()
        for token in range(16384):
            counts.update(rng.sample(range(256),8))
        for rank in range(8):
            hist=[counts[rank*32+e] for e in range(32)]
            assert sum(hist)<=18432
            records.append(census(hist,choose))
    totals=sum(records,Counter())
    mean={k:v/len(records) for k,v in totals.items()}
    rows, active, blocks = [mean[k] for k in ["routes","active","mtiles"]]
    b_per = H*H*(.5+1/16)
    activations = rows*H*2
    a_live = activations*N_BANDS
    a_padded = mean["padded_rows"]*H*2*N_BANDS
    packed = b_per*blocks
    # B and its scale share the same 16-element group. Include sharing of both, stronger than stock.
    b_saved = mean["pairable_b_groups"]*b_per
    tail_a_saved = mean["tail_live_rows"]*H*2*N_BANDS/2
    measured = json.loads((ROOT/"experiments/modal_serving/20261113T065259Z/result.json").read_bytes())
    case = next(c for c in measured["cases"] if c["T"]==16384)
    dense_net = case["gflop"]*1e9 / (case["dense_ceiling"]["us"]*1e-6)
    # Each pass below counts one W2 BF16 destination write and includes only W2 GEMM.
    flops = 2*(H**2)*mean["padded_rows"]
    grid={}
    for fraction in (0,.5,1):
        a = activations + fraction*(a_padded-activations)
        b = active*b_per + fraction*(packed-active*b_per)
        diag=[]
        for tf in (dense_net/1e12, 900):
            for tb in (2.4,3.6,4.8):
                floor=flops/(tf*1e9)
                def duration(ab,bb):
                    return max(floor, (ab+bb+activations)/(tb*1e9))
                prior=duration(a,b)
                diag.append(dict(effective_TFLOPs=round(tf,2),effective_TBps=tb, compute_floor_ms=floor,
                    v181_ms=prior, perfect_4A_ratio=prior/duration(max(activations,a-.75*fraction*a_padded),b),
                    perfect_Bpairs_plus_tail_A_ratio=prior/duration(max(activations,a-fraction*tail_a_saved),max(active*b_per,b-fraction*b_saved))))
        grid[str(fraction)]=diag
    return dict(samples=len(records),uniform_top8_without_replacement=True,practical_hardware_comparison=case,
        dense_composite_effective_TFLOPs=dense_net/1e12,mean=mean,
        per_layer_W2_MiB=dict(A_live=a_live/MIB,A_full_boxes=a_padded/MIB,B_scaled=packed/MIB,unique_B_scaled=active*b_per/MIB,
            destination_write=activations/MIB,nominal_4A_source_requests_saved_live=.75*a_live/MIB,
            ideal_pair_B_and_scale_requests_saved=b_saved/MIB,odd_tail_A_requests_saved=tail_a_saved/MIB),
        padded_cluster_B_without_tail_switch_extra_MMA_fraction=mean["odd_m_tails"]/blocks,
        ideal_W2_raw_output_write_plus_finalizer_row_read_MiB=2*activations/MIB,
        duplicate_miss_fraction_sensitivity=grid,
        assumptions="Zero cluster synchronization and occupancy costs. A traffic uses whole TMA destination tiles in the sensitivity model. Duplicate HBM miss fractions are varied, not measured. TFLOP/s from the dense composite combines GEMMs and SiLU and is a sensitivity parameter, not grouped W2 timing.")


if __name__=="__main__":
    print(json.dumps(model(),indent=2))
