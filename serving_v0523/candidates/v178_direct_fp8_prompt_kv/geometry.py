"""Count exact required pairs and redistribution for token-owner attention on the fixed TP8 node."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = json.loads((HERE.parent/"v170_long_prefill_attention_feasibility/check.json").read_text())
LENGTH = BASE["shape"]["first_turn"]
BLOCK = 16384
LEFT = 2047
H, D, KV = 16, 64, 2
BPER = 2*KV*D
LAYERS = ((16, None),(46, LEFT))


def intervals(start, end, width):
    return (0 if width is None else max(0,start-width), end)


def union(ranges):
    a,b=sorted(ranges)[0];total=0
    for c,d in sorted(ranges)[1:]:
        if c>b:
            total+=b-a
            a,b=c,d
        else:
            b=max(b,d)
    return total+b-a


def rank_work(ranges,width):
    return sum(sum((i+1) if width is None else min(i+1,width+1) for i in range(a,b)) for a,b in ranges)


def evaluate(cohort, split):
    # Existing layout has every source GPU's own KV heads for all tokens. The transposed
    # prompt algorithm computes every head for its subset of token positions, then sends
    # O back to the original head owners for their existing row-parallel projection.
    # Subsets cover logical row numbers independent of interleaved physical stores.
    load,ideal=[0]*cohort,0
    sends=[0]*cohort
    for p in range(0,LENGTH,BLOCK):
        n=min(BLOCK,LENGTH-p)
        divs=cohort*(2 if split=="paired_edges" else 1)
        ranges=[(p+i*n//divs,p+(i+1)*n//divs) for i in range(divs)]
        owners=[[ranges[r],ranges[-r-1]] if split=="paired_edges" else [ranges[r]] for r in range(cohort)]
        assert sorted(i for part in owners for a,b in part for i in range(a,b))==list(range(p,p+n))
        for reps,width in LAYERS:
            work=[rank_work(part,width)*reps for part in owners]
            ideal+=sum(work)
            for r,wr in enumerate(work):load[r]+=wr
            needs=[union([intervals(a,b,width) for a,b in part]) for part in owners]
            for sender in range(cohort):
                sends[sender]+=sum(size for i,size in enumerate(needs) if i!=sender)*BPER*reps
    # Useful QK/AV FLOPs acquire a factor C in head count and lose C in rows.
    rank_flops=[w*cohort*4*D*H for w in load]
    assert sum(rank_flops)/cohort==BASE["math"]["useful_qk_and_av_flops_per_rank"]
    move_qo=LENGTH*62*H*D*(1+2)*(cohort-1)/cohort
    return dict(cohort=cohort,row_partition=split,logical_q_fp8_o_bf16_sent_per_rank=int(move_qo),
                kv_bytes_sent_per_rank=sends,flops_per_rank=rank_flops,
                largest_rank_over_incumbent=max(rank_flops)/BASE["math"]["useful_qk_and_av_flops_per_rank"])


def main():
    rows=[];gather=0;all_gather=0
    for p in range(0,LENGTH,BLOCK):
        n=min(BLOCK,LENGTH-p);sizes={}
        for reps,width in LAYERS:
            halo=p if width is None else min(p,width)
            sizes["full" if width is None else "swa"]=dict(halo=halo,arena_bytes=(halo+n)*BPER)
            gather+=reps*halo*BPER
            all_gather+=reps*(halo+n)*BPER
        rows.append(dict(prefix=p,query=n,**sizes))
    assert all_gather==sum(BASE["math"][key] for key in
              ("first_turn_kv_bytes_copied_per_rank_if_full_packed_all_chunks","first_turn_kv_bytes_copied_per_rank_if_swa_packed_all_chunks"))
    print(json.dumps(dict(v64_id=BASE["official_id"],cold_rows=rows,fp8_kv_bytes_a_full_gather_would_collect=all_gather,
              fp8_kv_bytes_gathered_by_direct_cast=gather,avoided_extra_gather_read_plus_write_bytes=2*(all_gather-gather),
              peak_packed_payload=max(row["full"]["arena_bytes"] for row in rows),
              cp_native_resolver=BASE["layernorm_sp_resolver"],native_math=BASE["math"],
              head_to_token_redistribution=[evaluate(c,p) for c in (2,4,8) for p in ("contiguous","paired_edges")]),indent=2))


if __name__=="__main__":
    main()
