#!/usr/bin/env python3
"""One small-WGMMA geometry's arithmetic; no candidate, CUDA or GPU work."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py'
ROUTES=ROOT/'experiments/modal/20261104T052256Z-4c806134/result.json'
OUT=ROOT/'experiments/iterations/20261104-small-wgmma-feasibility.json'
G=132


def jobs8(counts):
    return [min(8,c-k) for c in counts for k in range(0,c,8)]


def geometry(rows, units_a, units_b, stages_per_unit):
    phases=[]
    for units in (units_a,units_b):
        total=len(rows)*units
        touches=0;global_x_bytes=0
        for cta in range(G):
            lo=total*cta//G;hi=total*(cta+1)//G
            if lo==hi:continue
            for j in range(lo//units,(hi-1)//units+1):
                touches+=1;global_x_bytes+=rows[j]*4096*2
        phases.append(dict(units=total,active_ctas=min(G,total),
            max_weight_stages_per_cta=math.ceil(total/G)*stages_per_unit,
            x_job_cache_fills=touches,global_x_logical_bytes=global_x_bytes))
    return dict(phases=phases,global_x_logical_bytes=sum(p['global_x_logical_bytes'] for p in phases),
                x_job_cache_fills=sum(p['x_job_cache_fills'] for p in phases))


def row(label,counts,kind):
    rows=jobs8(counts);old=geometry(rows,512,256,1);new=geometry(rows,128,64,4)
    return dict(label=label,kind=kind,local_counts=counts,job_rows=rows,J8=len(rows),
                local_routes=sum(counts),incumbent=old,selected=new,
                ratio_max_B_weight_stages=(new['phases'][1]['max_weight_stages_per_cta']/old['phases'][1]['max_weight_stages_per_cta'] if rows else None),
                selected_full_padded_x_cache_store_bytes=new['x_job_cache_fills']*65536)


def main():
    records=[]
    for j in (0,1,2,3,4,5,8):
        # General valid local histograms; these are not claims about their
        # probability or the official busiest-shard M1 distribution.
        records.append(row(f'general_{j}_one_token_experts',[1]*j+[0]*(32-j),'valid_histogram'))
    r=json.loads(ROUTES.read_text())
    for case in r['cases']:
        if case['case']['bs']==16:
            for route in case['routing_replays']:
                records.append(row(f'M16_saved_{route["input_seed"]}',route['local_counts'],'saved_standalone_not_decoder'))
    footprint=36608+3*36864+8*4096*2
    assert footprint==212736
    assert 128*4+64*4==512+256==768
    weight=(512+256)*36864
    assert weight==27*1024**2
    partial_old=768*(8*32*4*4)*2
    partial_new=192*(2*128*4*4)*2
    activation_new=192*(2*128*16*8*2)
    assert (partial_old,partial_new,activation_new)==(6*1024**2,int(1.5*1024**2),12*1024**2)
    result=dict(verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        route_evidence_sha256=hashlib.sha256(ROUTES.read_bytes()).hexdigest(),
        decision='NO_GO_IMPLEMENTATION_WITH_CURRENT_EVIDENCE',
        selected_geometry=dict(threads=288,consumer_warpgroups=2,helper_warps=1,
            output_rows=64,token_job_max=8,K_per_warpgroup=2048,K_per_warpgroup_stage=512,
            weight_stages_per_unit=4,weight_bytes_per_stage=36864,weight_slots=3,
            full_activation_cache_bytes=65536,existing_metadata_bytes_rounded=36608,
            dynamic_shared_bytes=footprint,intended_ctas_per_sm=1),
        per_eight_padded_token_job=dict(weight_bytes=weight,weight_TMA_copies_old=768,weight_TMA_copies_new=768,
            units_old=768,units_new=192,partial_shared_store_plus_load_old=partial_old,
            partial_shared_store_plus_load_new=partial_new,additional_shared_B_operand_payload=activation_new,
            net_extra_shared_payload_before_cache_fills=activation_new+partial_new-partial_old,
            mathematical_K_partial_add_reduction=589824,
            current_B_norm_atomics_per_real_route=256,potential_grouped_B_norm_atomics_per_real_route=64),
        records=records,limits=[
            'Logical payloads/request counts, not measured HBM/shared transactions or latency.',
            'No compiler allocation, occupancy query or proof of proposed shared descriptor/handshake implementation.',
            'The full activation cache adds stores on every CTA/job fill; exact padded-zero strategy is unspecified.',
            'Task granularity can enlarge the critical work tail even when all SMs have work.',
            'Saved routes are standalone M16 examples, not decoder evidence or a tuning rule.',
            'Grouped norm atomics require an epilogue redesign and are not an automatic benefit of WGMMA.'
        ],gpu_used=False,candidate_created=False)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),shared=footprint,payload=result['per_eight_padded_token_job'],
        cases=[dict(label=x['label'],J8=x['J8'],A_ctas=x['selected']['phases'][0]['active_ctas'],
                    B_ctas=x['selected']['phases'][1]['active_ctas'],B_maxstage_ratio=x['ratio_max_B_weight_stages']) for x in records]),indent=2))


if __name__=='__main__':main()
