#!/usr/bin/env python3
"""Independent prospective SS-A layout and SIMT-store demand; no GPU/candidate."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0'
OUT=ROOT/'experiments/iterations/20261104-wgs-shared-a-independent-feasibility.json'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    source=ROOT/'experiments/candidates/muse_moe_w4a16'/f'{BASE}.py'
    assert digest(source)==BASE
    def offset(m,k):return (m//8)*256+(k//8)*128+(m%8)*16+(k%8)*2
    seen={};maximum=0
    for bank in range(2):
        for wg in range(4):
            for jj in range(2):
                matrix={}
                for wt in range(128):
                    wp,lane=divmod(wt,32)
                    for reg in range(4):
                        m=16*wp+lane//4+8*(reg&1)
                        k=2*(lane&3)+8*(reg>>1)
                        for half in range(2):
                            address=bank*16384+wg*4096+jj*2048+offset(m,k+half)
                            assert address%2==0 and address not in seen
                            seen[address]=(bank,wg,jj,m,k+half)
                            matrix[(m,k+half)]=address;maximum=max(maximum,address+2)
                assert set(matrix)=={(m,k) for m in range(64) for k in range(16)}
                for m in range(64):
                    for k in range(16):
                        expected=bank*16384+wg*4096+jj*2048+(m%8)*16+(m//8)*256+(k%8)*2+(k//8)*128
                        assert matrix[(m,k)]==expected
    assert len(seen)==16384 and maximum==32768
    for wp in range(4):
        for reg in range(4):
            addresses=[offset(16*wp+lane//4+8*(reg&1),2*(lane&3)+8*(reg>>1)) for lane in range(32)]
            assert len({(a//4)%32 for a in addresses})==32
    cases=0
    for initial in range(4):
        for stages in range(1,17):
            live=[]
            for stage in range(stages):
                for ci in range(8):
                    bank=ci&1
                    assert all(oldbank!=bank for _,oldbank in live)
                    live.append((initial+stage,bank));live=live[-1:]
                    if ci==0 and stage:assert all(seq!=initial+stage-1 for seq,_ in live)
            live=[]
            cases+=1
    groups=48*16*8;payload=groups*16384
    receipt_path=ROOT/'experiments/runs/d24dd50196cb4158.json'
    receipt=json.loads(receipt_path.read_text());result=receipt['response']['result']
    assert receipt['source_sha256']==BASE and result['passed'] is True
    clocks={result['machine'][p]['clocks.sm'] for p in ('gpu_before','gpu_after')}
    assert clocks=={1980}
    diagnostic_path=ROOT/'experiments/iterations/20261104-decoder-selected-resource-analysis.json'
    diagnostic=json.loads(diagnostic_path.read_text())
    assert diagnostic['job']=='c0b4201c5ad345d1' and diagnostic['official_passed'] is True
    sms={r['SMs'] for r in diagnostic['resources']['records'] if r['all_queries_succeeded']}
    assert sms=={132}
    # 128 B/SM/clock is the documented 32 banks * 32 bits. This idealized
    # generic-store service model does not add tensor A reads or TMA traffic.
    scenarios=[]
    for row in result['records']:
        if row['event']!='benchmark' or row['case']['bs'] not in (32,128,256):continue
        for jobs in (16,24,27,29):
            for mhz in (1500,1980):
                us=payload*jobs/(132*128*mhz)
                scenarios.append({'M':row['case']['bs'],'J_scenario_not_official_route':jobs,
                    'clock_MHz_parameter':mhz,'store_bytes':payload*jobs,
                    'ideal_generic_store_service_us':us,
                    'public_harness_median_us':row['median_ns']/1000,
                    'public_raw_median_us':row['median_raw_ns']/1000,
                    'service_over_harness_fraction':us/(row['median_ns']/1000),
                    'service_over_raw_fraction':us/(row['median_raw_ns']/1000)})
    return {'status':'PASS_PROSPECTIVE_MODEL_ONLY','parent_sha256':BASE,
        'independent_model_sha256':digest(Path(__file__)),
        'ownership_correction':'Reconstructed independent model; prior d59b1e46 capture was author script after filename collision and is not independent evidence.',
        'layout':{'BF16_coordinates_checked':len(seen),'A_bank_bytes':16384,'A_total_bytes':32768,
            'max_relative_byte_exclusive':maximum,'descriptor_LBO':128,'descriptor_SBO':256,
            'scalar_b32_store_banks_per_warp':32,'lifetime_cases':cases},
        'shared_budget':{'metadata':22272,'two_weight_slots':73728,'two_X_slots':65536,
            'two_A_banks':32768,'dynamic_total':194304,'parent_dynamic':231168,'reduction':36864},
        'per_job':{'joined_stages':768,'TMA_weight_plus_X':1536,'packed_weight_requests':28311552,
            'groups_per_WG':groups,'new_generic_A_stores':payload,'new_logical_tensor_A_reads':payload,
            'new_WG_publication_barriers':groups*4,'new_WG_reuse_barriers':groups*4,
            'scalar_b32_thread_store_operations_if_not_combined':groups*512*8},
        'store_service_model':{'bytes_per_SM_clock':128,'SMs':132,
            'SM_evidence_sha256':digest(diagnostic_path),'receipt_sha256':digest(receipt_path),
            'recorded_before_after_SM_MHz':1980,'recorded_clock_is_not_continuous_in_kernel_clock':True,
            'formula_us':'96*2**20*J/(132*128*clock_MHz)',
            'ideal_us_per_job_at1980MHz':payload/(132*128*1980),'scenarios':scenarios},
        'limits':['No candidate source, hardware throughput, compiler resources, GPU memory order or numerical outputs measured.',
            'Full-WG barriers before reuse and after publication are model assumptions.',
            'J scenarios are not attributed to the official public runs; before/after clock samples are not continuous telemetry.',
            'Generic-store service demand is potentially overlapped, not additive latency; tensor/TMA paths are not charged against it.',
            'The 128B/clock documented bank model is not a measured Hopper port-contention or attainable throughput result.',
            'Parent C7511 status remains unknown.']}

if __name__=='__main__':
    result=run();OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'saved':str(OUT.relative_to(ROOT)),'sha256':digest(OUT),'layout':result['layout'],
        'ideal_store_us_per_job_at1980MHz':result['store_service_model']['ideal_us_per_job_at1980MHz']}))
