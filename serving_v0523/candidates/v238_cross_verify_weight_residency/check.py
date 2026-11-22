"""Bound extra first-touch expert-weight hits due solely to cross-verification-window L2 persistence."""
import ast
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
ROOT = CAND.parents[1]
MIB = 1 << 20
LAYERS, GPUS, GLOBAL_EXPERTS, TOPK = 62, 8, 256, 8
HIDDEN, INTERMEDIATE = 4096, 4096


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def suffix(match):
    return next(CAND.glob(match))


def files_added_by(candidate):
    path = candidate / 'patch.diff'
    raw = path.read_bytes()
    manifest = read(candidate / 'validation.json')
    assert digest(raw) == manifest['patch_sha256']
    files = {}
    for chunk in raw.decode().split('diff --git ')[1:]:
        header, _, hunk = chunk.partition('\n@@')
        if not hunk or '\n--- /dev/null\n' not in header:
            continue
        key = header.split('\n+++ b/')[1].strip()
        lines = hunk.splitlines()[1:]
        assert all(line.startswith('+') for line in lines)
        files[key] = '\n'.join(line[1:] for line in lines) + '\n'
    return manifest, files


def check_parent():
    latest, final = files_added_by(suffix('v235_*'))
    prior, old = files_added_by(suffix('v213_*'))
    source = {}
    for name in read(suffix('v226_*')/'results.json')['unchanged_decode_bodies']:
        full = next(path for path in final if path.endswith('/'+name))
        assert old[full] == final[full]
        source[name] = digest(final[full].encode())
    path = next(p for p in final if p.endswith('/muse_spark_humming_moe.py'))
    def body(src, name):
        tree = ast.parse(src)
        fn = next(n for n in tree.body if getattr(n,'name',None)==name)
        return ast.dump(fn, include_attributes=False)
    for name in ('humming_grouped_decode_outputs','humming_decode_eligible','_plan_grouped_decode',
                 '_gather_grouped_decode'):
        code = body(old[path],name)
        assert code == body(final[path],name)
        source[name] = digest(code.encode())
    return dict(V235=latest['patch_sha256'],V213=prior['patch_sha256'],identical=source)


def cache_oracle():
    # Page IDs include layer, expert, and packed-weight tile; the small replay
    # includes empty/disjoint/fully repeating layers and a previous-epoch oracle.
    ids = [(layer,expert,page) for layer in range(2) for expert in range(2) for page in range(2)]
    counts=0
    for previous_masks in product(range(4),repeat=2):
        previous={key for key in ids if previous_masks[key[0]] & (1<<key[1])}
        for masks in product(range(4),repeat=2):
            # Repeat touched pages deliberately; intra-window hits are charged separately.
            trace=[key for key in ids for _ in range(1+key[1]) if masks[key[0]] & (1<<key[1])]
            touched=set(trace)
            for capacity in range(4):
                ideal=0
                for k in range(min(capacity,len(previous))+1):
                    for resident in combinations(sorted(previous),k):
                        already=set()
                        hits=0
                        for key in trace:
                            if key not in already and key in resident: hits+=1
                            already.add(key)
                        ideal=max(ideal,hits)
                assert ideal==min(capacity,len(touched & previous))
                counts+=1
    return counts


def summarize(global_rows, capacity=50):
    # Every token has a distinct top-k draw. Different rows are independent here.
    fraction_active=1-(1-TOPK/GLOBAL_EXPERTS)**global_rows
    expert_bytes=HIDDEN*INTERMEDIATE*3*(1/2+1/16)
    distinct_mib=expert_bytes/MIB*(GLOBAL_EXPERTS//GPUS)*fraction_active
    total=LAYERS*distinct_mib
    return dict(rows=global_rows,expected_distinct_experts_local=(GLOBAL_EXPERTS//GPUS)*fraction_active,
                expected_first_touch_MiB_per_layer_rank=distinct_mib,expected_first_touch_GiB_per_epoch_rank=total/1024,
                optimistic_ratio_of_expected_first_touch_bytes_saved=capacity/total,
                proportional_byte_work_ratio=total/(total-capacity))


def main():
    file=suffix('v213_*')/'physical.json'
    physical=read(file)
    stock=read(ROOT/'serving_v0523/runs/abd217baf0ed49fd.json')['result']
    v68=read(ROOT/'serving_v0523/runs/04b01f958aa5487c.json')['result']
    bank=int(HIDDEN*INTERMEDIATE*3*(1/2+1/16)/MIB)
    assert bank==27
    histograms=[]
    for q in physical['cases']:
        initial=q['active_experts']*bank
        histograms.append(dict(rows=q['rows'],actual_active_experts=q['active_experts'],
            distinct_weight_MiB_one_layer=initial,tiled_requested_MiB_one_layer=q['nominal_Humming_requested_weights_MiB'],
            if_repeated_across_layers_first_touch_GiB=initial*LAYERS/1024,
            carryover_ratio_if_repeated=50/(initial*LAYERS),same_layer_repeats_MiB=q['nominal_Humming_requested_weights_MiB']-initial,
            observed_different_kernel_bandwidth_percent=q['measured_percent_dram_peak'],observed_kernel=q['measured_different_kernel']))
    rungs=[]
    for r in stock['rungs']:
        n=r['concurrency'];period_ms=1000*r['acceptance']/r['tok_s_per_user_p50']
        metrics={k:r[k] for k in ('concurrency','tok_s_per_gpu','tok_s_per_user_p50','acceptance','ttft_first_p90_s')}
        metrics.update(summarize(8*n))
        metrics['proxy_verify_period_ms']=period_ms
        metrics['time_fraction_on_proxy_at_3_6TBs_rank']=50*MIB/(3.6e12*period_ms/1000)
        metrics['serial_sum_of_eight_rank_fractions_at_3_6TBs']=GPUS*metrics['time_fraction_on_proxy_at_3_6TBs_rank']
        rungs.append(metrics)
    guaranteed=LAYERS*TOPK*bank
    print(json.dumps(dict(status='no production patch',source=check_parent(),model='62 independent layers, 256 experts, EP8, 8 distinct routes/token. Full-K BF16 inputs, group-16 E4M3 scaling; previous-step prediction, byte allocation and execution overhead idealized.',cache_capacity_MiB=50,
        resident_cap_in_whole_expert_equivalents=50/bank,minimum_node_first_touch_MiB=guaranteed,node_capacity_MiB=50*GPUS,
        worst_routing_node_first_touch_save_fraction=50*GPUS/guaranteed,exhaustive_tiny_epoch_cases=cache_oracle(),
        physical_histograms=histograms,physical_input_sha256=digest(file.read_bytes()),V64_scored_rungs=rungs,V68_private_16_tok_s_per_gpu=v68['rungs'][0]['tok_s_per_gpu'],
        transfer_us_per_rank_if_fully_critical={str(bw):50*MIB/(bw*1e12)*1e6 for bw in (2.4,3.6,4.8)},
        limitations='First-touch bytes only; repeated row tiles in the same step, expert-group scheduling, indirect latency, caches, per-layer overlaps and complete serving times are separate. Actual H200 histograms are from a different standalone kernel; scaled-up 62-layer values are illustrations, not measured service traffic.'),indent=2))

if __name__=='__main__':
    main()
