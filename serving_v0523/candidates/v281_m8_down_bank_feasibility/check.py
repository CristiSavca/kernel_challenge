"""Test retained M8 W2 operand sharing and price a second Marlin-formatted bank."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json,math
from pathlib import Path
import runpy
from urllib.request import urlopen
from zipfile import ZipFile

here=Path(__file__).resolve().parent
root=here.parents[2]
mib=1<<20
p275=here.parent/'v275_wide_decode_output_tiles'
p278=here.parent/'v278_overlapped_wide_decode'
source=runpy.run_path(str(here.parent/'v216_decode_weight_multicast_feasibility/check.py'))
wheel=urlopen(source['WHEEL'],timeout=25).read()
assert sha256(wheel).hexdigest()==source['WHEEL_SHA']
headers=['memory/s2r_loader/loader_b.cuh','memory/s2r_loader/loader_bs.cuh','datatype/dequant.cuh']
with ZipFile(BytesIO(wheel)) as z:
    body={n:z.read('humming/include/humming/'+n) for n in headers}
for s in (b'kLoadHalfGroup',b'warp_id % 2 * kLoadIters + j',b'TRUE_N_WARPS * 32 * warp_weight_blocks'):
    assert s in body[headers[0]]
assert b'lane_id / 4' in body[headers[1]] and b'n_warp_id % kNumWarpsPerPackBlock' in body[headers[1]]
assert b'uint32_t qb_index = kResultPackedNumSourceBits * index / 32;' in body[headers[2]]

selected=json.loads((p278/'selected_codegen.json').read_bytes())['selected']['w2_8_native']
kernel=json.loads((p275/'assembly.json').read_bytes())['variants']['w2_8_native']
assert selected['source_sha256']==kernel['source_sha256'] and selected['matched_cubin_sha256']==kernel['cubin_sha256']
assert kernel['block']==[8,128,256] and kernel['warp']==[8,32,64] and kernel['threads']==640 and kernel['math_threads']==512

# The b-loader uses uint2 in the native half-group. The scale loader uses uint32.
# Distinguish every physical register destination, including K-worker and K16 step.
words,scales,bad=Counter(),Counter(),Counter()
for kg in range(4):
 for it in range(4):
  group=4*kg+it
  for nw in range(4):
   for lane in range(32):
    for j in range(2):
     q=2*((32*(nw//2)+lane)*2+nw%2)+j
     words[group*256+q]+=1
     bad[group*256+q-2*(nw%2)]+=1
    for j in range(4):
     s=4*(16*(nw//2)+2*(lane//4)+nw%2)+j
     scales[group*128+s]+=1
assert set(words)==set(range(128*256//8)) and set(words.values())=={1}
assert set(scales)==set(range(128*256//16)) and set(scales.values())=={4}
assert len(bad)*2==len(words) and set(bad.values())=={2}

# Conditional on U already seen, exactly 32-U of 256 expert IDs are still new
# locals. One token uniformly chooses a distinct subset of eight global IDs.
def span_prob(rows):
    P={0:1.0}
    for _ in range(rows):
        nxt=Counter()
        for u,p in P.items():
            for j in range(min(32-u,8)+1):
                nxt[u+j]+=p*math.comb(32-u,j)*math.comb(224+u,8-j)/math.comb(256,8)
        P=nxt
    assert math.isclose(sum(P.values()),1)
    return P
prob=span_prob(8);exp=sum(u*p for u,p in prob.items())
assert math.isclose(exp,32*(1-(1-8/256)**8))
expected=lambda fn:sum(p*fn(u) for u,p in prob.items())
# Check occupancy endpoints and verify the model against exact exhaustive toy draws.
assert len(span_prob(0))==1 and math.isclose(sum(span_prob(1).values()),1)
from itertools import combinations,product
toy=list(combinations(range(6),2))
toy_count=Counter(len({e for row in r for e in row if e<3}) for r in product(toy,repeat=3))
pm={0:1}
for iteration in range(3):
 nxt=Counter()
 for u,count in pm.items():
  for j in range(min(3-u,2)+1):
   nxt[u+j]+=count*math.comb(3-u,j)*math.comb(3+u,2-j)
 pm=nxt
assert pm==toy_count

def data(path):return json.loads(path.read_bytes())
h200=root/'experiments/modal_serving/20261112T191039Z/result.json'
counter=root/'experiments/iterations/20261104-moe-hardware-profile-recovered.json'
old=next(c for c in data(h200)['cases'] if c['T']==8)
assert old['local_routes']==10 and old['stock_us']==129.79 and old['v40_us']==123.62
z=next(c for c in data(counter)['cases'] if c['M']==32)
dram=z['counter_launch']['metrics']['dram__bytes_read.sum']['value']/mib
unique=z['routing']['unique_weight_bytes']/mib
w2_bytes=4096*4096//2+4096*(4096//16)
assert w2_bytes==9*mib
a_per_expert=8*4096*2*(4096//128)
reach=32*62*w2_bytes
print(json.dumps(dict(decision='Retain V278 M8 W2. Reject cross-warp packed-weight sharing and retained dual-layout/per-pass materialization as the next check.',
    M8_kernel=selected|dict(regs=kernel['registers'],stack_store_load=kernel['stack_store_load'],inner_loop_instruction_sites=kernel['full_K_loop_instruction_sites']),
    hashes={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in (p275/'assembly.json',p278/'selected_codegen.json',h200,counter)},
    wheel=source['WHEEL_SHA'],header_sha256={n:sha256(b).hexdigest() for n,b in body.items()},
    per_CTA_K256=dict(unique_packed_words=len(words),unique_FP4_values=8*len(words),owners_per_weight_word=1,
        distinct_scale_bytes=len(scales),lane_uses_per_scale_byte=4,same_word_reuse_removed=0,
        negative_control_halfgroup_parity_deleted=dict(unique_words=len(bad),owners_per_word=2)),
    uniform_M8=dict(expected_local_experts=exp,expected_W2_weight_scale_MiB=9*exp,
        expected_requested_W2_A_MiB=exp*a_per_expert/mib,expected_one_read_per_M_tile_A_MiB=exp*a_per_expert/(32*mib),
        batch_repack_extra_logical_read_write_MiB=18*exp,perfect_50_MiB_L2_excess_MiB=expected(lambda u:max(0,9*u-50)),
        prob_second_bank_larger_than_50_MiB=expected(lambda u:9*u>50),toy_route_sequences_checked=sum(toy_count.values())),
    retained_all_layers_and_experts=dict(local_experts=32,layers=62,W2_bank_GiB=reach/2**30,W2_bank_bytes=reach),
    H200=dict(M8_whole_stock_Marlin_expert_us=old['stock_us'],M8_whole_V40_Marlin_expert_us=old['v40_us'],M8_local_routes=old['local_routes'],
        observed_M8_unique_experts_min=math.ceil(old['local_routes']/8),observed_M8_unique_experts_max=old['local_routes'],
        M32_separate_standalone_DRAM_MiB=dram,M32_unique_all_weight_MiB=unique,M32_other_DRAM_MiB=dram-unique),
    scope='FP4 word ownership is exact for every M8 histogram: each expert fits in one row tile. Route probabilities assume independent uniform distinct top-eight choices per token. A and repack volumes count requests, not DRAM traffic. The capacities use a generous entire 50 MiB L2 for one bank.'),indent=2))
