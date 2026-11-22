"""Evaluate a two-CTA expert multicast schedule against the V215 geometry and H200 counters."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import json
import math
import random


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MIB = 1 << 20
N_TILES = 32  # W13 N=8192, CTA N=256. W2 N=4096, CTA N=128.
BANK_BYTES = 3 * 4096**2 * 9 // 16
WHEEL = ("https://files.pythonhosted.org/packages/ec/a2/f96912ee7d68f56e61c15555657ffa057"
         "b8afbaa21241c63342b5969bb74/humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl")
WHEEL_SHA = "cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"


def scheduling_oracle(counts, width):
    writes, transfers = Counter(), Counter()
    for e, count in enumerate(counts):
        tiles = (count + width - 1) // width
        for m in range(tiles):
            for n in range(N_TILES // 2):
                first = (m // 2) * 2
                hot = first + 1 < tiles
                for rank in range(2):
                    row = first + rank if hot else m
                    column = n * 2 + (m % 2 if hot else rank)
                    writes[e, row, column] += 1
                transfers["same_weight" if hot else "same_input"] += 1
    assert all(v == 1 for v in writes.values())
    for e, count in enumerate(counts):
        for m in range((count + width - 1)//width):
            assert all((e, m, n) in writes for n in range(N_TILES))
    return len(writes),transfers


def table(tiles, pairs, width):
    single = tiles-2*pairs
    a_bytes=width*4096*2*N_TILES*2
    return dict(tiles=tiles, paired_row_tilesets=pairs, unpaired_tiles=single,
                v215_packed_requested_mib=tiles*BANK_BYTES/MIB,
                weight_request_savings_mib=pairs*BANK_BYTES/MIB,
                operand_a_request_savings_mib=single*a_bytes/2/MIB,
                incremental_request_savings_vs_a_only_v69_mib=pairs*(BANK_BYTES-a_bytes)/MIB,
                weight_only_ideal_ratio=tiles/(tiles-pairs))


def main():
    patch=(HERE.parent/"v215_decode_reduction_paired_context/patch.diff").read_bytes()
    assert sha256(patch).hexdigest()=="bbf69a796f2e2ae2edb2b5d913f0e0dc799037d555c10dfe58d00eb71f7cd7b6"
    archive=ROOT/"experiments/iterations/20261104-moe-hardware-profile-recovered.json"
    assert sha256(archive.read_bytes()).hexdigest()=="0921485edb8a6494302288956fe1ef3614e7980f5fbf84fd831f27977a4c36b7"
    for fragment in [b"stage + sizeof(Smem::reduce)", b"kernel.multi_cast_size_a != 1",
                     b"cp.async.bulk.tensor.4d.shared::cta", b"cuLaunchKernel(reinterpret_cast<CUfunction>"]:
        assert fragment in patch,fragment
    wheel=urlopen(WHEEL,timeout=20).read()
    assert sha256(wheel).hexdigest()==WHEEL_SHA
    files={
        "scheduler.cuh":["tmp_m_blocks += CEIL_DIV(ctx.smem.expert_tokens[index], BlockShape::M);",
                         "m_block_id = m_block_id * kMultiCastSizeB + cluster_rank"],
        "memory/g2s_pipeline.cuh":["cluster_rank == 0 && thread_id < kNumStages", "ctx.cluster_rank() >= 1 && stage_id < kNumStages"],
        "kernel/humming_ws.cuh":["producer.wait_math_epilogue();", "consumer.arrive(kNumStages);"],
        "utils/storage.cuh":["StageStorage stages[kNumStages];","reduce[MAX(kWarpReduceSize, kBlockOutputSize)]"],
    }
    vendor={}
    with ZipFile(BytesIO(wheel)) as z:
        for name,snippets in files.items():
            data=z.read("humming/include/humming/"+name)
            assert all(s.encode() in data for s in snippets)
            vendor[name]=sha256(data).hexdigest()
    recorded=[]
    for case in json.loads(archive.read_bytes())["cases"]:
        route=case["routing"];counts=route["local_counts"]; width={32:8,256:24}[case["M"]]
        jobs=[math.ceil(n/width) for n in counts]
        request=table(sum(jobs),sum(n//2 for n in jobs),width)
        coords,transfers=scheduling_oracle(counts,width)
        assert coords==N_TILES*sum(jobs)
        assert transfers["same_weight"] == N_TILES*request["paired_row_tilesets"]
        read=case["counter_launch"]["metrics"]["dram__bytes_read.sum"]["value"]
        unique=sum(n>0 for n in counts)*BANK_BYTES
        assert unique==route["unique_weight_bytes"] and read>=unique
        recorded.append(dict(input_rows=case["M"],live_experts=route["U"],**request,
                             actual_other_kernel_read_mib=read/MIB,
                             unavoidable_unique_mib=unique/MIB,
                             physical_excess_over_unique_mib=(read-unique)/MIB,
                             physical_excess_fraction=(read-unique)/read,
                             observed_elapsed_us=case["conditional_same_traffic_4_8TB_s_calculation"]["profile_duration_us"]))
    uniform=[]
    for users,w in ((1,8),(2,8),(4,8),(8,8),(16,16),(32,24)):
        tokens=users*8
        p=8/256
        probabilities=[math.comb(tokens,i)*p**i*(1-p)**(tokens-i) for i in range(tokens+1)]
        assert abs(sum(probabilities)-1)<1e-10
        q=[math.ceil(i/w) for i in range(tokens+1)]
        expectation=lambda fn:32*sum(v*fn(n) for v,n in zip(probabilities,q))
        uniform.append(dict(users=users,width=w,**table(expectation(lambda n:n),expectation(lambda n:n//2),w)))
    rng=random.Random(216);cases=0;cells=0
    for w in (8,16,24):
        for counts in ([0]*32,[w*9+1]+[0]*31,[1]*32,[w*2]*32,
                       *([rng.choice((0,1,w-1,w,w+1,2*w+1,rng.randrange(256))) for _ in range(32)] for _ in range(48))):
            cells+=scheduling_oracle(counts,w)[0];cases+=1
    races=[]
    for name,k,n in (("W13",128,256),("W2",256,128)):
        m=8;groups=k//64;owners=512//groups
        reduce=max(16*n*32//128*(groups//2),m*n//8)*16
        scratch=(groups-1)*2*owners*16
        b=(m*k*2,m*k*2+n*k//2)
        overlap=min(reduce+scratch,b[1])-max(reduce,b[0])
        assert overlap>0
        races.append(dict(phase=name,m=8,retired_scratch=[reduce,reduce+scratch],
                          next_stage_zero_weight_region=b,potential_direct_overlap_bytes=overlap))
    print(json.dumps(dict(v215_patch=sha256(patch).hexdigest(),vendor_wheel=WHEEL_SHA,
                          checked_vendor_headers=vendor,physical_input=archive.name,
                          coordinate_cases=cases,ownership_checks=cells,recorded=recorded,
                          uniform=uniform,cross_job_race=races),indent=2))


if __name__=="__main__":
    main()
