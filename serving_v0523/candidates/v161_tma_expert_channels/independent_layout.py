"""Check TMA X/Y byte addresses against V153 vector loads and per-CTA barrier phases."""
import hashlib
import json
from pathlib import Path
import sys

src=Path(sys.argv[1])/'python/sglang/kernels/jit/csrc/muse_spark'
tma=(src/'expert_tma_route_pair.cuh').read_text()
pair=(src/'expert_route_pair.cuh').read_text()
for t in ('{weight ? 256u : 64u, kKSteps}', '2 * x, y, barrier','x / 2, y, barrier',
          'uint32_t phase = 0;', 'sizeof(dest.weights) + sizeof(dest.scales)',
          'alignas(128) PackedTile<Gate> buffers[2]', 'mbarrier.init.shared::cta',
          'fence.proxy.async.shared::cta', 'mbarrier.arrive.expect_tx.shared::cta',
          'mbarrier.try_wait.parity.shared::cta', '*phase ^= 1'):
    assert t in tma,t
assert 'if (first + kKSteps < tiles)' in pair and 'stage.publish();' in pair
vector_slices=0; byte_samples=0; spans=0; handoffs=0; repeated=0
for hidden,intermediate in [(128,128),(256,384),(384,256),(4096,4096),(6144,4096)]:
    for gate in (False,True):
        width=intermediate if gate else hidden; k=hidden if gate else intermediate
        planes=1+int(gate); dims_w=2*width*planes; dims_s=width*planes//2
        shape_bytes=(planes*8*256*4,planes*8*64*2,8*8*12*4)
        assert sum(shape_bytes)%128==0 and shape_bytes[0]%128==0
        assert 8*256*4%128==8*64*2%128==0
        parity=0
        for e in (0,1,31,0):
            reads=set()
            for first in range(0,k//16,8):
                y=e*(k//16)+first
                assert first+8<=k//16
                for s in range(8):
                    for col in sorted({0,128 if width>=256 else 0,width-128}):
                        for plane in range(planes):
                            x=col+plane*width
                            base=(y+s)*(planes*width)+x
                            for off in range(0,256,4):
                                weight_tma=( (y+s)*dims_w+2*x+off)*4
                                weight_v153=(2*base+off)*4
                                assert weight_tma==weight_v153 and weight_tma%16==0
                            for off in range(0,64,8):
                                scale_tma=((y+s)*dims_s+x//2+off)*2
                                scale_v153=base+2*off
                                assert scale_tma==scale_v153 and scale_tma%16==0
                            vector_slices+=72;byte_samples+=1152;spans+=2
                    assert first+s not in reads;reads.add(first+s)
                parity ^= 1;handoffs+=1
            assert reads==set(range(k//16));repeated+=1
        assert parity==0
print(json.dumps(dict(header_sha256=hashlib.sha256(tma.encode()).hexdigest(),vector_source_sha256=hashlib.sha256(pair.encode()).hexdigest(),
    source_vector_ranges=vector_slices,transferred_bytes_compared=byte_samples,rectangular_slices=spans,
    simulated_barrier_transactions=handoffs,job_transitions=repeated,
    limitation='Exact address identities and expected-phase model only. No actual TMA transaction or wall time.'),indent=2))
