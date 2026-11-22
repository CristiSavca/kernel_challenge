"""Check V153's shared layout, hardware MMA lane owners, and guard from the complete export."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HEADER = "python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh"
DIGEST = "c44ec0bef375ae6ce1fcf1e8540959719d9323345811ac5fce1fec9fc14bce9d"


def source():
    if len(sys.argv) == 2:
        return Path(sys.argv[1]).read_text()
    patch = (HERE / "patch.diff").read_text()
    section = patch.split("diff --git a/" + HEADER + " b/" + HEADER + "\n", 1)[1]
    section = section.split("\ndiff --git ", 1)[0].split("\n@@ ", 1)[1].split("\n", 1)[1]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+")) + "\n"


def main():
    src = source()
    assert sha256(src.encode()).hexdigest() == DIGEST
    for excerpt in (
        "uint32_t weights[Gate ? 2 : 1][kKSteps][256]",
        "uint16_t scales[Gate ? 2 : 1][kKSteps][64]",
        "uint32_t inputs[kKSteps][kRows][12]",
        "part = (warp + lane / 8) % 4",
        "wi = 128 * band + 4 * lane + part",
        "si = 32 * band + 4 * (lane / 4) + part",
        "src.inputs[s][lane / 4][lane % 4]", "lo = x[0], hi = x[4]",
        "column + 64 * band + 16 * part + lane / 4 + 8 * (i / 2)",
        "input + 8 * lane", "const uint32_t valid_bytes = slot < 0 ? 0u : 16u",
        "cp.async.wait_group 0;", "__syncthreads();",
        "const uint32_t phase = (first / kKSteps) % 2", "buffers[1 - phase]",
        "if (aligned_packed_projection(p))", "return (bits & 15u) == 0",
    ):
        assert excerpt in src, excerpt
    producer_vectors = {}
    for planes in [1, 2]:
        # Byte coordinates include structure member placement and both buffers.
        weights = planes * 8 * 256 * 4
        scales = planes * 8 * 64 * 2
        inputs = 8 * 8 * 12 * 4
        size = weights + scales + inputs
        assert size * 2 == (24576 if planes == 1 else 43008)
        for phase in [0, 1]:
            base = phase * size
            addr = {}
            for thread in range(256):
                for i in range(thread, planes * 8 * 64, 256):
                    plane, step, vector = i // 512, (i // 64) % 8, i % 64
                    start = base + ((plane * 8 + step) * 256 + vector * 4) * 4
                    for byte in range(16):
                        assert (start+byte) not in addr
                        addr[start+byte] = ('w', plane, step, 16*vector+byte)
                for i in range(thread, planes * 8 * 8, 256):
                    plane, step, vector = i // 64, (i // 8) % 8, i % 8
                    start = base + weights + ((plane * 8 + step) * 64 + vector * 8) * 2
                    for byte in range(16):
                        assert (start+byte) not in addr
                        addr[start+byte] = ('s', plane, step, 16*vector+byte)
                lane, route = thread % 32, thread // 32
                if lane < 16:
                    step, half = divmod(lane, 2)
                    start = base + weights + scales + ((step*8+route)*12+4*half)*4
                    assert start % 16 == 0
                    for byte in range(16):
                        assert (start+byte) not in addr
                        addr[start+byte] = ('x', step, route, 16*half+byte)
            assert min(addr) == base and max(addr) < base+size
            consumer = set()
            owns = Counter()
            for warp in range(8):
                wbank, xbank = set(), set()
                for lane in range(32):
                    band, part = warp//4, (warp+lane//8)%4
                    wi, si = 128*band+4*lane+part, 32*band+4*(lane//4)+part
                    wbank.add(wi%32)
                    xbank.add((12*(lane//4)+lane%4)%32)
                    # The four lanes that cooperate along K for each MMA A row
                    # must select the same incumbent 16-channel fragment.
                    assert all((warp+l//8)%4 == part for l in range(lane//4*4,lane//4*4+4))
                    for i in range(4):
                        n = 64*band+16*part+lane//4+8*(i//2)
                        r = 2*(lane%4) + i%2
                        owns[n,r] += 1
                    for step in range(8):
                        for plane in range(planes):
                            for byte in range(4):
                                pos = base+4*((plane*8+step)*256+wi)+byte
                                assert addr[pos] == ('w', plane,step,4*wi+byte)
                                consumer.add(pos)
                            for byte in range(2):
                                pos = base+weights+2*((plane*8+step)*64+si)+byte
                                assert addr[pos] == ('s',plane,step,2*si+byte)
                                consumer.add(pos)
                        for byte in range(4):
                            for kd in [0,16]:
                                pos = base+weights+scales+4*((step*8+lane//4)*12+lane%4)+kd+byte
                                assert addr[pos] == ('x',step,lane//4,4*(lane%4)+kd+byte)
                                consumer.add(pos)
                assert len(wbank) == len(xbank) == 32
            assert len(owns) == 128*8 and set(owns.values()) == {1}
            assert consumer == addr.keys()
            producer_vectors[planes] = len(addr)//16
    return dict(header=DIGEST, sparse_and_zero_filled_rows="same addresses; source-size selects zero or 16B",
                double_buffer_bytes={"gate":43008,"down":24576},
                sixteen_byte_vectors_per_buffer={"gate":producer_vectors[2],"down":producer_vectors[1]},
                logical_output_cells_per_cta=1024, distinct_weight_and_input_banks_per_warp=32,
                guard="16-byte input/weight/scale addresses; inherited 2-warp projection otherwise",
                hardware=False)


if __name__ == "__main__":
    print(json.dumps(main(),indent=2))
