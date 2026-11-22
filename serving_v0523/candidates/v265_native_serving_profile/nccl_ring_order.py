"""Check one allowed equal-ring, equal-chunk mapping in the published NCCL 2.30.7 kernel."""
import json
import struct


def fp32(val):
    return struct.unpack('=f', struct.pack('=f', val))[0]


def bf16(val):
    bits = struct.unpack('=I', struct.pack('=f', val))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('=f', struct.pack('=I', bits))[0]


def main():
    ranks, chunk, row = 8, 65536, 16
    values = [2**24, 1, -(2**24), 0, 0, 0, 0, 0]
    assert [bf16(v) for v in values] == values
    output = []
    for width in (8192, 4096):
        offset = row * width
        # See src/device/all_reduce.h: a ring's first send picks (ringIx-1) % nranks.
        finishing_index = offset // chunk % ranks
        rank_order = [(finishing_index + 1 + j) % ranks for j in range(ranks)]
        sums = {}
        for name, rounded in (("BF16_per_hop", bf16), ("FP32_per_hop", fp32)):
            total = values[rank_order[0]]
            for rank in rank_order[1:]:
                total = rounded(values[rank] + total)
            sums[name] = bf16(total)
        output.append(dict(width=width, flat_offset=offset, ring_order=rank_order, result=sums))
    assert output[0]["result"] == dict(BF16_per_hop=0.0, FP32_per_hop=0.0)
    assert output[1]["result"] == dict(BF16_per_hop=1.0, FP32_per_hop=1.0)
    return dict(row=row, column=0, full_shape=[16384,8192], half_shape=[16384,4096], TP=ranks,
                assumed_channel_chunk_elements=chunk, assumed_ring=list(range(ranks)), rank_inputs=values,
                cases=output, source="https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/all_reduce.h#L11-L57",
                scope="Source-level counterexample to size- or ring-family-only guarantees. One full interior loop, same ring/channel 0 and fixed per-channel chunk width. No claim that NCCL chooses this launch geometry on the judge.")


if __name__=="__main__":
    print(json.dumps(main(), indent=2))
