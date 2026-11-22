"""Traffic of full-input and feature-sharded expert unpool, with rounding witnesses."""
import json
import struct


def f32(x):
    return struct.unpack("f", struct.pack("f",x))[0]


def bf16(x):
    n, = struct.unpack("I",struct.pack("f",x))
    return struct.unpack("f",struct.pack("I",(n+0x7fff+((n>>16)&1)) & 0xffff0000))[0]


def model():
    cases=[]
    for rows in (8,32,128,256):
        x=rows*4096*2*7//8
        y=2*x
        cases.append(dict(rows=rows, per_rank_layer_ring_send_B=dict(
            parent=2*x+(y if rows<=8 else 0),
            V142_output_sharded=2*x+y,
            input_sharded_with_bf16_partial= x+2*y,
            input_sharded_with_fp32_output_reduce_scatter_then_bf16_gather=x+3*y,
            input_sharded_with_fp32_output_all_reduce=x+4*y),
            parent_unpool_cold_MiB=8 if rows<=8 else 64, input_sharded_unpool_cold_MiB=8))
    bf_witness=dict(full=bf16(f32(f32(256.+1.)-256.)),
                    partial=bf16(f32(bf16(256.+1.)+bf16(-256.))))
    fp_witness=dict(full=bf16(f32(f32(2.**25-2.**25)+1.)),
                    partial=bf16(f32(2.**25+f32(-2.**25+1.))))
    assert bf_witness==dict(full=1.,partial=0.) and fp_witness==dict(full=1.,partial=0.)
    return dict(TP=8,hidden=8192,expert_hidden=4096,rows=cases,
                bfloat16_partial_rounding_example=bf_witness, float32_partition_order_example=fp_witness,
                ring="ideal balanced unidirectional sends per rank, excluding metadata and local transpose",
                timings=False)

if __name__=="__main__":
    print(json.dumps(model(),indent=2))
