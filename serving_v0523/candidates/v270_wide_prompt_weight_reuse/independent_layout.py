"""Check actual V270 reader expressions against the vendor's half-group and epilogue geometry."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

stage, vendor = map(Path, sys.argv[1:])
reader = stage/"python/sglang/kernels/jit/csrc/muse_spark/prompt_weight_pipeline.cuh"
includes = vendor/"include/humming"
native_b = includes/"memory/s2r_loader/loader_b.cuh"
native_bs = includes/"memory/s2r_loader/loader_bs.cuh"
writer = includes/"epilogue/smem_writer.cuh"
dequant = includes/"datatype/dequant.cuh"
code = reader.read_text()
assert "warp_id % 2 * kLoadIters + j" in native_b.read_text()
assert "n_warp_id % kNumWarpsPerPackBlock + pack_block_id * kPackBlockSize" in native_bs.read_text()
assert "uint32_t qb_index = kResultPackedNumSourceBits * index / 32;" in dequant.read_text()
assert "col2 = (n_warp_id / count) * (BlockShape::M / kNumWriteSplits * 64 / 2)" in writer.read_text()
def expr(pattern, source=code):
    return compile(re.search(pattern,source)[1].replace(" / ", " // "),"<staged address>","eval")

wide_b = expr(r"words\[0\] = packed\[(.+?)\];")
wide_s = expr(r"const uint2 pair = scales\[(.+?)\];")
narrow_s = expr(r"this->mma.scales\[k % 2\] = scales\[(.+?)\];")
assert "warp % 2 ? pair.y : pair.x" in code
assert "muse_scale_bits((w >> 16) & 255) | (muse_scale_bits(w >> 24) << 16)" in code
writer_code = writer.read_text()
col_one = expr(r"uint32_t col1 = (.+?);",writer_code)
col_two = re.search(r"uint32_t col2 = (.+?);",writer_code)[1]
col_two = compile(col_two.replace("BlockShape::M","block_m").replace(" / "," // "),"<native output address>","eval")

def output(warp,width,laneid,m,r,smem,split):
    sub_row = (laneid % 4)*2 + (laneid % 8)//4
    env = dict(n_warp_id=warp,count=64//width,row_8x8block=r,laneid=laneid,sub_row=sub_row,
               smem=smem,block_m=256,kNumWriteSplits=split)
    return (8*m+sub_row)*32 + eval(col_one,{},env) + eval(col_two,{},env)

words = scales = writes = 0
for k in range(4):
    coverage = set()
    for warp in range(8):
        source_warp, half = divmod(warp,2)
        for lane in range(32):
            env = dict(warp=warp,lane=lane,k=k)
            new_word = eval(wide_b,{},env)
            # Native 32-channel warp loads uint2 from the half-group. Humming
            # dequant_b1248 takes one complete word per j and eight FP4 codes per word.
            old_pair = (32*(source_warp//2) + lane)*2 + source_warp%2
            assert new_word == k*256 + 2*old_pair + half
            coverage.add(new_word)
            new_scale_half = 2*eval(wide_s,{},env) + half
            assert new_scale_half == 2*eval(narrow_s,{},env|dict(warp=source_warp))+half
            # Native loader_bs layout1, halfword offsets for 16 channels at K16.
            assert new_scale_half == 32*(warp//4)+4*(lane//4)+warp%4
            words += 1
            scales += 2
    assert coverage == set(range(k*256, (k+1)*256))

for align in range(8):
    for split in (1,2):
        touched = set()
        for w in range(8):
            for lane in range(32):
                for m in range(32//split):
                    for r in range(2):
                        new = output(w,16,lane,m,r,align,split)
                        old = output(w//2,32,lane,m,2*(w%2)+r,align,split)
                        assert new == old and new not in touched
                        touched.add(new); writes += 1
        assert touched == set(range(256*128//(2*split)))
print(json.dumps(dict(packed_weight_word_addresses=words, fp4_code_coordinates=8*words,
    scale_byte_coordinates=scales, native_half2_output_addresses=writes, shared_swizzle_phases=8,
    write_splits=[1,2], hashes={p.name:sha256(p.read_bytes()).hexdigest()
    for p in (reader,native_b,native_bs,writer,dequant)},scope="Static index equivalence for one 256 x 128 x 64 tile, including every K16, both 64-channel groups and all native epilogue swizzles. No hardware execution."),indent=2))
