"""Trace every widened RF source word to the same packed global word used by two compact gate/up tiles."""
import hashlib
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
vendor=Path(sys.argv[1])/"include/humming"
prior=next(here.parent.glob("v275_*/operands.json"))
expected=json.loads(prior.read_bytes())
paths=["memory/s2r_loader/loader_b.cuh","memory/s2r_loader/loader_bs.cuh"]
hashes={p:hashlib.sha256((vendor/p).read_bytes()).hexdigest() for p in paths}
assert all(expected["actual_vendor_headers"][path] == digest for path, digest in hashes.items())
maps=json.loads((here/"independent_w13_maps.json").read_bytes())
assert [x["block_width"] for x in maps["cases"]]==[256,512]
# In a compact [gate | up] TMA destination the channel groups are already paired by plane.
def packed(word,band,tile,per_channel):
    krow, remainder=divmod(word,2*band*per_channel)
    plane, in_plane=divmod(remainder,band*per_channel)
    return ((krow*2+plane)*4096+tile*band)*per_channel+in_plane

counts=dict(weight_words=0,scale_bytes=0,wrong_if_two_native_tiles_are_concatenated=0)
for tile in range(4096//256):
    for kr in range(8):  # each of the two K64 workers, four K16 fragments
        for warp in range(8):
            for lane in range(32):
                for scale in (False, True):
                    width=8 if scale else 4
                    band_words=width//2
                    for frag in range(width):
                        group=2*warp+frag//band_words
                        plane,chunk=divmod(group,8)
                        sub_tile,quarter=divmod(chunk,4)
                        old_warp=4*plane+quarter
                        if scale:
                            small=kr*256+((old_warp//2)*16+(lane//4)*2+old_warp%2)*4+frag%4
                            wide=kr*512+(warp*8+lane//4)*8+frag
                            per_channel=1
                        else:
                            small=kr*512+((old_warp//2*32+lane)*2+old_warp%2)*2+frag%2
                            wide=kr*1024+(warp*32+lane)*4+frag
                            per_channel=2
                        lhs=packed(small,128,2*tile+sub_tile,per_channel)
                        rhs=packed(wide,256,tile,per_channel)
                        assert lhs==rhs, (scale,tile,kr,warp,lane,frag,lhs,rhs)
                        counts['scale_bytes' if scale else 'weight_words']+=1
                        # Two entire narrow gate/up tiles concatenate as gate0/up0/gate1/up1,
                        # whereas the compact wide epilogue expects gate0/gate1/up0/up1.
                        local=wide%(512*per_channel)
                        wrong_tile,word=divmod(local,256*per_channel)
                        wrong=packed(kr*256*per_channel+word,128,tile*2+wrong_tile,per_channel)
                        counts['wrong_if_two_native_tiles_are_concatenated']+=lhs!=wrong
assert counts['wrong_if_two_native_tiles_are_concatenated']>0
print(json.dumps(dict(counts=counts,vendor_sha256=hashes,source_descriptor_sha256=maps['source_sha256'],
    physical_gpu=False,scope="Pinned Humming RF packed weight-word and scale-byte addressing combined with the exercised source TMA gate/up layout. All output tiles of a 4096-wide expert at all lane/fragment coordinates in one K128 stage. The same physical row offset is added at later K and expert indices."),indent=2))
