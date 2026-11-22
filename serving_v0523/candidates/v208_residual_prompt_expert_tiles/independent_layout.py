"""Invert native m64nNk16 FP32 lane/register coordinates and exercise grouped cursor boundaries."""
from bisect import bisect_right
from hashlib import sha256
from itertools import accumulate, product
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
full=(HERE/"patch.diff").read_bytes()
assert sha256(full).hexdigest()=="726ee2c1ab1e6e09a7cbe37a8c7aef14cafc33f979c50f7fddbb9766c213f82d"
source=full.decode()
for line in ("if (row >= live) continue;", "scheduler.current_shape_m - scheduler.m_offset",
             "_WIDTHS = (32, 96)", "mma.regs_c[0][j][0]", "kUseStreamK"):
    assert line in source

# Invert the SM90 m64nNk16 FP32-D ownership: hardware M is logical
# output channel; hardware N is the token row. d advances in 8-token bands.
slots=set()
per_row=[]
for token in range(176):
    row=[]
    for channel in range(256):
        warpgroup, local = divmod(channel,128)
        j, channel64 = divmod(local,64)
        warp, subrow = divmod(channel64,16)
        lane=32*warp + (subrow%8)*4 + (token%8)//2
        d=4*(token//8)+2*(subrow//8)+token%2
        word=(warpgroup,j,lane,d)
        assert word not in slots
        slots.add(word)
        row.append(d)
        assert 2*(lane%4)+d%2+8*(d//4)==token
        assert 128*warpgroup+64*j+16*(lane//32)+(lane%32)//4+8*((d//2)%2)==channel
    per_row.append(row)

checked=0
for live in range(1,177):
    width=32 if live<=32 else 96 if live<=96 else 176
    for token in range(176):
        words=per_row[token]
        if token<live:
            assert max(words)<width//2
            checked+=len(words)
        if token>=width:
            assert min(words)>=width//2

# Direct reference uses prefix sums of tile counts and bisect, independent
# of the monotone cursor and modulo arithmetic used by the published scheduler.
histograms=jobs=skipped=0
boundary=(0,1,32,33,96,97,175,176,177,208,209,272,273,352,353)
for counts in product(boundary, repeat=3):
    histograms+=1
    rows=[0,*accumulate(counts)]
    ntiles=[(v+175)//176 for v in counts]
    tile_ends=list(accumulate(ntiles))
    for stride in (1,2,5,11):
        for start in range(min(stride,tile_ends[-1])):
            expert=offset=previous=0
            for tile in range(start,tile_ends[-1],stride):
                offset+=(tile-previous)*176
                while offset>=counts[expert]:
                    offset-=ntiles[expert]*176
                    assert offset>=0
                    skipped+=1
                    expert+=1
                expected=bisect_right(tile_ends,tile)
                expected_offset=(tile-(tile_ends[expected-1] if expected else 0))*176
                assert expert==expected and offset==expected_offset
                assert rows[expert+1]-(rows[expert]+offset)==counts[expected]-expected_offset>0
                previous=tile
                jobs+=1

print(json.dumps(dict(all_lane_words=len(slots),live_FP32_coordinates=checked,all_final_row_counts=176,
                      expert_histograms=histograms,active_tile_seeks=jobs,crossed_expert_boundaries=skipped,
                      physical_device=False),indent=2))
