"""Check owner address and native post-shuffle row coverage for every small tile."""
from hashlib import sha256
import json
from pathlib import Path

here=Path(__file__).resolve().parent
manifest=json.loads((here/"validation.json").read_text())
assert sha256((here/"patch.diff").read_bytes()).hexdigest()==manifest["patch_sha256"]
counters=dict(owner_words=0,live_lane_components=0,clipped_lane_components=0)
for groups in (2,4):
    owners=512//groups
    for rows in (8,16,24):
        parts=2*(rows//8)
        for live in range(1,rows+1):
            writes=set()
            for tid in range(512):
                rank=tid//owners
                owner=tid%owners
                lane=tid%32
                vendor_rank=(tid//32)//(owners//32)
                vendor_warp=(tid//32)%(owners//32)
                assert (rank,owner)==(vendor_rank,vendor_warp*32+lane)
                for part in range(parts):
                    token_pair=8*(part%(rows//8))+2*(lane%4)
                    if token_pair>=live:continue
                    if rank:
                        address=(rank-1,part,owner)
                        assert address not in writes
                        writes.add(address)
                    else:
                        # Native WGMMA shared writer swaps one component with
                        # lane XOR 4; both lane groups represent this same pair.
                        for source in (lane,lane^4):
                            for parity in (0,1):
                                row=8*(part%(rows//8))+2*(source%4)+parity
                                assert row in (token_pair,token_pair+1)
                                if row<live:counters["live_lane_components"]+=1
                                else:counters["clipped_lane_components"]+=1
            for _,part,owner in writes:
                assert all((k,part,owner) in writes for k in range(groups-1))
            counters["owner_words"]+=len(writes)*4
print(json.dumps(dict(candidate=manifest["patch_sha256"],**counters,device=False),indent=2))
