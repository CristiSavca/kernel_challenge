"""Bounded host-only look at reassigning M tiles inside each already-resident wave."""
import json
from pathlib import Path
here=Path(__file__).resolve().parent
old=here.parent/"v274_four_block_prompt_multicast"
cases=json.loads((old/"routes.json").read_text())["cases"]
baseline=json.loads((old/"grid_haircut.json").read_text())["reconstructed"]
out=[]
for c,reference in zip(cases,baseline):
    results=[]
    for hist,rank in zip(c["histograms"],reference["ranks"]):
        tiles=[(next(w for w in (32,128,224,256) if w>=min(256,n-off)),e,off) for e,n in enumerate(hist) for off in range(0,n,256)]
        costs=[0]*4
        for start in range(0,len(tiles),4):
            live=tiles[start:start+4]
            slots=sorted(range(len(live)),key=lambda lane:(costs[lane],lane))
            order=sorted(range(len(live)),key=lambda i:(-live[i][0],i))
            assert {live[i] for i in order}==set(live)
            for slot,i in zip(slots,order):
                costs[slot]+=live[i][0]
        results.append(dict(native_pair=rank["pair132"]["critical_issued_rows"],
            greedy_wave_quad=max(costs),spread=max(costs)-min(costs)))
    out.append(dict(tokens=c["tokens"],ranks=results))
print(json.dumps(dict(scope="Host lower-cost scheduling investigation only. K/N=4096, grid=128, each wave contains four consecutive M tiles with 32 N channels blocks each. Preserve active weights per wave and original active slots in the final partial wave. Most expensive tile assigned to least loaded eligible prior slot, using cumulative widths.",cases=out),indent=2))
