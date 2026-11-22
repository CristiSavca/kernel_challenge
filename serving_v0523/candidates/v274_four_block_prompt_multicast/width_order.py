"""Group full and residual M tiles by issued width to split long tails across CTAs."""
from pathlib import Path
import json

here=Path(__file__).resolve().parent
counts=json.loads((here/"routes.json").read_text())["cases"]
raw=json.loads((here/"grid_haircut.json").read_text())["reconstructed"]

def sorted_cost(hist,g,fan):
    widths=(256,224,128,32)
    tiles=[]
    for count in hist:
        for offset in range(0,count,256):
            width=min(w for w in widths if w>=min(256,count-offset))
            tiles.append(width)
    # Every consecutive cluster-sized group of N channels remains a legal multicast.
    work=[w for w in sorted(tiles,reverse=True) for _ in range(4096//128//fan)]
    bins=[sum(work[idx::g//fan]) for idx in range(g//fan)]
    assert sum(bins)==sum(work)
    assert max(bins)-min(bins)<=256
    return dict(max=max(bins),min=min(bins),mean=sum(bins)/len(bins))
out=[]
for src,baseline in zip(counts,raw):
    trials=[]
    for h,ref in zip(src["histograms"],baseline["ranks"]):
        trials.append(dict(pair132=ref["pair132"]["critical_issued_rows"],
                default_quad128=ref["quads"]["128"]["critical_issued_rows"],ordered_pair132=sorted_cost(h,132,2),ordered_quad128=sorted_cost(h,128,4)))
    out.append(dict(tokens=src['tokens'],ranks=trials))
print(json.dumps(dict(shape="K/N=4096, 32 local experts; descending actual residual width; exact number of clusters and same arithmetic, CPU schedule estimate",
                      groups=out),indent=2))
