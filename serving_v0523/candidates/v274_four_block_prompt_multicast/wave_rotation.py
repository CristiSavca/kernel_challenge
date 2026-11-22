"""Host-only experiment: cyclically distribute heavy M tiles on a smaller quad grid."""
import contextlib
import io
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
ns=dict(__file__=str(here/"grid_haircut.py"))
s=(here/"grid_haircut.py").read_text().split("rows=[]",1)[0]
exec(compile(s,str(here/"grid_haircut.py"),"exec"),ns)
cost=ns["cost"]
def rotated(hist,g=128,fan=4,n=4096):
    group_cols=n//128//fan
    mtiles=[ns["ns"]["issued"](min(256,r-off)) for r in hist for off in range(0,r,256)]
    total=len(mtiles)*group_cols
    groups=g//fan
    per_lane=[0]*groups
    seen=set()
    for wave,base in enumerate(range(0,total,groups)):
        live=min(groups,total-base)
        rotation=wave*group_cols%live
        for lane in range(live):
            idx=base+(lane+rotation)%live
            assert idx not in seen
            seen.add(idx)
            per_lane[lane]+=mtiles[idx//group_cols]
    assert seen==set(range(total))
    return max(per_lane)

rows=[]
for case in json.loads((here/"routes.json").read_text())["cases"]:
    items=[]
    for hist in case["histograms"]:
        a=cost(hist,132,2)["critical_issued_rows"];b=cost(hist,128,4)["critical_issued_rows"]
        items.append(dict(pair132=a,ordinary_quad128=b,rotated_quad128=rotated(hist)))
    rows.append(dict(tokens=case["tokens"],ranks=items))
rng=__import__("random").Random(27)
worst=(0,[]);best=(20,[])
for i in range(500):
    hist=[rng.choice((32,64,128,224,256,257,272,300,480,512,520,768,1024)) for _ in range(32)]
    ratio=rotated(hist)/cost(hist,132,2)["critical_issued_rows"]
    if ratio>worst[0]: worst=ratio,hist
    if ratio<best[0]: best=ratio,hist
print(json.dumps(dict(scope="Unshipped scheduling experiment. Cyclic worker rotation by one M tile within each contiguous persistent wave; exact coverage and equal K, uncertain runtime.",
    examples=rows,worst_admissible_ratio=worst,best_admissible_ratio=best),indent=2))
