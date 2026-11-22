"""Replay serial grouped cold jobs and a balanced lower bound using V279 production shape decisions."""
import ast
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import sys
from types import SimpleNamespace as NS

here=Path(__file__).resolve().parent
model=Path(sys.argv[1])/"python/sglang/srt/layers/muse_spark"
route_file=here.parent/"v274_four_block_prompt_multicast/routes.json"
traffic=here.parent/"v277_multicast_prompt_gate_up/traffic.json"
down=here.parent/"v274_four_block_prompt_multicast/schedule.json"

class Tensor:
    def __init__(self,rows):self.ranges=rows
    def flatten(self):return self
config=(model/"prompt_weight_pipeline.py").read_text()
function=next(n for n in ast.parse(config).body if getattr(n,"name","")=="_prompt_grid_ranges")
scope=dict(torch=NS(tensor=lambda x,**_:Tensor(x),int64=None))
exec(ast.unparse(function),scope)
ranges=scope["_prompt_grid_ranges"](32,("narrow",),("wide",)).ranges

def tiles(hist,width):
    for count in hist:
        for begin in range(0,count,width):
            live=min(width,count-begin)
            yield next(s for s in ((32,96,176) if width==176 else (32,128,224,256)) if live<=s)

def phase(hist,width,ncols,n_tile,grid):
    issued=list(tiles(hist,width))
    time=[0]*grid
    for i,rows in enumerate(issued):
        for j in range(ncols//n_tile):
            time[(i*(ncols//n_tile)+j)%grid]+=rows*n_tile
    return dict(tail=max(time),work=sum(time),tiles=len(issued))

def run(hist,grid):
    n=sum(hist)
    selected=next(choice for lo,hi,choice in ranges if lo<n<=hi)
    w2m,w2n=(256,128) if selected=="wide" else (176,256)
    a=phase(hist,176,8192,256,grid)
    b=phase(hist,w2m,4096,w2n,grid)
    native=a["tail"]+b["tail"]
    ideal=(a["work"]+b["work"])/grid
    windows={}
    for size in (1,2,4,8,16,32):
        groups=[hist[i:i+size] for i in range(0,len(hist),size)]
        durations=[phase(g,176,8192,256,grid)["tail"]+phase(g,w2m,4096,w2n,grid)["tail"] for g in groups]
        windows[size]=dict(projection_launches=2*len(groups),intermediate_MiB=max(map(sum,groups))*4096*2/2**20,
                           native_issue_time_ratio=sum(durations)/native)
    return dict(live_routes=n,w2=selected,grid=grid,W13_tiles=a["tiles"],W2_tiles=b["tiles"],native=native,
                perfect_schedule_lower_bound=ideal,optimistic_projection_speed_ratio=native/ideal,
                windows=windows)
out=[]
for record in json.loads(route_file.read_bytes())["cases"]:
    ranks=[]
    for r,hist in enumerate(record["histograms"]):
        result={grid:run(hist,grid) for grid in (128,132)}
        # The four-block V277 persistent loop has exactly these critical issued rows.
        check=next(c for c in json.loads(traffic.read_bytes())["cases"] if c["prompt_tokens"]==record["tokens"])["ranks"][r]
        for grid in result:
            assert phase(hist,176,8192,256,grid)["tail"]//256 == check["four_block"][str(grid)]["critical_issued_rows"]
        if result[128]["w2"]=="wide":
            d=next(c for c in json.loads(down.read_bytes())["results"] if c["tokens"]==record["tokens"])["ranks"][r]
            assert d["issued_tensor_rows"] == sum(tiles(hist,256))
        ranks.append(result)
    out.append(dict(rows=record["tokens"],ranks=ranks))
print(json.dumps(dict(source_sha256={str(p.name):sha256(p.read_bytes()).hexdigest() for p in (route_file,traffic,down)},
    native_range_function_sha256=sha256(ast.unparse(function).encode()).hexdigest(),results=out,
    method="V279 source picks wide/narrow W2 from live routes. Each physical CTA processes its strided jobs; work in issued tensor rows x N channels, full K shared. Full-wave occupancy assumed at 128 or 132 CTAs under each shape. Perfect W13/W2 mixes ignore all data dependencies, register/cluster constraints, nonlinear costs and synchronization. Window estimates reset the persistent grid at expert boundaries and preserve the initial full-shape geometry; a smaller real call may select a different kernel. These are issue-unit bounds, not device timings."),indent=2))
