"""Exact binomial expectation for a cold-prompt model with 256 experts and distinct uniform top eight."""
import json, math

def issued(count, widths):
    whole, tail = divmod(count,176)
    return 176*whole + (next(w for w in widths if w >= tail) if tail else 0)

def estimate(q):
    # Use a recurrence centered on the mode to avoid underflow. All support is covered.
    p=8/256
    mode=int((q+1)*p)
    weights=[0.]*(q+1)
    weights[mode]=1.
    for x in range(mode,q):
        weights[x+1]=weights[x]*(q-x)*p/((x+1)*(1-p))
    for x in range(mode,0,-1):
        weights[x-1]=weights[x]*x*(1-p)/((q-x+1)*p)
    total=math.fsum(weights)
    def expect(f):
        return math.fsum(f(n)*w for n,w in enumerate(weights) if w)/total
    old=32*expect(lambda n:issued(n,[176]))
    new=32*expect(lambda n:issued(n,[32,96,176]))
    fine=32*expect(lambda n:issued(n,[32,96,160,176]))
    return dict(tokens=q,mean_rank_routes=q,expected_176_M_tiles_per_rank=old/176,
                prior_MMA_rows=old,new_MMA_rows=new,compute_work_ratio=old/new,
                reduction_percent=100*(1-new/old),four_width_ratio=old/fine,
                unchanged_weight_scale_tile_requests_MiB=old/176*27,
                removed_TFLOP_per_layer=(old-new)*6*4096**2/1e12)
print(json.dumps(dict(assumption="independent uniform 8-of-256 selection per input token, 32 local experts. Roofline work, no elapsed time.",
                      data=[estimate(q) for q in (8192,10240,12288,14336,16384)],
                      expert_boundary_counts={n:[issued(n,[176]),issued(n,[32,96,176])] for n in (0,1,32,33,96,97,175,176,177,352,353,511,512,527,528,529)}),indent=2))
