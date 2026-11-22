"""Exact uniform-top-eight occupancy moments and reduced work for both native expert projections."""
import json,math
def estimate(users,width):
    tokens=users*8;p=8/256
    weights=[math.comb(tokens,i)*p**i*(1-p)**(tokens-i) for i in range(tokens+1)]
    data=dict(users=users,verify_tokens=tokens,native_width=width)
    def ex(fn):return 32*sum(w*fn(i) for i,w in enumerate(weights))
    def tilerows(total,unit):
        rem=total%width
        return total//width*width+((rem+unit-1)//unit)*unit
    tiles=ex(lambda i:(i+width-1)//width)
    old_rows=tiles*width;new_rows=ex(lambda i:tilerows(i,8));new_vectors=ex(lambda i:tilerows(i,2))
    data.update(expected_experts=ex(lambda i:i>0),expected_local_expert_tiles=tiles,
                native_MMA_rows=old_rows,issued_MMA_rows=new_rows,MMA_reduction_percent=100*(1-new_rows/old_rows),
                unchanged_packed_weights_and_scales_MiB=27*tiles)
    for phase,columns,groups in [('W13',8192,2),('W2',4096,4)]:
        data[phase]=dict(native_k_merge_sync=2*int(math.log2(groups))*math.ceil(width/16),
                         live_k_merge_sync=1,nominal_native_partial_shared_MiB=2*(groups-1)*4*columns*old_rows/2**20,
                         nominal_new_partial_shared_MiB=2*(groups-1)*4*columns*new_vectors/2**20)
    return data
print(json.dumps(dict(assumption="Independent uniform distinct top-eight expert IDs per token, 32 of 256 resident per TP8 rank. Shared vector store and read bytes only. No HBM or time measurement.",
    data=[estimate(u,w) for u,w in [(1,8),(2,8),(4,8),(8,8),(16,16),(32,24)]]),indent=2))
