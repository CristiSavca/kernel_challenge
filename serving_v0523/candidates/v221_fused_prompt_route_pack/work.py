"""Shape-only byte and launch model for V219 vs V221; no claim about cache hit rates or achieved bandwidth."""
import json
from pathlib import Path

def compare(rows, live, width=4096, topk=8, route_bytes=4):
    slots=rows*topk
    # V219: div reads/writes int64; index_select consumes divided indices; arange writes IDs;
    # full_like writes S defaults; index_copy reads original slot + new route ID, writes result.
    v219=live*(8+8+8+route_bytes+8+route_bytes+route_bytes)+slots*route_bytes
    # V221 reads one live int64 slot per gathered row and all original slot indices once for the map.
    v221=live*8+slots*(8+route_bytes)
    return dict(rows=rows,live_routes=live,topk=topk,width=width,route_bytes=route_bytes,
                activation_read_and_write=live*width*4,old_metadata_bytes=v219,
                new_metadata_bytes=v221,extra_logical_metadata=v221-v219,
                early_rowmap_bytes=slots*route_bytes,old_index_buffers_saved=live*(8+route_bytes),
                pack_ctas=live+(slots+255)//256,launches_removed_per_sparse_layer=4)

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    source=next(here.parent.glob('v197_*/independent_reach.json'))
    history=json.loads(source.read_bytes())['archive']
    print(json.dumps(dict(hardware_timing=False,bytes='logical array accesses; allocator, hardware cache and coalescing behavior are unmeasured',
        gathered_shape=compare(16384,16224),sensitivity=[compare(16384,r) for r in (2048,8192,16384,32768,65536)],
        archive=[dict(users=t['users'],rep=t['rep'],logical_passes=t['logical_grouped_passes'],
                      four_replaced_launches_times_62=t['logical_grouped_passes']*4*62,
                      mean_us_per_removed_launch_needed_for_5pct=t['net_ms_per_layer_call_for_5pct_raw']*1000/4)
                 for t in history]),indent=2))
