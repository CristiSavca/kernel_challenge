"""Replay the measured V41 prefill geometry and price the overlap of two output stripes."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TP, HIDDEN, OUTPUT, LAYERS = 8, 1024, 8192, 62


def shape(n):
    message = 2*n*OUTPUT
    owner = 2*((n+TP-1)//TP)*OUTPUT
    return dict(eligible=n>=2048 and n*OUTPUT>=64*1024*1024,
                per_rank_flop=2*n*OUTPUT*HIDDEN,
                native_full_message_bytes=message, stripe_message_bytes=message//2,
                single_rank_owner_bytes=owner, common_weight_bytes=2*OUTPUT*HIDDEN,
                extra_logical_input_read_bytes=2*n*HIDDEN, local_owner_copy_read_write_bytes=2*owner,
                implied_ring_AR_total_send_bytes=2*message*(TP-1)//TP,
                implied_ring_RS_total_send_bytes=message*(TP-1)//TP,
                explicit_live_dense_inplace_AR_bytes=dict(parent=message, striped=message+owner),
                explicit_live_dense_outplace_AR_bytes=dict(parent=2*message, striped=message+owner+message//2))


def report():
    source = json.loads((HERE.parent/'v251_hicache_residency/trace.json').read_bytes())['v41']
    wall = json.loads((HERE.parent/'v193_scheduler_cache_poll/trace.json').read_bytes())['V41']['rounds']
    file = next(p for p in source['sources'] if p['path'].endswith('server.log'))
    raw = (ROOT/file['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==file['sha256']
    groups=[]
    pattern=re.compile(r'Prefill batch, #new-seq: \d+, #new-token: (\d+)')
    for line in raw.decode().splitlines():
        if 'Cache flushed successfully!' in line:
            groups.append([])
        elif groups and 'TP0 EP0' in line and (batch:=pattern.search(line)):
            groups[-1].append(int(batch[1]))
    assert len(groups)==len(wall)==7
    observed=[]
    for ns, info in zip(groups,wall):
        assert sum(ns)==info['real_prefill_rows'] and len(ns)==info['prefill_scheduler']['passes']
        chosen=[p for n in ns if (p:=shape(n))['eligible']]
        estimates=[]
        for rate, tf in ((450,989),(225,400)):
            # For a ring, half the parent reduction can overlap GEMM two.
            arn=sum(min(p['per_rank_flop']/(2*tf*1e12),p['implied_ring_AR_total_send_bytes']/(2*rate*1e9)) for p in chosen)
            rsn=sum(min(p['per_rank_flop']/(2*tf*1e12),p['implied_ring_RS_total_send_bytes']/(2*rate*1e9)) for p in chosen)
            estimates.append(dict(directional_GB_per_s=rate, dense_BF16_TFLOP_per_s=tf,
                                  ring_AR_perfect_overlap_s=arn*LAYERS, ring_RS_perfect_overlap_s=rsn*LAYERS,
                                  AR_fraction_recorded_wall=arn*LAYERS/info['wall_s']))
        observed.append(dict(users=info['concurrent_users'],rep=info['rep'],wall_s=info['wall_s'],
                             eligible_batches=len(chosen),observed_batches=len(ns),scenarios=estimates))
    return dict(patch_sha256=json.loads((HERE/'validation.json').read_bytes())['patch_sha256'],
                trace_source=file, full_chunk=shape(16384), observations=observed,
                limits='Full chunk named buffers exclude GEMM/collective workspaces and graph allocator. A ring has the same total send volume with two stripes. The recorded default enables NVLS, whose actual algorithm, cost and overlap are unmeasured. Archived V41 logical row counts are used; graph padding can differ.', new_H200_run=False)


if __name__=="__main__":
    print(json.dumps(report(),indent=2))
