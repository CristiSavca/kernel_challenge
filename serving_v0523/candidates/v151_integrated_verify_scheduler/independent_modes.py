"""Compare the frozen V147 paged forward against V151 on four real verify geometries."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
fixture=runpy.run_path(str(HERE.parent/"v132_verify_scheduler_metadata/verify_dispatch.py"))
PARENT="cd089ab0b54e9fa2ee4c1cfdcb3f354370748d85"


class Forbidden:
    def view(self,*args):
        raise AssertionError("verification touched an ordinary-prefill physical span")
    def __getattr__(self,name):
        raise AssertionError("verification dereferenced an already-written raw V: "+name)


def run(stage):
    torch.set_num_threads(1)
    restored=0
    num_replays=0
    shape_rows=[]
    host=SimpleNamespace(cuda_stream=31)
    groups=[fixture["build"](stage),fixture["build"](stage,PARENT)]
    vendor=json.loads((HERE.parent/"v132_verify_scheduler_metadata/splits.json").read_text())
    expected_splits={r["batch"]:(r["full"],r["swa"]) for r in vendor["shapes"]
                     if r["page_size"]==1 and r["physical_k_capacity"]==131072}
    with patch.object(torch.cuda,"current_device",return_value=0),patch.object(torch.cuda,"current_stream",lambda _:host):
        for b in (1,4,16,32):
            for case in ("standard","short","split-streams","wrong-width","flag-off","ragged"):
                cols=512 if case=="short" else 131072
                md,fb,layer,q,kv,_=fixture["tensors"](b,8,1,cols,16,2)
                md.prefill_kv_span=Forbidden()
                md.window_kv_span=Forbidden()
                fb.spec_info.ragged_verify_layout=1 if case=="ragged" else None
                fb.spec_info.num_tokens_per_req=7 if case=="wrong-width" else 8
                for backend,owner,outer,trace in groups:
                    trace.fa.clear();trace.nodes.clear();trace.prepare.clear()
                    backend._inspect_prefill_spans=True
                    backend._verify_schedule_sms=132
                    backend.page_size=1
                    backend.forward_metadata=md
                    backend.token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda _:kv)
                    owner._share_verify_schedule=case!="flag-off"
                    def target(ids,pos,batch,input_embeds=None):
                        for i in range(62):
                            host.cuda_stream=31+(i%2) if case=="split-streams" else 31
                            layer.sliding_window_size=-1 if i%4==0 else 2047
                            backend.forward_extend(q,None,Forbidden(),layer,batch)
                        return q
                    owner.model=target
                    outer(owner,q,None,fb)
                    assert fb._fa3_verify_schedules is None
                    restored+=1
                current,parent=groups[0][3],groups[1][3]
                assert fixture["signatures"](current)==fixture["signatures"](parent)
                assert len(current.fa)==len(parent.fa)==62
                assert sum(call["page_table"] is md.page_table for call in current.fa)==16
                assert sum(call["page_table"] is md.swa_page_table for call in current.fa)==46
                prepares=2 if case=="standard" else 3 if case=="split-streams" else 0
                assert len(current.prepare)==prepares and len(parent.prepare)==0,(case,b,len(current.prepare))
                assert sum("scheduler_metadata" in c for c in current.fa)==(62 if prepares else 0)
                if prepares:
                    for increment in (1,5,8):
                        md.cache_seqlens_int32.add_(increment)
                        for node in current.nodes:
                            node()
                        num_replays+=1
                if case=="standard":
                    full,swa=expected_splits[b]
                    assert min(full,swa)>1
                    shape_rows.append(dict(users=b,splits_full=full,splits_swa=swa,full_layers=16,swa_layers=46,
                                           recorded_builders=prepares,potential_removed_prepares=62-prepares))
    return dict(parent_tree=PARENT,page_size=1,physical_capacity=131072,head_geometry=[16,2,64],
                scenarios=24,compared_native_calls=24*62,expected_scope_restorations=restored,
                changed_length_replays=num_replays,shapes=shape_rows,
                source_sha256={name:hashlib.sha256((stage/name).read_bytes()).hexdigest()
                               for name in (fixture["BACKEND"],fixture["MUSE"])},
                measurement="Real captured-program Python entry points; mocked CUDA and native FA calls, pinned C++ split receipts; no GPU timing.")


if __name__=="__main__":
    print(json.dumps(run(Path(sys.argv[1])),indent=2))
