"""Prove the ordering required by a remote reader for mixed published, legacy and declined epochs."""
import ast
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import urllib.request

stage=Path(sys.argv[1])
source=stage/"python/sglang/srt/layers/muse_spark/peer_prompt_pack.py"
code=source.read_text()
root=ast.parse(code)
def method(name):
    return ast.unparse(next(n for n in ast.walk(root) if isinstance(n,ast.FunctionDef) and n.name==name))
def in_order(text,parts):
    pos=0
    for part in parts:
        pos=text.index(part,pos)+len(part)
in_order(method("_try_published_routes"),["with workspace.lock","caller.wait_stream(workspace.stream)",
    "bank = workspace.next_bank","local = packets","if eligible","torch.cat(","local.zero_()",
    "workspace.handle.barrier(channel=0)","received = torch.stack(packets)","decision = ","if accepted",
    "_grouped_humming_and_combine_impl","workspace.next_bank ^= 1","workspace.stream.wait_stream(caller)"])
in_order(method("_published_experts_into"),["_try_published_routes","sp_exit_gather(scores","sp_exit_gather(ids","_peer_experts_into"])
legacy=next(ast.unparse(n) for n in ast.walk(root) if isinstance(n,ast.FunctionDef) and n.name=="pack" and "with self.lock" in ast.unparse(n))
in_order(legacy,["with self.lock","self.stream.wait_stream(caller)","with torch.cuda.stream(self.stream)","storage[:shard.shape[0]].copy_",
    "self.handle.barrier","pack_peer_rows(","self.next_bank ^= 1","caller.wait_stream(self.stream)"])

rev="5c4886908584029761b579af026dcfb627c84070"
paths={"buffer":"torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp",
    "dispatch":"torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu",
    "barrier":"torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh",
    "stack":"aten/src/ATen/native/TensorShape.cpp","copy":"aten/src/ATen/native/cuda/Shape.cu"}
localnames={"buffer":"sym.cpp","dispatch":"cuda.cu","barrier":"barrier.cuh","stack":"tensors.cpp","copy":"shape.cu"}
upstream={}
for k,path in paths.items():
    tmp=Path(sys.argv[2])/("upstream-"+localnames[k]) if len(sys.argv)>2 else None
    data=tmp.read_bytes() if tmp else urllib.request.urlopen(f"https://raw.githubusercontent.com/pytorch/pytorch/{rev}/{path}").read()
    text=data.decode()
    if k=="buffer":
        in_order(text[text.index("auto data_ptr = reinterpret_cast<uint8_t*>(peer_ptr)"):],["handle->get_device()","at::for_blob(data_ptr, sizes)",".target_device(device)"])
    elif k=="dispatch":
        in_order(text[text.index("void CUDASymmetricMemory::barrier("):],["barrier_kernel<<<","at::cuda::getCurrentCUDAStream()"])
    elif k=="barrier":
        in_order(text[text.index("void barrier_kernel("):],["try_put_signal<std::memory_order_release>","try_wait_signal<std::memory_order_acquire>"])
    elif k=="stack":
        in_order(text[text.index("Tensor stack(TensorList"):],["at::cat(tensors, wrapped_dim)","out.view"])
    else:
        assert "slice_size % alignment == 0" in text and "alignment = 16" in text
        assert "CatArrayBatchedCopy_vectorized" in text
    upstream[k]={"path":path,"sha256":sha256(data).hexdigest()}
def ordering(world, scenarios, omit=None):
    tail={}
    ancestry={}
    def event(stream, extra=()):
        name=len(ancestry)
        parents=[e for e in (tail.get(stream),*extra) if e is not None]
        ancestry[name]=set(parents).union(*(ancestry[e] for e in parents))
        tail[stream]=name
        return name
    writes=defaultdict(list); reads=[]
    epochs=[kind for scenario in scenarios for kind in {
        "ok":["ok"],"empty":["empty"],"veto_dense":["veto"],"veto_grouped":["veto","old"],
        "outside":["old"],"capture":[]}[scenario]]
    for i,kind in enumerate(epochs):
        bank=i%2 if omit!="single" else 0
        callers=[(r, "caller"+str((i+r)%3)) for r in range(world)]
        dedicated=[(r,"peer") for r in range(world)]
        workers=dedicated if kind=="old" else callers
        arrived=[]
        for r in range(world):
            dest,other=(dedicated[r],callers[r]) if kind=="old" else (callers[r],dedicated[r])
            event(dest,[] if omit=="entry" else [tail.get(other)])
            if kind!="veto" or r!=world-1:
                writes[r,bank,"activation"].append((i,event(dest)))
            if kind!="old":
                writes[r,bank,"packet"].append((i,event(dest)))
            arrived.append(event(dest))
        for r in range(world):
            stream=workers[r]
            event(stream,arrived if omit!="barrier" else ())
            if kind!="old":
                reads.append((i,r,bank,"packet",event(stream)))
            if kind in ("old","ok","empty") and not (kind=="empty" and r%2==0):
                reads.append((i,r,bank,"activation",event(stream)))
            dest=callers[r] if kind=="old" else dedicated[r]
            event(dest,[] if omit=="exit" else [tail[stream]])
    checks=failures=0
    for i,rank,bank,plane,op in reads:
        for producer in range(world):
            hist=writes[producer,bank,plane]
            current=next(w for e,w in hist if e==i)
            checks+=1
            failures+=current not in ancestry[op]
            for e,w in hist:
                if e>i:
                    checks+=1
                    failures+=op not in ancestry[w]
    return checks,failures
cases=[("ok","ok","ok","ok","ok"),("outside","empty","capture","ok","veto_grouped","ok"),
       ("veto_dense","veto_grouped","empty","outside","ok","ok"),
       ("ok","veto_grouped","outside","capture","veto_dense","empty","ok"),
       ("capture","ok","outside","outside","veto_grouped","ok","outside")]
summary={}
for omission in (None,"entry","exit","barrier","single"):
    vals=[ordering(world,c,omission) for world in (2,4,8) for c in cases]
    checks,bad=map(sum,zip(*vals))
    assert (bad==0)==(omission is None),(omission,bad)
    summary[omission or "both_stream_edges_and_barrier"]=dict(invariants=checks,missing_order=bad)
print(json.dumps(dict(source_sha256=sha256(source.read_bytes()).hexdigest(),pinned_pytorch=rev,source=upstream,
    results=summary,scope="Transitive closure for all GPU schedules under the supplied edges. Three caller streams per rank, one peer stream, two reused banks, publication skips, fallback and veto epochs. Omits extra ordering from local host synchronization or NCCL; missing edges in negative controls are model witnesses."),indent=2))
