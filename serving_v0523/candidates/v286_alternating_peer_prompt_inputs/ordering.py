"""Check all topological orderings of the publication dependencies via transitive closure."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

stage=Path(sys.argv[1])
base="python/sglang/srt/"
workspace=base+"layers/muse_spark/peer_prompt_pack.py"
humming=base+"layers/muse_spark/muse_spark_humming_moe.py"
model=base+"models/muse_spark.py"
sp=base+"layers/layernorm_sp.py"
graph=base+"model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py"

def src(path):
    if (stage/path).is_file():return (stage/path).read_text()
    return subprocess.check_output(["git","-C",stage,"show","HEAD:"+path],text=True)
def method(path, name, owner=None):
    top=ast.parse(src(path))
    body=top.body if not owner else next(n.body for n in top.body if getattr(n,"name",None)==owner)
    return ast.unparse(next(n for n in body if getattr(n,"name",None)==name))
def ordered(text,parts):
    off=-1
    for part in parts:off=text.index(part,off+1)

pack=method(workspace,"pack","PeerPromptInputs")
ordered(pack,["with self.lock","self.stream.wait_stream(caller)","with torch.cuda.stream(self.stream)",
              "self.banks[self.next_bank]",".copy_(shard)","self.handle.barrier(channel=0)",
              "if result is not None","pack_peer_rows(addresses","if len(self.banks) == 1",
              "self.handle.barrier(channel=0)","self.next_bank ^= 1","shard.record_stream","caller.wait_stream(self.stream)"])
publication=method(workspace,"_peer_experts_into")
if "_gather_prompt_routes" in publication:
    election = "_gather_prompt_routes"
else:
    election = "group.all_gather_into_tensor(votes, vote)"
ordered(publication,[election,"decision = ","if all(decision[:-1])","_grouped_humming_and_combine_impl","else:"])
dispatch=method(humming,"humming_grouped_expert_outputs")
assert dispatch.count("workspace.pack(")==2
ordered(dispatch,["if num_routes == 0","workspace.pack(hidden_states, None, None)","return ","workspace.pack(hidden_states, mapped_rows, inputs)","run_gate_up("])
outer=method(model,"_forward_sp_replicated_experts","MuseSpark1xMoE")
ordered(outer,["not torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph()","peer_experts_into(workspace","reduce_scatter_token_rows(output"])
assert "full = group.all_reduce(full)" in method(sp,"reduce_scatter_token_rows") and "return group.reduce_scatter_along_dim(full, dim=0)" in method(sp,"reduce_scatter_token_rows")
assert "capture_stub=_capture_stub" in src(workspace)
assert "self._break_fns[i]()" in src(graph) and "capture_stub(*args, **kwargs)" in src(graph)

def lineage(world, epochs, banks, retire):
    ancestors={}; ids={}
    def node(name, before):
        mask=0
        for parent in before:mask |= ancestors[parent] | (1<<ids[parent])
        ids[name]=len(ids);ancestors[name]=mask
        return name
    for e in range(epochs):
        for r in range(world):
            deps=[] if e==0 else ([(e-1,"retire")] if retire else [(e-1,r,"read")])
            node((e,r,"write"),deps)
        pub=node((e,"ready"),[(e,r,"write") for r in range(world)])
        for r in range(world):node((e,r,"read"),[pub])
        if retire:node((e,"retire"),[(e,r,"read") for r in range(world)])
    missing=[];checks=0
    for e in range(epochs):
        for dst in range(world):
            read=(e,dst,"read")
            for source in range(world):
                assert ancestors[read] & (1<<ids[(e,source,"write")])
                if e+banks<epochs:
                    checks+=1
                    if not ancestors[(e+banks,source,"write")] & (1<<ids[read]):
                        missing.append((e,dst,source))
    return checks,missing

result=[]
for world in (2,4,8):
    for epochs in (3,4,17,63):
        alt, bad=lineage(world,epochs,2,False); assert not bad
        old, bad=lineage(world,epochs,1,True); assert not bad
        _,lost=lineage(world,epochs,1,False); assert lost
        result.append(dict(world=world,epochs=epochs,old_reuse_pairs=old,alternating_reuse_pairs=alt,
                           missing_without_alternation=len(lost),example_missing=lost[0]))
print(json.dumps(dict(sources={p:hashlib.sha256(src(p).encode()).hexdigest() for p in (workspace,humming,model,sp,graph)},
    vote_method=election,closure=result,total_safe_variable_order_pairs=sum(r["old_reuse_pairs"]+r["alternating_reuse_pairs"] for r in result),
    scope="Edges capture per-rank single worker-stream FIFO and the common device publish barrier. All topological orders are included, with delayed or empty readers. Host fallback does not publish on any rank. Actual symmetric-memory GPU execution is unmeasured."),indent=2))
