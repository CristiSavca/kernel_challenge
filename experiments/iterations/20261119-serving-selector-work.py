"""Bound a lazy greedy DFLASH selector and dense draft activation fusion for V296, using released shapes."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/"serving_v0523/candidates/v296_normalized_peer_routes"
V=json.loads((C/"validation.json").read_bytes())
PATCH=(C/"patch.diff").read_bytes()
assert sha256(PATCH).hexdigest()==V["patch_sha256"]


def source(name):
    path="python/sglang/srt/"+name
    base=subprocess.check_output(["git","-C",ROOT/"serving_v0523/sglang","show",V["base"]+":"+path]).decode().splitlines(keepends=True)
    hunks=next(x for x in PATCH.decode().split("diff --git ") if x.startswith(f"a/{path} b/{path}\n"))
    result=[]
    pos=0
    for line in hunks.splitlines(keepends=True):
        if line.startswith("@@ "):
            m=re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@",line)
            start=int(m[1])-1
            assert start>=pos
            result.extend(base[pos:start]);pos=start
        elif line.startswith(("+++","---","index ","a/")):
            continue
        elif line[0] in " +-":
            if line[0]!="+":
                assert base[pos]==line[1:]
                pos+=1
            if line[0]!="-":
                result.append(line[1:])
    return "".join(result+base[pos:])


def defined(src, name, env):
    n=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name==name)
    n.decorator_list=[]
    exec("from __future__ import annotations\n"+ast.unparse(n),env)
    return env[name]


def synthetic(src):
    ctx=dict(torch=torch,F=F)
    edges=defined(src,"_score_edges",ctx)
    defined(src,"_follow_maps",ctx)
    walk=defined(src,"sample_path",ctx)
    g=torch.Generator().manual_seed(8192)
    compared=queries=0
    for users in (1,8,32):
        for trial in range(16):
            gamma,k,r,vocab=7,16,256,521
            a,b,h=[torch.randn(shape,generator=g).bfloat16() for shape in
                ((vocab,r),(vocab,r),(users,gamma,r))]
            unary=torch.randn((users,gamma,k),generator=g)
            ids=torch.randint(vocab,(users,gamma,k),generator=g)
            anchor=torch.randint(vocab,(users,),generator=g)
            if trial%4==1: unary.zero_(); a.zero_()  # exact tie, including repeated candidates
            if trial%4==2: ids[:,:,1::2]=ids[:,:,::2]
            if trial%4==3: h[:,::2].zero_()
            dense=edges(predecessor_table=a,successor_table=b,candidate_ids=ids,unary_logits=unary,
                        hidden=h,anchor_token_ids=anchor,top_k=k)
            base,probs=walk(SimpleNamespace(top_k=k),candidate_ids=ids,scores=dense,
                uniforms=torch.rand((users,gamma),generator=g),temperatures=torch.ones(users),
                greedy_mask=torch.ones(users,dtype=torch.bool))
            current=anchor; idx=torch.zeros(users,dtype=torch.int64); batches=torch.arange(users)
            for t in range(gamma):
                dots=torch.bmm((a[current]*h[:,t]).unsqueeze(1),b[ids[:,t]].transpose(1,2))[:,0]
                scores=dots+unary[:,t]
                assert torch.equal(scores.view(torch.int32),dense[batches,t,idx].contiguous().view(torch.int32))
                idx=scores.argmax(dim=-1);current=ids[batches,t,idx]
                assert torch.equal(current,base[:,t]) and torch.equal(F.one_hot(idx,k).float(),probs[:,t])
                compared+=scores.numel();queries+=users
    return dict(cases=3*16,fp32_score_words_checked=compared,proposal_ids_and_onehot_rows=queries)


def budget(users):
    slots,topk,rank,hidden,inner,local_vocab=7,16,256,8192,28672//8,202048//8
    preds=users*slots
    verify=users*8
    dense=2*preds*topk*topk*rank
    lazy=2*preds*topk*rank
    head=2*preds*hidden*local_vocab
    return dict(users=users,prediction_rows=preds,verify_rows=verify,native_edge_dot_flops=dense,
        prospective_edge_dot_flops=lazy,eliminated_flops=dense-lazy,
        all_position_hidden_projection_flops=2*preds*hidden*rank,local_vocabulary_projection_flops=head,
        eliminated_vs_unavoidable_local_vocab_percent=100*(dense-lazy)/head,
        baseline_predecessor_logical_bf16_bytes=preds*topk*rank*2,
        prospective_predecessor_logical_bf16_bytes=preds*rank*2,
        baseline_predecessor_bf16_distinct_row_upper_per_edge_summed=(1+(slots-1)*topk)*users*rank*2,
        successor_bf16_bytes_both=preds*topk*rank*2,full_score_output_fp32_bytes=preds*topk*topk*4,
        five_dense_draft_mlp_flops=5*2*verify*hidden*(3*inner),
        five_dense_draft_mlp_weight_bytes=5*hidden*3*inner*2,
        draft_mlp_intermediate_read_plus_write_upper_bytes=5*verify*3*inner*2*2)


model,worker=source("models/dflash.py"),source("speculative/dflash_worker_v2.py")
assert 'hidden = self.hidden_projection(hidden_states)' in model and '"blpr,blcr->blpc"' in model
assert 'self.selector.sample_path(' in worker and 'candidate_ids, scores = _selector_lattice(' in worker
assert 'self.gate_up_proj(x)' in model and 'self.down_proj(x)' in model and 'self.act_fn = SiluAndMul()' in model
trace=next((ROOT/"experiments/modal_node/20261113T204343Z-bench").glob("*v41__server.log"))
official=ROOT/"serving_v0523/runs/abd217baf0ed49fd.log"
assert "DFLASH selector decode folded into the draft cuda graph (sampling_enabled=True)" in official.read_text()
live={n:[s for s in trace.read_text().splitlines() if re.search(r"Decode batch, #running-req: "+str(n)+",",s)]
      for n in (8,32)}
torch.set_num_threads(1)
print(json.dumps(dict(tree=V["tree"],patch_sha256=V["patch_sha256"],
    draft_model_sha256=sha256(model.encode()).hexdigest(),worker_sha256=sha256(worker.encode()).hexdigest(),
    shape_source="https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/blob/46c90f412549c472941a68382d91a99e5e6372a0/config.json",
    official_capture_log_sha256=sha256(official.read_bytes()).hexdigest(),v41_log_sha256=sha256(trace.read_bytes()).hexdigest(),
    observed_v41_decode_lines=live,torch_cpu=str(torch.__version__),greedy_host_comparison=synthetic(model),
    budgets=[budget(n) for n in (8,32)],physical_gpu=False,
    limit="Tested host reductions and first-index ties. Launch latency and HBM reads cannot be derived from these work counts. No timing at eight users is available in these traces."
),indent=2))
