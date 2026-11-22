"""Follow the paired prompt projection through the native model tail and replay trimming."""
import ast
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from check_transport import fixture, projection, STAGE


def method(path, owner, name, env):
    tree=ast.parse((STAGE / "python/sglang/srt" / path).read_text())
    cls=next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name==owner)
    fn=next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name==name)
    exec("from __future__ import annotations\n"+ast.unparse(fn),env)
    return env[name]


class Output(NS):
    def __getattr__(self, name):
        return None


def worker(rank, size, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=size, init_method='file://'+str(Path(folder)/'init'),
                            timeout=timedelta(seconds=90))
    fixture.install("sglang.srt.batch_invariant_ops", {"is_batch_invariant_mode_enabled":lambda:False})
    comm=fixture.Comm(rank)
    group=NS(rank_in_group=rank,world_size=size,cpu_group=dist.group.WORLD,pynccl_comm=comm)
    def gather(x,num_tokens):
        with torch.inference_mode(False):
            parts=[torch.empty_like(x) for _ in range(size)]
        dist.all_gather(parts,x)
        return torch.cat(parts)[:num_tokens]
    class Logits(nn.Module):
        expected_aux_hidden_states=None
        keep_last_hidden_states=False
        def forward(self,ids,h,head,b,aux,sp_prompt_tokens=None):
            total=len(ids) if ids is not None else len(b.embeds)
            full=gather(h,total)
            return Output(hidden_states=aux, next_token_logits=full.index_select(0,b.endpoints),
                          dflash_prefill_context_done=False,dflash_prefill_kv_projected=False,mm_input_embeds=b.embeds)
    class Draft(nn.Module):
        quantizable=False
        def project_target_hidden(self,x):return self.hidden_norm(self.fc(x))
    torch.manual_seed(185)
    draft=Draft()
    draft.fc=nn.Linear(320,64,bias=False,dtype=torch.bfloat16)
    draft.hidden_norm=nn.RMSNorm(64,eps=1e-5,dtype=torch.bfloat16)
    model=nn.Module()
    model.capture_aux_hidden_states=True
    model.logits_processor=Logits()
    model.lm_head=None
    target=projection._PrefillContext(draft,draft.fc,draft.hidden_norm,draft.project_target_hidden,group,model,model.logits_processor)
    env=dict(get_parallel=lambda:NS(tp_group=group),get_forward=lambda:NS(set=lambda *args:None),
             layernorm_sp=NS(runs_sp=lambda mode:True,sp_exit_gather=gather),
             pack_aux_hidden_states=lambda xs:torch.cat(xs,-1),project_dflash_prefill_shard=projection.project_dflash_prefill_shard,
             project_paired_draft_kv=projection.project_paired_draft_kv)
    forward=method('models/muse_spark.py','MuseSparkTextForCausalLM','forward',env)
    trim=method('model_executor/runner/prefill_cuda_graph_runner.py','PrefillCudaGraphRunner','_trim_logits_output',{'LogitsProcessorOutput':Output})
    counts=Counter()
    weight=torch.randn(64,10,generator=torch.Generator().manual_seed(50+rank)).bfloat16()
    helper=NS(flat_kv_weight_t=weight,n_layers=5,layer_out_dim=2)
    with torch.inference_mode(),patch.object(projection,'_supports_shard_norm',return_value=True):
        for span,delta in ((64,0),(128,0),(130,3)):
            n=span*size-delta
            taps=torch.randn(size*span,320).bfloat16()
            h=taps[:,:64].contiguous()
            normalized=draft.project_target_hidden(taps)[:n]
            model.model=lambda *args,**kw:(h.chunk(size)[rank],taps.chunk(size)[rank].split(64,dim=1))
            for state in ('paired','no_helper','no_context','special'):
                b=NS(embeds=h[:n], endpoints=torch.tensor([1,n-1]),forward_mode='extend',capture_hidden_mode=NS(is_full=lambda:True),
                     multi_item_delimiter_indices=None,token_indices_to_pool=True if state=='special' else None)
                with projection.dflash_sp_prefill_scope(None if state=='no_context' else target, helper if state=='paired' else None):
                    for ids in (None,torch.arange(n)):
                        o=forward(model,ids,None,b,b.embeds)
                        use=n>=size*128 and state in ('paired','no_helper')
                        paired=use and state=='paired'
                        expected=(normalized@weight) if paired else (normalized if use else taps[:n])
                        assert o.dflash_prefill_context_done is use and o.dflash_prefill_kv_projected is paired
                        torch.testing.assert_close(o.hidden_states,expected,rtol=0.008,atol=0.015)
                        assert torch.equal(o.next_token_logits,h[b.endpoints])
                        for full_graph in (True,False):
                            runner=NS(raw_num_tokens=n,raw_bs=2,_is_full_backend=full_graph,
                                      model_runner=NS(spec_algorithm=NS(is_speculative=lambda:True)))
                            # The raw prompt count and output columns are independent.
                            padded=Output(**vars(o))
                            padded.hidden_states=torch.nn.functional.pad(o.hidden_states,(0,0,0,7),value=53)
                            result=trim(runner,padded)
                            assert result.dflash_prefill_kv_projected is paired
                            assert result.dflash_prefill_context_done is use
                            assert torch.equal(result.hidden_states,o.hidden_states)
                            assert torch.equal(result.next_token_logits,o.next_token_logits)
                            counts['trimmed_cases']+=1
                        counts['model_forwards']+=1
                        if paired:
                            counts['paired_payloads']+=1
                            counts['compared_projected_values']+=expected.numel()
    all_counts=[None]*size
    dist.all_gather_object(all_counts,dict(counts))
    if rank==0:
        summed=Counter()
        for c in all_counts:summed.update(c)
        (Path(folder)/'result.json').write_text(json.dumps(summed))
    dist.destroy_process_group()


if __name__=='__main__':
    data={}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker,args=(size,folder),nprocs=size)
            data[size]=json.loads((Path(folder)/'result.json').read_bytes())
    print(json.dumps({'TP':data,'physical_H200':False,'scope':'Production Muse model tail, projection, communication, trim; CPU head observer and RMS implementation.'},indent=2))
