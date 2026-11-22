"""Execute frozen parent and candidate JIT bodies and compare every guarded output with an independent stable reference."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl

from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf
from compiler_check import stage_warps


def run_case(module, tokens, topk, experts, width, mode, dtype, rng):
    slots = tokens*topk
    span = triton.next_power_of_2(slots)
    mapped = mode not in ("no_map","no_map_empty","no_map_hot")
    global_experts = experts*8 if mapped else experts
    ids = rng.integers(-2,global_experts+3,slots,dtype=dtype)
    model_map = np.full(global_experts,-1,dtype="i4")
    if mapped:
        if mode.startswith("native"):
            model_map[3*experts:4*experts] = np.arange(experts,dtype="i4")
            if mode=="native":
                ids=np.array([rng.choice(global_experts,topk,replace=False)
                             for _ in range(tokens)],dtype=dtype).ravel()
        else:
            model_map[1::8]=np.arange(experts,dtype="i4")[::-1]
            if mode=="map_overflow":
                model_map[1:20:2]=np.arange(1,20,2,dtype="i4")+experts
                ids[::3]=1
            if mode=="repeat":
                ids[::2]=1
                ids[1::4]=min(global_experts-1,9)
        if mode=="all_empty": ids.fill(-1)
    elif mode=="no_map_empty": ids.fill(global_experts+1)
    elif mode=="no_map_hot":ids.fill(experts-1)
    if dtype==np.int64 and mode not in ("native","all_empty","no_map_empty","no_map_hot"):
        ids[0]=(1<<40)+1
    local=np.full(slots,-1,dtype="i4")
    valid=(ids>=0)&(ids<global_experts)
    local[valid]=model_map[ids[valid]] if mapped else ids[valid].astype("i4")
    bucket=np.zeros(span,dtype="i4")
    bucket[:slots]=np.where((local>=0)&(local<experts),local+1,0)
    key=np.sort(bucket*span+np.arange(span,dtype="i4"),kind="stable")
    padding=span-slots
    expected_keys=key[padding:]
    expected_offsets=np.bincount(bucket,minlength=experts+1).cumsum().astype("i4")-padding
    expected_rows=np.full(slots,-1,dtype="i4")
    live_keys=expected_keys[expected_offsets[0]:]
    expected_rows[live_keys%span]=np.arange(expected_offsets[0],slots,dtype="i4")
    input_bits=rng.integers(0,65536,(tokens,width),dtype="u2")
    # Include signed zero, infinities, subnormals, and noncanonical NaN payloads.
    input_bits[:,:8]=np.array([0,0x8000,1,0x7f80,0xff80,0x7fa1,0xffc9,0x3f80],dtype="u2")
    expected_stage=np.full((slots,width),0xacc7,dtype="u2")
    expected_stage[expected_offsets[0]:]=input_bits[(live_keys%span)//topk]
    dtype_tl=tl.int64 if dtype==np.int64 else tl.int32
    originals=ids.copy(),model_map.copy(),input_bits.copy()
    common=(buf(ids,dtype_tl),buf(model_map,tl.int32) if mapped else buf(ids,dtype_tl))
    goal=(local,expected_offsets,expected_keys,expected_rows,expected_stage.ravel())
    for which in ("parent","candidate"):
        allocations=[]
        for reference in goal:
            entry=np.full(reference.size+4,0xacc7 if reference.dtype==np.uint16 else -27109,dtype=reference.dtype)
            allocations.append(entry)
        outputs=[a[2:-2] for a in allocations]
        tlout=[buf(a,tl.bfloat16 if a.dtype==np.uint16 else tl.int32) for a in outputs]
        positional=(experts,global_experts,mapped,span,triton.next_power_of_2(experts+1))
        width_span=triton.next_power_of_2(width)
        if which=="parent":
            module._plan_grouped_decode[(1,)](*common,*tlout[:4],slots,*positional,num_warps=4)
            module._gather_grouped_decode[(min(slots,128),)](buf(input_bits.ravel(),tl.bfloat16),
                tlout[2],tlout[1],tlout[4],slots,topk,width,span,width_span,num_warps=8 if width>=4096 else 4)
        else:
            module._stage_grouped_decode[(slots+1,)](*common,*tlout[:4],buf(input_bits.ravel(),tl.bfloat16),
                tlout[4],slots,*positional,topk,width,width_span,num_warps=stage_warps(source,slots,width,experts))
        for index,(actual,expected,guarded) in enumerate(zip(outputs,goal,allocations,strict=True)):
            np.testing.assert_array_equal(actual,expected,err_msg=f"{which}: tokens={tokens} {mode} output {index}")
            guard=0xacc7 if index==4 else -27109
            assert np.all(guarded[[0,1,-2,-1]]==guard)
    for actual,original in zip((ids,model_map,input_bits),originals,strict=True):
        np.testing.assert_array_equal(original,actual)
    return dict(tokens=tokens,topk=topk,slots=slots,experts=experts,hidden=width,map=mapped,warps=stage_warps(source,slots,width,experts),
                ids="i64" if dtype==np.int64 else "i32",distribution=mode,live_routes=len(live_keys),
                output_BF16_words=expected_stage.size,live_BF16_words=len(live_keys)*width,
                checked_guard_words_both_paths=40,checked_metadata_int32=3*slots+experts+1)


source=Path(sys.argv[1])
rng=np.random.default_rng(261)
cases=[
    (1,1,1,1024,"no_map_hot"), (1,8,32,4096,"native"), (2,8,32,4096,"no_map"),
    (3,7,7,1025,"map_overflow"), (4,8,32,4096,"native"), (7,8,17,3073,"repeat"),
    (8,8,32,4096,"native"), (16,8,32,4096,"native"), (31,8,32,4096,"native"),
    (32,8,32,4096,"native"), (16,16,64,2048,"all_empty"), (17,15,17,1024,"no_map_empty"),
    (32,8,256,4096,"no_map_hot"), (2,2,1,4095,"repeat"), (1,3,3,2049,"map_overflow")]
with tempfile.TemporaryDirectory() as temporary:
    module=load_jit_module(source,Path(temporary))
    results=[run_case(module,*case,dtype,rng) for case in cases for dtype in (np.int32,np.int64)]
print(json.dumps(dict(triton=triton.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    physical_gpu=False,parent_and_candidate_vs_independent_reference=True,cases=results,
    metadata_int32_words=sum(r["checked_metadata_int32"] for r in results),
    BF16_output_words=sum(r["output_BF16_words"] for r in results),
    BF16_live_words=sum(r["live_BF16_words"] for r in results),
    checked_parent_and_candidate_guard_words=sum(r["checked_guard_words_both_paths"] for r in results)),indent=2))
