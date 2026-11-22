"""Source-bound, CPU-only ordering checks for W13 MC4 and W2 MC2/4.

Usage: python independent_fence.py HUMMING_INCLUDE W13_MC4.cu W2_MC2.cu W2_MC4.cu
Use the .cu files emitted by the respective archived production-constructor renders.
"""
import hashlib
import json
import re
import sys
from graphlib import TopologicalSorter
from pathlib import Path


def in_order(source, *tokens):
    pos = 0
    for token in tokens:
        pos = source.index(token, pos) + len(token)


def audit_image(image, factor, gate):
    source = image.read_text()
    expected = dict(kUseTmaA="true", kUseTmaC="false", kUseStreamK="false",
                    kReduceOverlapLastStageOnly="false", kMultiCastSizeA=f"{factor}u",
                    kMultiCastSizeB="1u", kNumMathThreads="256u", kNumStages="4u")
    for key, value in expected.items():
        assert f"static constexpr auto {key} = {value};" in source, key
    in_order(source, "void wait_math_epilogue()", "mbarrier_wait(", "ctx.sync_load_threads();",
             "tma_fence_async_shared();", "if (ctx.cluster_rank() && ctx.load_thread_id() == 0)",
             "__cluster_map_shared_rank(&ctx.smem.math_mbar[kNumStages], 0)", "mbarrier_arrive<true>(ready)",
             "phases[kNumStages] ^= 1;")
    assert "kNumMathThreads * factor + (cluster_rank == 0 && thread_id == kNumStages ? kMultiCastSize - 1 : 0)" in source
    assert "cluster_rank == 0 && thread_id < kNumStages" in source
    assert "ctx.cluster_rank() >= 1 && stage_id < kNumStages" in source
    in_order(source, "if constexpr (!Ctx::kIsIndexedGemm) producer.wait_math_epilogue();",
             "producer.load_stage<true, true>(0);", "producer.wait_stage(stage_id);",
             "producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);",
             "consumer.arrive(kNumStages);", "while (scheduler.get_next_block())", "epilogue.call(",
             "if constexpr (!kReduceOverlapLastStageOnly) consumer.arrive(kNumStages);",
             "__syncthreads();", "barrier.cluster.arrive;", "barrier.cluster.wait;")
    if gate:
        in_order(source, "muse_residual_mma<Op, false, Ctx>", "if (warp_iter_id == Ctx::kWarpIters - 2)", "consumer.arrive(stage_id);")
        in_order(source, "iter * 2 * sizeof(int4)", "Op::fma(", "if (!Paired || iter % 2)", "wgmma_wait<0>();")
        in_order(source, "return ctx.smem.reduce[", "auto gate = fragment(s, col);",
                 "auto up = fragment(s + 2 * MUSE_GATE_M, col);", "output[static_cast<uint64_t>(start + row)")
    else:
        assert "if (warp_iter_id + 1 == Ctx::kWarpIters) consumer.arrive(stage_id);" in source
    return dict(image=image.name, sha256=hashlib.sha256(image.read_bytes()).hexdigest(), cluster=factor)


def closure(deps):
    before = {}
    for n in TopologicalSorter(deps).static_order():
        parents = deps.get(n, set())
        before[n] = set(parents).union(*(before[p] for p in parents))
    return before


def jobs(cluster, count, mode):
    # Nodes denote completion, including all participant math lanes as a group.
    # The initial seed is release(-1); the very last release has no subsequent job.
    deps = {}
    for r in range(cluster):
        deps['end-math', -1, r] = set()
    for j in range(count):
        for r in range(cluster):
            local = ('end-math', j-1, r)
            deps['fence', j, r] = {local}
            if r == 0:
                if mode == "producer":
                    deps['fence', j, r].update(('fence', j, s) for s in range(1, cluster))
                if mode == "math":
                    deps['fence', j, r].update(('end-math', j-1, s) for s in range(1, cluster))
            deps['multicast', j] = {('fence', j, 0)}
            deps['last-read', j, r] = {('multicast', j), ('fence', j, r)}
            deps['end-math', j, r] = {('last-read', j, r)}
    for r in range(cluster):
        deps['exit', r] = {('end-math', count-1, s) for s in range(cluster)}
    before = closure(deps)
    hazards = [(j,r) for j in range(1,count) for r in range(1,cluster)
               if ('last-read', j-1, r) not in before['multicast', j]]
    unfenced = [(j,r) for j in range(count) for r in range(1,cluster)
                if ('fence', j, r) not in before['multicast', j]]
    # Simulate the exact root wait parity, expected total, and startup/final epochs.
    # A nonempty cluster gets one remote ACK per follower per job start, never on exit.
    epochs = []
    parity = 0
    for j in range(count):
        assert parity == j % 2
        epochs.append((parity, 256 + cluster - 1 if mode == "producer" else 256 * (cluster if mode == "math" else 1)))
        parity ^= 1
    return dict(jobs=count, stale_read_orders=len(hazards), missing_peer_fence_orders=len(unfenced),
                root_waits=epochs, remote_producer_arrivals_per_follower=count if mode=="producer" else 0)


def ring(stages, cluster, slices, late):
    # A stage's refill waits every rank's designated release; with >=3 stages the
    # released physical stage is one later than the destination of the refill.
    deps={}
    for slice_id in range(slices):
        load=('load',slice_id)
        parents=set()
        if slice_id >= stages:
            waited = slice_id-stages + int(stages>2)
            parents.update(('k16',waited,r,3 if late else 2) for r in range(cluster))
        deps[load]=parents
        for r in range(cluster):
            for k in range(4):
                last = [('k16',slice_id,r,k-1)] if k else ([('k16',slice_id-1,r,3)] if slice_id else [])
                deps['k16',slice_id,r,k]=set(last+[load])
    before=closure(deps)
    return sum(('k16',i-stages,r,3) not in before['load',i]
               for i in range(stages,slices) for r in range(cluster))


def main():
    header,w13,w22,w24=map(Path,sys.argv[1:])
    ep=(header/'epilogue/pipeline.cuh').read_text()
    in_order(ep,'void call(', 'smem_writer.write(', 'ctx.sync_math_threads();', 'gmem_writer.write(', 'ctx.sync_math_threads();')
    sm=(header/'utils/storage.cuh').read_text()
    in_order(sm,'union alignas(1024)', 'stages[kNumStages];', 'IF_REDUCE_LAST_STAGE_ONLY(StageStorage reduce_skip', 'int4 reduce[')
    full=(header/'mma/wgmma.cuh').read_text()
    in_order(full,'void run(uint32_t stage_id, uint32_t iter_id)', 'offsetof(SharedStorage, stages)', 'k_slab * 2 * sizeof(int4)', 'make_wgmma_smem_desc', 'MmaOpClass::fma(', 'wgmma_commit();','wgmma_wait<0>();')
    native=(header/'scheduler.cuh').read_text()
    in_order(native,'dp_mn_iters = mn_blocks / kNumCtaGroups;', 'blockIdx.x / kMultiCastSize < mn_blocks % kNumCtaGroups', 'dp_mn_next_index = blockIdx.x / kMultiCastSize;')
    images=[audit_image(w13,4,True),audit_image(w22,2,False),audit_image(w24,4,False)]
    by_factor={}
    for f in (2,4):
        by_factor[f]={mode:[jobs(f,j,mode) for j in (0,1,2,5)] for mode in ('local','math','producer')}
        for a in by_factor[f]['producer']:
            assert a['stale_read_orders']==a['missing_peer_fence_orders']==0
        assert by_factor[f]['local'][-1]['stale_read_orders'] and by_factor[f]['math'][-1]['missing_peer_fence_orders']
    rings={k:ring(*config) for k,config in {
        'w13_four_stages':(4,4,64,False),'w13_two_stages_negative_control':(2,4,64,False),
        'w2_four_stages':(4,4,64,True)}.items()}
    assert rings==dict(w13_four_stages=0,w13_two_stages_negative_control=248,w2_four_stages=0)
    files={}
    for label, image, names in (
        ("w13",w13,["prompt_multicast.py","decode_gate_up.py","residual_prompt_mma.py"]),
        ("w2",w24,["prompt_weight_pipeline.py","residual_prompt_mma.py"])):
        package=image.parent.parent/"sglang/python/sglang/srt/layers/muse_spark"
        for name in names:
            path=package/name
            files[label+"/"+name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("epilogue/pipeline.cuh","utils/storage.cuh","mma/wgmma.cuh","scheduler.cuh",
                 "kernel/humming_ws.cuh","memory/g2s_pipeline.cuh"):
        files["vendor/"+name]=hashlib.sha256((header/name).read_bytes()).hexdigest()
    print(json.dumps(dict(images=images,source_sha256=files,ordering=by_factor,unsafe_in_job_pairs=rings,
                          scope="CPU happens-before graph, native generated geometry and source checks; no CUDA scheduling or numeric execution"),indent=2))
if __name__=="__main__":
    main()
