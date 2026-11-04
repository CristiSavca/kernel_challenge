# Fixed three-stage bulk L2 weight prefetch feasibility

2026-11-04. Read-only study on exact standalone KPERM parent
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
SC6 is a separate closed attempt and is not a parent or part of this proposal.
No candidate, GPU allocation or official call was made for this study.

Verdict: structurally feasible and sufficiently small/distinct for a reviewed
fixed experiment. This is a feasibility finding, not implementation permission
or evidence of a speedup. Independent critic and root acceptance remain required.

## Mechanism and primary-source support

The parent's `wgs_wproducer` (lines2807–2832) issues real bulk TMA weight copies
through three shared slots. A slot becomes reusable only after all consumers
release it. Its existing L2 cache hint applies to that real demand copy; there
is no bulk prefetch of later weight stages. The proposed hint requests the exact
weight stage three positions later in the current phase's producer sequence,
immediately after issuing the current real copy. It may start fetching that
future stage before a shared slot is available. Shared capacity, real copy
order, transaction counts, release waits, activation producer and compute stay
unchanged. It does not eliminate any required copy or weight read.

NVIDIA documents `cp.async.bulk.prefetch.L2.global [addr], size` as a nonblocking
hint that may initiate a weak memory operation. Address and byte count must be
16-byte aligned; the instruction dates to PTX8.0 and requires sm90 or newer.
Use the plain form, without newer eviction-priority qualifiers or a completion
mechanism. Correctness never depends on the hint completing or being honored;
the original real TMA and mbarrier still supply all required data. [NVIDIA PTX
bulk-prefetch documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch)

Scoped search of `tools/RESEARCH_LOOP.md`, WGS/MoE notes, dated iteration records,
MoE candidates and saved MoE sources found no prior bulk weight-prefetch attempt.
Parent scalar `prefetch.global.L2` uses concern skinny activation rows or initial
activation warming. Those are separate paths; this search does not claim no
such experiment has ever existed outside the saved repository evidence.

## Exact future address and sequence boundaries

Keep the current loop and its `gs == g1` wrap unchanged. Immediately after its
existing real TMA, before advancing `ws`, the entire added block would be:

```cpp
if (i + 3 < g1 - g0) {
  int pgs = gs + 3;
  if (pgs >= g1) pgs = g0 + (pgs - g1);
  const int pu = pgs >> 4, pst = pgs & 15;
  const int pj = pu >> ubl, pub = pu & ((1 << ubl) - 1);
  const unsigned char* psrc = base
      + (size_t)(((S.jE[pj] << ubl) + pub) * 16 + pst)
        * (size_t)UNIT_BYTES;
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
               :: "l"(psrc), "r"(UNIT_BYTES) : "memory");
}
```

This operates only in the existing lane0 weight producer after `WS_JOIN`, where
routing tables and phase ranges are published. `S.jE` is immutable for the call;
W13/W2 packed weights are immutable prepared data. The phase selects `base` and
`ubl` exactly as before: A has32 units/job, B16, each with16 stages.

Let N=g1-g0. The actual order is `[sp,g1)` then `[g0,sp)`. For any i with i+3<N,
the expression above equals the actual stage at i+3. N must exceed3, so at most
one wrap is needed. The last three sequence items issue no new hints. Empty and
N<=3 ranges issue none. No hint crosses into the next phase or a job outside the
CTA's original phase range; moving into the next job within that range is valid
and uses its own `S.jE[pj]`. The first three items are not newly prefetched.

Each future target is requested once per CTA phase sequence. Different jobs
may name the same expert, and split ranges/other CTAs can request nearby or
duplicate cache lines; no uniqueness claim about physical DRAM traffic follows.
Stage36864 is16/128-byte aligned. With expertID0..31, unit and stage fields stay
within the exact parent packed bank. There is no additional shared write,
mbarrier arrival/expected byte, wait, commit group, or activation dependency.

Bounded CPU arithmetic checked both phases, g0=0..63 and N=0..128:
16512 ranges,1008000 future-target matches,19904 wrap targets,128 empty ranges,
512 ranges with zero hints. Every hinted stage equaled `sequence[i+3]`, and its
bank-relative address was checked at expert endpoints0/31. This is arithmetic
feasibility, not actual emitted candidate/native-source proof. A later verifier
should extract the actual inserted source and bind it to the unchanged real-TMA
trace, including large job boundaries, ragged ranges and the A→B phase boundary.

## Cost and evidence limits

For phase lengths N[c,p], extra bulk requests are exactly
`P = sum(max(N[c,p]-3,0))`. There are S=768J real weight stages and S activation
copies for J complete routed jobs. At132 CTAs, `max(0,S-792) <= P <= S`; the
upper bound is conservative. Thus this adds nearly50% as many bulk requests as
the existing combined weight+activation streams, with almost one hint per real
weight stage. It adds a guarded address calculation and shared job-expert load
to lane0. Actual generated instruction count and register allocation are unknown.

| Illustrative J | Real weight stages | Extra hint lower bound | Hint payload lower–upper, MiB |
| ---: | ---: | ---: | ---: |
|16|12288|11496|404.156–432|
|24|18432|17640|620.156–648|
|29|22272|21480|755.156–783|

These are requested hint-byte ranges, **not extra DRAM bytes**. A successful hint
could pull a line that the later demand hits; it could also be ignored, arrive
too late, compete with demand copies, or be evicted and force another DRAM fetch.
Three extra future stages correspond to108KiB of nominal lookahead per CTA,
13.922MiB across132 CTAs. This is not a measured L2 footprint, guaranteed number
of in-flight stages, or a guaranteed six-stage hardware pipeline.

Saved ef7 NCU evidence has DRAM reads near1.007–1.009 times nominal packed weight
bytes, and roughly75% DRAM utilization. Those observations neither measure
available latency hiding nor prove that this hint can help. Required weight
bytes, activation work and MMA/decode arithmetic remain the same. Additional
cache pressure, request-port contention or compiler changes can make it slower.
No percentage speed forecast is supported.

The affected standalone cases remain MAIN M32/128/256, with M1/8/16 controls.
The change can be confined to one producer function plus a unique build namespace
and bounded resource evidence if root requires it. Keep576 threads,231168 shared
bytes, all parent scale representations, GS2/wait1, costs, fixups and dispatch.
Native source checking should reject a missing tail guard, wrong wrap, expertID
reuse across a job boundary, or any changed real-copy/barrier sequence. CUDA
assembly acceptance and valid actual BIG0/BIG1 launch resources are prerequisites
to interpreting timing; hints cannot be proven effective by source or a PASS.

A single fixed experiment is justified by its distinct attempt to increase
memory lookahead without larger shared rings or decoded operand transport.
Root should predeclare a modest gain/control screen before implementation and
compare both exact247888 receipts. Failure of that fixed screen closes this
distance3 attempt; no automatic distance, cache-policy or combined-SC6 sweep.
