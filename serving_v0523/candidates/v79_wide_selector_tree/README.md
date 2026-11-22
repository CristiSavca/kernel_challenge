# V79 wider DFLASH verification

V79 targets **committed tokens per target pass**. Its first experiment gives the selector twelve target-verified proposal nodes, up from V76's eight. A high-confidence branch may extend deeper, or the proposer may spend the extra nodes on alternative continuations. The target model still chooses every committed token. Width is a launch setting shared across requests.

V76's target attention supports at most 64 query/head pairs per KV head in its tree. Twelve positions and eight grouped query heads need 96. V79 splits them into two independent bands. Each band traverses the original ordered KV partition with its own FP32 probability accumulator. The merge indexes each band's partial results with the same padded row stride used by the writer. Full attention, the sliding window and the tree ancestry mask apply to every active row.

## First H200 comparison

Use the [V76 control](../v76_sharded_draft_target_decode/README.md) at 16 users and the same six options. Add `--flag speculative-num-draft-tokens=12` for V79. The seven launch settings, file limits and frozen SHA-256 are recorded in the [validation receipt](validation.json).

The draft checkpoint was trained with eight positions. A wider pass can generate weaker proposals, and both target and draft execute 50% more rows at width twelve. The second attention band revisits the KV prefix. Compare observed throughput, committed length and both TTFTs against V76 at the same load. An increase in accepted tokens only pays off when it exceeds the increase in time per step. If the judge exposes only aggregate timings, treat the phase split as unknown.

The 16-user test uses 1,536 expert route slots and fits the grouped path. At 32 users the same width needs 3,072 slots and uses indexed Humming. At the maximum graph batch, the wider startup logits tensor alone adds about 99 MiB. The H200 check must establish full output lengths, agreement, later turns, observed peak within stock +2 GB/GPU and TTFT within twice stock. V64's scored peak was +1.9 GB/GPU. Net peak for the V76-derived patch is unknown. [Official rules](https://openkernelchallenge.dev/rules)

## Local evidence

The [frozen patch](patch.diff) changes two files relative to V76. Official export, reverse application and Python compilation passed. The [independent CPU review](independent_review.md) covers the actual launcher, wide attention, tree proposal, acceptance, slot permutation and graph metadata. The V67 capacity suite also passed 2,286,109 allocator-tail checks and 33,792 graph shapes on this source. For a repeat, apply the patch to the pinned SGLang checkout and run `cpu_check.py /path/to/patched/sglang` with NumPy. An official GPU measurement is pending.
