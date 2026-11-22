# DFLASH selector tree feasibility

The Muse DFlash2 checkpoint uses five draft layers and a trained width of
eight. The block contains one anchor and seven proposals. The rank-256
selector scores up to 16 candidates per position and conditions later
scores on the previous chosen candidate. Its existing `build_lattice`
in `srt/models/dflash.py` exposes the needed predecessor relationships.
The tree design can reuse this lattice and verify up to seven levels without
changing trained width. [Muse DFlash2 model card](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2)

Use a fixed target row budget, including an anchor row, and order proposals
by cumulative prefix probability. With calibrated conditional probabilities,
if `mass(node)` is the product along a root-to-node path, expected target
tokens committed by exact greedy verification are
`1 + sum(mass(node))` over the verified non-root nodes. The initial one
accounts for the target bonus token. A child cannot exceed its parent's mass,
so a best-first expansion with parents first on ties maximizes this expectation
under the fixed budget and ancestor closure. The target's own greedy token
still determines every acceptance, so imperfect selector calibration affects
efficiency only. A softmax over top-16 selector scores is an uncertain proxy
for true path mass and needs calibration or replay evidence.

The [CPU proof](tree_cpu_check.py) contains no model weights. It passed 960
fixed-budget optimality and exact acceptance cases against exhaustive
enumeration on small trees. It also passed 868 packed sliding-window mask
and path-order checks with arbitrary sibling slot ordering. Run
`python3 tree_cpu_check.py` to reproduce these cases.

| Illustrative conditional probabilities | Linear 8 rows | Tree 8 rows | Tree 16 rows |
| --- | ---: | ---: | ---: |
| 0.79 top choice at every depth | 4.0395 | 4.0574 | 5.1815 |
| 0.55 top choice at every depth | 2.2036 | 3.1664 | 4.0915 |
| 0.55 at root, then 0.90 | 3.8694 | 4.1109 | 6.2720 |

The numbers are expected committed tokens per step for hypothetical binary
paths. They are not judge acceptance or measurements of selector entropy.
The first example gains 0.44% at eight rows. With 16 rows the full step
can cost at most 28.3% more for equal decode throughput. Doubling target
rows also doubles activation and per-token expert work at a given live
batch, and it crosses V62's packed-Q width limit. The third example needs
far less uniform uncertainty than the second to benefit from additional
breadth. A trained depth of seven limits every tree to eight committed tokens
in one target step, regardless of how many nodes were verified.

## Integration requirements

- `EagleVerifyInput`, `build_tree_kernel_efficient` and
  `verify_tree_greedy_func` supply candidate, ancestor and exact greedy
  traversal interfaces. The UNO builder emits a related tree layout but
  scores tokens by depth. DFLASH's selector also needs the predecessor
  candidate for each edge. The DFLASH hook forces `topk=1`, and the
  target CUDA graph captures the linear `DFlashVerifyInput`. A tree needs
  a bounded variant for target verification, while the draft retains its
  original width and attention.
- FA3's native full-attention tree uses a prefix call and a masked suffix
  call, then merges the softmax states. Its SWA path instead creates one
  full prefix-plus-ancestors page row per node and requires `page_size=1`.
  At 32 requests, eight nodes and context 131,072, the static SWA table
  consumes roughly 128 MiB. At the context limit, every rebuild writes
  roughly that much table data even though each query only sees its short
  window. The per-node one-query layout also removes the packed reuse
  of shared prefix KV across nodes. A viable version needs exact packed
  sliding-window attention using each node's logical position and
  ancestor membership. The CPU proof specifies which prefix and
  ancestors it can read. A simpler prototype can materialize only each
  node's visible window, while keeping the attention arithmetic identical.
- The DFLASH worker currently reads the first accepted rows of a contiguous
  target aux tensor and their contiguous slot IDs. Branch acceptance returns
  an ordered index list. The worker must gather aux rows in path order and
  remap or compact full and SWA KV together, keeping unused reserve
  reachable to the existing allocator. The remap must complete before
  the next step can use the newly published prefix. Page-one slot
  permutation can preserve both accepted and unaccepted reservations;
  wider pages need a separate layout and allocation proof.

There is no drop-in tree launcher in the pinned source. The CPU proof
establishes the fixed-budget objective and mask and commit order. It does
not establish numerical agreement for a packed GPU implementation or
speed. A representative ordinary run's lattice calibration and current
step timing can establish a budget worth implementing. The same-request
draft also depends on the preceding accepted target token and aux state,
so scheduling it early cannot remove that dependency without a new
prediction algorithm.
