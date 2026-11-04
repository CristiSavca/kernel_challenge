# Lean setup cleanup — 2026-11-04

Completed 2026-11-04T18:29:10.157928+00:00 under the standing cleanup authorization and root's exact-path assignment. Removed only the nine inspected regular generated files below using non-forced individual unlink operations. No directories, glob patterns or Git cleanup commands were used.

| Removed path | Logical bytes | Allocated bytes before removal |
| --- | ---: | ---: |
| `.toolchains/lean-4.35.0-darwin_aarch64.tar.zst` | 570,422,530 | 570,425,344 |
| `formal/.lake/build/lib/lean/OKC/BVKernel.ilean.hash` | 16 | 4,096 |
| `formal/.lake/build/lib/lean/OKC/BVKernel.olean.hash` | 16 | 4,096 |
| `formal/.lake/build/lib/lean/OKC/BVKernel.trace` | 1,845 | 4,096 |
| `formal/.lake/build/lib/lean/OKC/BVKernel.ilean` | 9,040 | 12,288 |
| `formal/.lake/build/lib/lean/OKC/BVKernel.olean` | 743,392 | 745,472 |
| `formal/.lake/build/ir/OKC/BVKernel.setup.json` | 137 | 4,096 |
| `formal/.lake/build/ir/OKC/BVKernel.c` | 371,191 | 372,736 |
| `formal/.lake/build/ir/OKC/BVKernel.c.hash` | 16 | 4,096 |

Total logical bytes removed: **571,548,183**; summed allocated blocks before removal: **571,576,320 bytes**.

Actual `df -k` available space immediately before: **40,135,600 KiB**; after: **40,693,732 KiB**; observed change: **+558,132 KiB** (+571,527,168 bytes). This is a filesystem-wide observation and can include concurrent system activity; it is not assumed to equal file sizes.

Checks before deletion: Git tracked none of the exact paths; the process snapshot showed no active Lean/Lake/verifier process; `lsof -nP` on all nine exact paths returned1 with empty stdout/stderr, including a second check immediately before removal. The abandoned `formal/OKC/BVKernel.lean` source was already absent, the active proof does not import it, and the Lake root is only `OKC.Dq4`.

The archive SHA256 was rechecked as `c0c6f34f35ba1dcc5c4a5fd106fefab9eb5e8469712624dd2c868d51b63177c2`, matching the retained `.toolchains/lean-release.json`. The installation was already complete and executable before deletion. After deletion all nine files are absent, installed Lean still reports:

```text
Lean (version 4.35.0, arm64-apple-darwin24.6.0, commit eb6cafb7290e8bd5f2937522d8d804f58f02bafc, Release)
```

Preserved the entire installed toolchain, its LICENSE and release metadata, active Dq4 source/package files, all active proof-run directories and unique result evidence. Active proof SHA256 remains `203f9527f1b94e85a79388b3030d74566a1a9f4809c833b5c9e5e26057d57bbd`. No proof runs or their parent directories were inspected for deletion or modified.

Exact filesystem snapshots:

```text
Filesystem   1024-blocks      Used Available Capacity iused     ifree %iused  Mounted on
/dev/disk3s5   971319460 893921704  40135600    96% 5844780 401356000    1%   /System/Volumes/Data

Filesystem   1024-blocks      Used Available Capacity iused     ifree %iused  Mounted on
/dev/disk3s5   971319460 893363572  40693732    96% 5844771 406937320    1%   /System/Volumes/Data
```
