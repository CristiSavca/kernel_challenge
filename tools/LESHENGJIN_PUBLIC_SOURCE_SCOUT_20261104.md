# Bounded LeshengJin public-source scout

November 4, 2026. **No newly released compatible Muse/H200 kernel mechanism was
found in this scope.** No candidate, code reuse, or further branch traversal is
proposed from this result.

The authoritative public-board summary at
`experiments/leaderboards/20261104T104834Z-selected-diagnostic-refresh.json`
lists team `LeshengJin` first overall and first in decoder (2.5510×) and MoE
(2.7817×). The summary contains no linked GitHub identity field. The identically
named [public GitHub account](https://github.com/LeshengJin) is therefore a
search lead, **not independently verified contestant identity**. This scout did
not inspect private jobs, private repositories, or participant communications.

The unauthenticated [repository listing](https://api.github.com/users/LeshengJin/repos?per_page=100&sort=updated&direction=desc)
returned 20 public repositories, with no next page, within the requested cap of
100. All 20 names, descriptions, fork status, licenses and update/push timestamps
were inspected. The newest repository push in that listing is September 23,
2026. No named OKC/Muse release appeared. Pinned default-branch README and LICENSE
reads were limited to the two recent agent projects and the most recently pushed
GPU-serving fork:

| Pinned public repository | Evidence and scope | License |
| --- | --- | --- |
| [omnigent at45316a2](https://github.com/LeshengJin/omnigent/blob/45316a2fd29e22ebdc37487277974b1050868640/README.md) | September 23, 2026 head. General agent orchestration and monitoring framework; the inspected README supplies no Muse kernel implementation or compatible kernel mechanism. | [Apache-2.0](https://github.com/LeshengJin/omnigent/blob/45316a2fd29e22ebdc37487277974b1050868640/LICENSE) |
| [humanize atae1888d](https://github.com/LeshengJin/humanize/blob/ae1888df3572542d4ffe266a18650cf93704f2a5/README.md) | September 23, 2026 head. Agent execution/orchestration framework; its README does not release a challenge kernel. | [Apache-2.0](https://github.com/LeshengJin/humanize/blob/ae1888df3572542d4ffe266a18650cf93704f2a5/LICENSE) |
| [sglang atf111649](https://github.com/LeshengJin/sglang/blob/f11164958088ef8786417e8f09c66574c4ce83ca/README.md) | Default head dated September 20, 2025; repository-wide last push January 5, 2026. Existing serving-framework fork, not a newly released challenge solution in the inspected README. Other inherited branches were not traversed. | [Apache-2.0](https://github.com/LeshengJin/sglang/blob/f11164958088ef8786417e8f09c66574c4ce83ca/LICENSE) |

Other relevant-looking metadata was older: FlashInfer's last push was October 2,
2024, libflash_attn's April 5, 2024, and the unlicensed `rocm_test` repository's
April 22, 2024. These were not expanded into source reads. An old repository may
contain useful general techniques; its age alone is not proof otherwise. This
bounded scout found no evidence tying any such technique to the new leading
submission, so it does not justify another implementation hypothesis.

`experiments/iterations/20261104-leshengjin-public-source-scout.json` preserves the
20 metadata records, pagination result, pinned heads/dates, six README/license
URLs and hashes, public-board snapshot hash and identity limitation. No checkout,
downloaded-source execution, GPU allocation, submission, official-job query or
participant message occurred. Private, differently owned, unlisted and future
releases remain outside this negative finding.
