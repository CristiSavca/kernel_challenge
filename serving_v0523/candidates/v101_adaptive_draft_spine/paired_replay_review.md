# Paired online replay on the binary selector lattice

High target agreement exposes a cold-start cost in V101. At 94% agreement, frozen V101 committed 5.362 rows per verification call over the first eight calls, compared with 5.889 for V99. Requiring strict evidence before overriding a V99-style endpoint brought that figure to 5.849. V101 performed best on rapidly alternating conditions. These results measure committed rows on a fixed synthetic lattice. They do not measure serving throughput, GPU runtime or the prevalence of each condition in the challenge.

| Target top-choice agreement | V101 / V99, 8 calls | V101 / V99, 64 calls | Strict interior / V99, 8 calls | Strict interior / V99, 64 calls |
| --- | ---: | ---: | ---: | ---: |
| 60% | 1.0180 | 1.0055 | 1.0100 | 1.0042 |
| 70% | 1.0307 | 1.0344 | 1.0225 | 1.0322 |
| 80% | 1.0088 | 1.0422 | 1.0182 | 1.0412 |
| 94% | 0.9104 | 0.9923 | 0.9931 | 1.0006 |
| 60% / 94%, alternating every call | 1.0643 | 1.1320 | 1.0724 | 1.1116 |
| 60% / 94%, alternating every eight calls | 0.9481 | 0.9823 | 0.9991 | 0.9918 |

The follow-up controller maintains a scalar V99 endpoint preference as well as V101’s adjacent margins. It chooses an interior spine only when its utility exceeds both endpoint utilities. Both states learn solely from that controller’s own accepted prefix and bonus. The remaining 64-call block-switch loss was 0.0353 rows per call relative to V99. Its paired 95% Monte Carlo interval was −0.0386 to −0.0320 rows. The eight-call block-switch difference was −0.0040 rows with an interval spanning zero. Intervals describe sampling uncertainty within this model.

The initial experiment fixed four stationary probabilities, two alternation rates, 4,096 seeds, and horizons of eight and 64 calls before running the comparisons. Half the seeds start each alternation high. Each controller sees the same underlying uniforms at a given request, call and depth. The experiment resets the lattice at each call, and a controller learns only what its chosen proposal revealed. It does not advance through a common continuous conversation when the policies commit different numbers of tokens. The follow-up used the same draws and recovered both initial controller means exactly.

The fixture uses the actual proposal, acceptance and history-update kernel bodies. It enumerates every binary continuation and each of the eight possible spine depths, producing 20 observable committed prefixes. Its transition lookups cover every reachable integer prior. For every observed prefix, 33 heterogeneous prior vectors reproduced the original selector and update. The stored results include fixed-policy expectations and paired per-request uncertainty. [Initial replay](paired_binary_replay.json) · [Follow-up replay](paired_endpoint_tie.json) · [Reproduction fixture](paired_binary_replay.py) · [Follow-up fixture](paired_endpoint_tie.py)
