# V23 exact-one rescale — independent terminal review

**REJECT and close the fixed attempt.** Official private job `0193a093baa446f3`
passes correctness but fails the predeclared performance screen against both
incumbent runs. No retry, branch/threshold sweep, public promotion or decoder port.
Active attention87a9 remains unchanged.

I independently read the terminal receipt, archived candidate, profile JSON and
both parent receipts/profiles. Candidate SHA256 is
`c6b90f168bfdc21a8583b6a4bd03cebc0e6c4e03f994f74759c54ab5b677335c`;
both parents bind87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c.
All three runs have16 passing tests,12 passing benchmarks and12 profiles,
returncode0, passed=true, clean checks and no flags. Their full evaluator/image
records and H200 GPU type match. Test/benchmark/profile case contracts match
after removing only seed. Archived profiles equal receipt records; benchmark
archives omit the receipt's empty `flags: []`, the only serialization difference.

The official score is **2.591127828409182**, CI
**[2.552260910983044,2.6476042884897177]**. I independently recomputed all three
scores as the geometric mean of six per-load weighted baseline/candidate latency
ratios (sliding/full weights46/16), then computed case ratios from paired ratios.
This agrees with the separate root/author analyzer; it is not a12-case timing GM.

| Fixed metric | Versus public8f9e5971f95841a0 | Versus privatec21d391dbf604bb5 | Required |
| --- | ---: | ---: | --- |
| Official aggregate speed ratio | 0.992020473 | 0.980544824 | ≥1.01 each |
| Five affected-case speed GM | 0.983746588 | 0.971391979 | ≥1.03 each |
| Five affected-case latency GM | 1.016521950 | 1.029450543 | descriptive reciprocal |
| Seven-control latency GM | 1.007551226 | 1.001613462 | [0.98,1.02] each: pass |
| Affected cases above1.02 latency | indices1,2 | indices1,5 | none |
| Jointly faster affected cases | 0/5 against both | 0/5 against both | ≥3/5 |

Public maximum affected regression is index2, B16sliding, **+5.46453%** normalized
latency. Private maximum is index5, B32full, **+13.06439%**. Public index5 alone is
faster; private indices3/4 are faster, leaving no joint winner. The aggregate,
affected-GM, individual-cap and joint-breadth gates fail; controls pass. The full
case/load values and source/evaluator hashes are in
`experiments/iterations/20261104-attention-exact-rescale-result-critic.json`.

All12 actual profile specializations match both parents. Affected1–5 remain
Q1non-ONE: B1full `<1,false,11>`, B16/B32sliding `<1,false,1>`, B16/B32full
`<1,true,0>`. Control0 remains ONE `<1,false,13>`;6–11 remain Q8. Thus no dispatch
change explains away the failed fixed screen. Profile symbols do not reveal how
often alpha equaled1 or whether the compiler omitted products; no SASS/resource
or branch-frequency evidence was collected.

These are separate official runs with different seeds/nodes and overlapping
confidence intervals. The operational screen rejects an unproven gain; it does
not prove a causal branch slowdown, GPU FTZ defect or a precise recoverable cost.
The source-only operation-removal argument survives as a semantic observation,
but it did not produce a qualifying measured improvement. No further analysis
or favorable subset selection can turn this result into acceptance under the
fixed plan70c4e49b.

Reviewer changed only this note and its paired JSON. No provider call, polling,
GPU execution, candidate edit or publication was performed.
