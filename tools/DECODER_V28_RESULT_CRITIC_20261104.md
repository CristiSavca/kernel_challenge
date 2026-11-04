V28 fixed attempt: **CLOSE; no scored submission, retry or shape rescue.** Job `2976c25cc3dc41b1` used exact frozen source `d1b1bcf5c241710697bc0d2fb39836b1d7064c262bff7f820b564085e2136d9b`. It passed setup, all13 numerical tests,12 benchmarks and12 profiles with clean checks and no flags. The official score is **2.1146266798**, CI[2.0829072796,2.1463460800].

I independently checked all three raw receipts, source and profile hashes, complete evaluator file inventory/image, case metadata except seeds, and the six46:16 weighted-load score formula. All recomputed ratios match root's separate analysis.

| Comparison | Whole score ratio | Affected load speedGM | Affected latencyGM | Control latencyGM |
|---|---:|---:|---:|---:|
| Public97af | 0.969486 | 0.951014 | 1.051510 | 1.011821 |
| Private97af | 0.971503 | 0.954594 | 1.047566 | 1.011417 |

Both affected averages fail the required1.03 speed ratio, and both whole scores fail1.015. Every affected case is slower than both parents. Public affected cases4,8,9,10 exceed the3% latency cap; private affected cases4,5,8,9,10 exceed it. Public controls pass the fixed gates; private unchanged controlcase0 is3.381% slower and also fails its individual cap. The affected result is materially worse than the roughly1.1–1.2% control-average drift. These are cross-run normalized observations on different nodes/seeds, not a same-input causal experiment.

All six affected profiles contain the uniquely named new MAIN kernel; all six controls contain the unchanged small-kernel marker. The MAIN rows are102.8/98.9µs atT32,179.1/172.9µs atT128 and210.3/217.7µs atT256. They are longer than the corresponding retained parent rows, but separate profiler durations do not equal benchmark critical-path time and cannot isolate atomic service, additional payload traffic, register allocation or PDL effects. The retained log contains only profiler USDT lines: actual register/local/occupancy data and absence of compiler warnings are unknown.

The outcome rejects transferring the serving scalar-CAS improvement to this public vector-atomic parent. The source-level tradeoff—another20MiB of scratch, doubled owner reads and extra sums/range work—remains a plausible explanation, not an identified cause. Preserve the passing public97af implementation. The independently passing V26 serving core and separately reviewed V27 adapter remain distinct work; this failed decoder port does not negate their evidence.
