#!/usr/bin/env python3
"""Read-only offline WAIT2 terminal audit; no polling, candidate or GPU work."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from analyze_kperm_and_decoder_roi import OUT, ROOT, geomean, load, sha
from analyze_dense_escape_result import kernel_rows


def main():
    receipt, profile, provenance = load("f685c2533ebc4214")
    assert provenance["source_sha256"] == "97ffccc30a90ce8400cddef800d932889822eaf876fbdedaec18b1013e8f66b6"
    result = receipt["response"]["result"]
    counts = Counter(r["event"] for r in result["records"])
    assert counts == {"setup": 1, "test": 5, "benchmark": 6, "profile": 6}
    assert result["passed"] and result["returncode"] == 0 and result["error"] is None
    assert not result["flags"] and result["checks"] == {"flagged": [], "ok": True, "rejected": []}
    assert all(r.get("status", "pass") == "pass" for r in result["records"])
    log_path = Path(receipt["response"]["output"]["log"])
    log = log_path.read_text()
    marker = re.search(r"OKC_WGS_WAIT2 resources threads=(\d+) dynamic=(\d+) occupancy=(\d+)/(\d+) regs=(\d+)/(\d+) local=(\d+)/(\d+) static=(\d+)/(\d+)", log)
    assert marker
    assert tuple(map(int, marker.groups())) == (576,231168,1,1,96,96,0,0,0,0)
    compiler = []
    lines = log.splitlines()
    for big in (0,1):
        symbol = f"_ZN3okc9moe_fusedILb{big}EEEvNS_6ParamsE"
        warnings = [dict(line=i+1,text=line) for i,line in enumerate(lines)
                    if "C7511" in line and symbol in line]
        assert len(warnings) == 1 and "insufficient register resources" in warnings[0]["text"]
        properties = lines.index("ptxas info    : Function properties for " + symbol)
        assert lines[properties+1].strip() == "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
        assert lines[properties+2].strip() == "ptxas info    : Used 96 registers, used 16 barriers"
        compiler.append(dict(BIG=bool(big),serialization=warnings[0],stack_bytes=0,
                             spill_store_bytes=0,spill_load_bytes=0,registers=96,barriers=16))
    comparisons = []
    for job in ("d24dd50196cb4158","bcb761dfca264db0"):
        old_receipt, old_profile, old_provenance = load(job)
        assert old_provenance["source_sha256"] == "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"
        assert provenance["evaluator_sha256"] == old_provenance["evaluator_sha256"]
        cases = []
        for new,old,np,op in zip(profile["benchmarks"],old_profile["benchmarks"],profile["profiles"],old_profile["profiles"]):
            assert {k:v for k,v in new["case"].items() if k != "seed"} == {k:v for k,v in old["case"].items() if k != "seed"}
            cases.append(dict(tokens=new["case"]["bs"],changed_path=new["case"]["bs"]>16,
                candidate_seed=new["case"]["seed"],parent_seed=old["case"]["seed"],
                speedup_ratio=new["speedup"]/old["speedup"],
                harness_median_latency_ratio=new["median_ns"]/old["median_ns"],
                raw_median_latency_ratio=new["median_raw_ns"]/old["median_raw_ns"],
                candidate_harness_median_us=new["median_ns"]/1000,parent_harness_median_us=old["median_ns"]/1000,
                candidate_raw_median_us=new["median_raw_ns"]/1000,parent_raw_median_us=old["median_raw_ns"]/1000,
                baseline_time_ratio=new["baseline_ns"]/old["baseline_ns"],
                candidate_profile_kernels=kernel_rows(np),parent_profile_kernels=kernel_rows(op)))
        groups = {}
        for name,changed in (("affected",True),("controls",False)):
            group = [c for c in cases if c["changed_path"] == changed]
            groups[name] = {key+"_geomean":geomean([c[key] for c in group]) for key in
                ("speedup_ratio","harness_median_latency_ratio","raw_median_latency_ratio","baseline_time_ratio")}
            groups[name]["inverse_speedup_ratio_geomean"] = 1/groups[name]["speedup_ratio_geomean"]
        assert math.isclose(geomean([c["speedup_ratio"] for c in cases]),provenance["speedup"]/old_provenance["speedup"],rel_tol=1e-12)
        old_log = Path(old_receipt["response"]["output"]["log"]).read_text()
        comparisons.append(dict(parent=old_provenance,cases=cases,groups=groups,
            aggregate_speedup_ratio=provenance["speedup"]/old_provenance["speedup"],
            same_gpu_node=provenance["node"]==old_provenance["node"],
            aggregate_reported_intervals_overlap=max(provenance["speedup_ci"][0],old_provenance["speedup_ci"][0]) <= min(provenance["speedup_ci"][1],old_provenance["speedup_ci"][1]),
            parent_retained_compiler_warning_lines=[line for line in old_log.splitlines() if re.search(r"C75\d\d",line)]))
    report = dict(candidate=provenance,record_counts=dict(counts),checks=result["checks"],flags=result["flags"],
        tests=[r for r in result["records"] if r["event"]=="test"],
        resources=dict(log=str(log_path.relative_to(ROOT)),log_sha256=sha(log_path),log_bytes=log_path.stat().st_size,
            log_equals_receipt_tail=log==result["log_tail"],marker=marker.group(),compiler_entries=compiler,
            interpretation="Explicit C7511 serialization due to insufficient WGMMA pipeline registers in both specializations, despite96 registers and zero reported stack/spill/local bytes. Actual host admission query gives1/1, not achieved occupancy.",
            truncation="Retained output includes the WGS compile command, both entry-property blocks, linker step and resource marker. It originates from log_tail; full original process output is not independently certified. Positive C7511 and the displayed zero-spill properties are established; absence of other warning codes is not a global proof."),
        comparisons=comparisons,
        verdict=dict(decision="REJECT WAIT2 promotion: no public score, decoder integration or identical measurement retry.",
            performance="Fails the prespecified5% affected-latency improvement target. All three changed raw medians regress versus both exact-parent runs, and all three relative speedups decline. Unchanged controls improve geometrically. Reported aggregate intervals overlap; do not claim a precise causal regression magnitude.",
            baseline_caution="M128 harness median_ns slightly improves while its relative speedup declines because baseline_ns shifts about5.7–6.1%. Inverse speedup ratios, direct harness median ratios and raw event-median ratios are distinct quantities and are retained separately.",
            compiler_causality="The candidate demonstrably serializes WGMMA. Parent logs do not expose equivalent compiler diagnostics, so this audit does not prove serialization was newly introduced by WAIT2 or quantify its timing contribution.",
            redistribution="Conditional GO for an exact bounded implementation plan addressing observed serialization, not a source or GPU approval. Full128-thread producer WG with32/112 quotas must conserve an established61440-register initial pool, preserve joins and be compiled under the96-register birth constraint. The mechanism succeeds only if intended quotas survive compilation and C7511/C7512 serialization is removed without material spills; then numerical and speed gates still apply.",
            recovery="First job a0dd88ebc8964b3f returned refunded infrastructure failure without measurements. This audit uses its one authorized exact-source recoveryf685c2533ebc4214; it is not a timing-selected rerun."),
        gpu_used_by_critic=False,candidate_edited=False)
    out=OUT/"20261104-wgs-wait2-official-critic.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(output=str(out.relative_to(ROOT)),resources=report["resources"],comparisons=[dict(parent=c["parent"]["job"],groups=c["groups"],aggregate_speedup_ratio=c["aggregate_speedup_ratio"]) for c in comparisons]),indent=2))


if __name__ == "__main__":
    main()
