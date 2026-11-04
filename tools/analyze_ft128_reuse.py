#!/usr/bin/env python3
"""CPU-only FT64/FT128 traffic/resource audit; no candidate execution or edits.

The independent NumPy draws reproduce standalone routing distribution formulas,
not Torch/CUDA RNG seeds or decoder routing. No timings or GPU feasibility claims.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "muse_moe_w4a16/candidates/native_wgs_compact_v10.py"
OUT = ROOT / "experiments/iterations/20261104-ft128-reuse-analysis.json"
TRIALS = 512
BYTES_PER_JOB = 27 * 1024**2


def metrics(counts):
    c = np.asarray(counts, dtype=np.int64)
    j64, j128 = int(((c + 63) // 64).sum()), int(((c + 127) // 128).sum())
    pad64 = sum((min(64, int(x) - s) + 7) // 8 * 8 for x in c for s in range(0, int(x), 64))
    pad128 = sum((min(128, int(x) - s) + 7) // 8 * 8 for x in c for s in range(0, int(x), 128))
    assert pad64 == pad128
    return dict(routes=int(c.sum()), U=int((c > 0).sum()), J64=j64, J128=j128,
                max_count=int(c.max(initial=0)), removed_jobs=j64-j128,
                packed_request_reduction=(j64-j128)/j64 if j64 else 0,
                packed_bytes64=BYTES_PER_JOB*j64, packed_bytes128=BYTES_PER_JOB*j128,
                activation_requests_bytes_both=96 * 4096 * 2 * pad64,
                stage_events64_K128=96 * 32 * j64,
                stage_events128_K64=96 * 64 * j128)


def standalone_counts(rng, m, dist):
    # Activation draws are independent and omitted: this is a distribution model,
    # deliberately not an attempt to reproduce official seeds or cached inputs.
    scores = rng.random((m, 256), dtype=np.float32)
    if dist == "normal":
        z = rng.standard_normal(256).astype(np.float32)
        u = np.maximum(rng.random(256, dtype=np.float32), np.float32(1e-12))
        scores += np.float32(.03) * z / np.sqrt(-np.log(u))
    else:
        heat = rng.random(256, dtype=np.float32)**6 * np.float32(4)
        scores += heat
    ids = np.argpartition(scores, -8, axis=1)[:, -8:]
    all_counts = np.bincount(ids.ravel(), minlength=256).reshape(8, 32)
    counts = all_counts[all_counts.sum(axis=1).argmax()]
    assert counts.max() <= m and counts.sum() <= 8*m
    return counts


def distribution_summary():
    results = []
    for di, dist in enumerate(("normal", "skewed")):
        for m in (128, 256, 384):
            rng = np.random.default_rng(np.random.SeedSequence([20261104, di, m]))
            rows = [metrics(standalone_counts(rng, m, dist)) for _ in range(TRIALS)]
            fractions = np.asarray([r["packed_request_reduction"] for r in rows])
            results.append(dict(M=m, dist=dist, trials=TRIALS,
                mean_U=float(np.mean([r["U"] for r in rows])),
                mean_J64=float(np.mean([r["J64"] for r in rows])),
                mean_J128=float(np.mean([r["J128"] for r in rows])),
                mean_fraction=float(fractions.mean()),
                quantile_fraction=dict(zip(("min", "p10", "p50", "p90", "max"),
                                          np.quantile(fractions, [0, .1, .5, .9, 1]).tolist())),
                zero_fraction=float(np.mean(fractions == 0)),
                fraction_at_least_15pct=float(np.mean(fractions >= .15))))
    return results


def native_metadata(source):
    tree = ast.parse(source)
    cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))
    a = cuda.index("struct Smem {")
    b = cuda.index("\n};", a)+3
    rows = []
    with tempfile.TemporaryDirectory(prefix="okc-ft128-audit-") as td:
        for ft in (64, 128):
            jmax = 32 + 3072//ft + 8
            native = ("#include <cstdint>\n#include <cstdio>\n#define OKC_WGS 1\n#define OKC_WS 0\n"
                      f"constexpr int NW=8,NRB=2,RED_W=4,JMAX={jmax},EMAX=3072;\n" + cuda[a:b] +
                      '\nint main(){printf("%zu\\n",sizeof(Smem));}\n')
            exe = str(Path(td)/f"metadata-{ft}")
            subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-O2", "-o", exe, "-"],
                           input=native, text=True, capture_output=True, check=True)
            size = int(subprocess.run([exe], text=True, capture_output=True, check=True).stdout)
            rounded = (size+127)//128*128
            for k in ((128,) if ft == 64 else (128, 64)):
                weight_slot = 128*k*9//16
                x_slot = ft*k*2
                dynamic = rounded + 3*(weight_slot+x_slot)
                rows.append(dict(FT=ft, K=k, JMAX=jmax, native_metadata=size,
                    rounded_metadata=rounded, packed_weight_slot=weight_slot, x_slot=x_slot,
                    dynamic_smem=dynamic, two_cta_with_driver=2*(dynamic+1024),
                    shared_memory_allows_two=2*(dynamic+1024) <= 228*1024,
                    full_K_stages=4096//k, accumulator_registers_max=4*(ft//8),
                    fixup_float_bytes=2*320*2*(ft//8)*512*4,
                    actual_doneAs_bytes=jmax*(4096//k)*4,
                    padded_x_rows_bound=3072+8*jmax,
                    packed_request_bytes_per_job=96*(4096//k)*weight_slot))
    assert rows[0]["dynamic_smem"] == 98048
    assert all(r["packed_request_bytes_per_job"] == BYTES_PER_JOB for r in rows)
    return rows


def main():
    source = SOURCE.read_bytes()
    recorded = json.loads((ROOT/"experiments/iterations/20261104-moe-wgs-diagnostic.json").read_text())
    observed = [dict(M=r["M"], input_seed=r["input_seed"], **metrics(r["local_counts"]))
                for r in recorded["rows"]]
    profile = json.loads((ROOT/"experiments/iterations/20261104-moe-hardware-profile-recovered.json").read_text())
    profiled = []
    for row in profile["cases"]:
        counted = metrics(row["routing"]["local_counts"])
        reads = row["counter_launch"]["metrics"]["dram__bytes_read.sum"]["value"]
        profiled.append(dict(M=row["M"], **counted, measured_dram_reads=reads,
            removable_logical_requests_over_measured_reads=counted["removed_jobs"]*BYTES_PER_JOB/reads))
    cases = []
    for m in (128, 256, 384):
        for label, counts in (("eight_always_local", [m]*8+[0]*24),
                              ("uniform_local_load", [m//8]*32),
                              ("one_hot_23_cold", [m]+[1]*23+[0]*8)):
            cases.append(dict(M=m, label=label, counts=counts, **metrics(counts)))
    # Same M/U/route count gives very different opportunities, so decoder load
    # targets alone cannot establish repeated-weight traffic.
    for label, counts in (("balanced_R1000_U24", [42]*16+[41]*8+[0]*8),
                          ("hot_R1000_U24", [128]*7+[7]*2+[6]*15+[0]*8)):
        assert sum(counts) == 1000 and sum(x > 0 for x in counts) == 24
        cases.append(dict(M=256, label=label, counts=counts, **metrics(counts)))
    for c in range(385):
        r = metrics([c])
        assert 0 <= r["packed_request_reduction"] <= .5
        assert r["stage_events128_K64"] >= r["stage_events64_K128"]
    evidence = dict(candidate_source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source).hexdigest(), gpu_used=False,
        candidate_edited=False, official_results=False,
        methodology="Independent NumPy distribution draws; not Torch seeds; not decoder route distribution. Packed requests are not measured DRAM bytes.",
        prior_evidence="tools/POST_NCU_TRAFFIC_REASSESSMENT.md",
        observed_standalone=observed, observed_profiled_standalone=profiled, valid_histogram_examples=cases,
        resource_budget=native_metadata(source.decode()), distribution_model=distribution_summary())
    OUT.write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps({"resources": evidence["resource_budget"], "distribution_model": evidence["distribution_model"]}, indent=2))


if __name__ == "__main__":
    main()
