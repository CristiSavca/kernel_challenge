#!/usr/bin/env python3
"""Independent CPU register-admission/budget audit, not CUDA compilation."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "experiments/iterations/20261104-cuda-register-limit-critic.json"
OUT = ROOT / "experiments/iterations/20261104-wgs-setmaxnreg-budget-critic.json"


def main():
    pin = json.loads(PIN.read_text())
    archive = urllib.request.urlopen(pin["source_url"], timeout=20).read()
    assert hashlib.sha256(archive).hexdigest() == pin["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tar:
        name = next(n for n in tar.getnames() if n.endswith("/include/cuda_occupancy.h"))
        header = tar.extractfile(name).read()
    assert hashlib.sha256(header).hexdigest() == pin["header_sha256"]
    cpp = r'''
#include <cstdio>
#include <initializer_list>
#include "cuda_occupancy.h"
int main() {
  cudaOccDeviceProp p{};
  p.computeMajor=9; p.computeMinor=0;
  p.regsPerBlock=65536; p.regsPerMultiprocessor=65536; p.warpSize=32;
  for (int threads: {576,640}) for (int regs: {80,88,89,96,97,104,112}) {
    cudaOccFuncAttributes a{}; a.numRegs=regs;
    cudaOccResult r{}; cudaOccPartitionedGCConfig gc=PARTITIONED_GC_OFF;
    int limit=-1;
    auto err=cudaOccMaxBlocksPerSMRegsLimit(&limit,&gc,&r,&p,&a,threads);
    printf("%d %d %d %d %d\n",threads,regs,int(err),limit,r.allocatedRegistersPerBlock);
  }
}
'''
    with tempfile.TemporaryDirectory(prefix="okc-setmaxnreg-budget-") as td:
        path = Path(td)
        (path / "cuda_occupancy.h").write_bytes(header)
        exe = str(path / "audit")
        subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-O2", "-I", td,
                        "-o", exe, "-"], input=cpp, text=True, capture_output=True, check=True)
        raw = subprocess.check_output([exe], text=True)
    names = ("threads", "reported_registers", "cuda_occ_error",
             "register_admissible_ctas", "allocated_registers_per_cta")
    rows = [dict(zip(names, map(int, line.split()))) for line in raw.splitlines()]
    assert all(row["cuda_occ_error"] == 0 for row in rows)
    by_key = {(r["threads"], r["reported_registers"]): r for r in rows}
    assert by_key[640,96]["register_admissible_ctas"] == 1
    assert by_key[640,96]["allocated_registers_per_cta"] == 61440
    assert by_key[640,97]["register_admissible_ctas"] == 0
    budgets = []
    for initial in (80,88,96):
        pool = by_key[640,initial]["allocated_registers_per_cta"]
        for producer, consumer in ((24,104),(32,104),(24,112),(32,112),(40,112),(24,120)):
            need = 128*producer + 512*consumer
            budgets.append(dict(initial_reported_registers=initial,
                assumed_cta_pool=pool, producer_quota=producer, consumer_quota=consumer,
                required_registers=need, balance=pool-need,
                pool_conservation_satisfied=need <= pool,
                consumer_increase_requests=512*(consumer-initial),
                producer_decrease_returns=128*(initial-producer)))
    balanced = next(r for r in budgets if r["initial_reported_registers"] == 96
                    and r["producer_quota"] == 32 and r["consumer_quota"] == 112)
    assert balanced["balance"] == 0
    over = next(r for r in budgets if r["initial_reported_registers"] == 96
               and r["producer_quota"] == 24 and r["consumer_quota"] == 120)
    assert over["balance"] == -3072 and over["required_registers"] < 65536
    result = dict(tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pinned_parent_evidence_sha256=hashlib.sha256(PIN.read_bytes()).hexdigest(),
        primary_url=pin["source_url"], archive_sha256=pin["archive_sha256"],
        header_sha256=pin["header_sha256"],
        method="Unmodified NVIDIA CUDA13.0.96 occupancy-header register-limit function compiled by native clang with SM90 properties; no CUDA runtime or GPU.",
        rows=rows, budgets=budgets,
        findings=[
            "At640 threads, reported96 registers is register-admissible with61440 allocated registers;97 is not admissible. This does not establish compiler output or runtime allocation for a new source.",
            "Assuming an initial61440-register CTA allocation,128 producer threads at32 plus512 consumers at112 exactly conserve the pool. All consumer increments require8192 registers, exactly what the producer decrease returns.",
            "Producer24/consumer120 needs64512, below the SM total65536 but3072 beyond the assumed CTA pool. The SM-total arithmetic is insufficient and could lead to blocked inc instructions.",
            "Initial80 or88 allocations do not support the112/32 target. A launch-bound maximum alone supplies no verified lower bound on actual initial allocation.",
            "Rows89 and97 expose allocation granularity; do not assume padding registers are available as owned register quotas without compiler/runtime evidence."
        ],
        limits=[
            "This is a static register-allocation model, not evidence setmaxnreg survives ptxas or that dynamic quotas are honored.",
            "Whole-warpgroup convergence, producer-before-consumer availability, barrier participation, numerical correctness, spills and speed require separate proofs or measurements.",
            "Register admission excludes shared memory, thread limits, barriers, actual compiler state and runtime device state."
        ], gpu_used=False, candidate_written=False)
    OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(output=str(OUT), rows=rows, selected_budget=balanced,
                          sm_fits_but_cta_pool_fails=over), indent=2))


if __name__ == "__main__":
    main()
