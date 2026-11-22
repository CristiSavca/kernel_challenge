"""Price a two-stage QKV exchange against the measured V41 schedule."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TP = 8
HIDDEN = 8192
OUTPUT = 2304
LAYERS = 62
BF16_BYTES = 2


def shape(rows):
    local = (rows + TP - 1) // TP
    half = TP // 2
    returned = local * TP * OUTPUT * BF16_BYTES
    half_input = local * half * HIDDEN * BF16_BYTES
    half_ops = 2 * local * half * HIDDEN * OUTPUT
    return dict(local=local, global_padded=local * TP, eligible=local >= 512 and local * half >= OUTPUT,
                local_cohort_send=(half - 1) * local * HIDDEN * BF16_BYTES,
                cross_half_send=half_input, total_flops=half_ops * 2,
                explicit_gather_plus_output=half_input * 2 + returned,
                dispatched_additional_output=returned // 2, additional_logical_weight_read=HIDDEN * OUTPUT * BF16_BYTES)


def overlap_seconds(plan, gb_per_s, tf_per_s):
    return min(plan["cross_half_send"] / (gb_per_s * 1e9), plan["total_flops"] / (2 * tf_per_s * 1e12))


def report():
    trace = json.loads((HERE.parent / "v251_hicache_residency/trace.json").read_bytes())["v41"]
    observed = json.loads((HERE.parent / "v193_scheduler_cache_poll/trace.json").read_bytes())["V41"]["rounds"]
    logged = next(item for item in trace["sources"] if item["path"].endswith("server.log"))
    raw = (ROOT / logged["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == logged["sha256"]
    pattern = re.compile(r"Prefill batch, #new-seq: \d+, #new-token: (\d+)")
    groups = []
    for line in raw.decode().splitlines():
        if "Cache flushed successfully!" in line:
            groups.append([])
        elif groups and "TP0 EP0" in line and (match := pattern.search(line)):
            groups[-1].append(int(match[1]))
    assert len(groups) == len(observed) == len(trace["windows"]) == 7
    rows = []
    for sizes, ref, wave in zip(groups, observed, trace["windows"]):
        assert len(sizes) == ref["prefill_scheduler"]["passes"]
        assert sum(sizes) == ref["real_prefill_rows"] == wave["uncached_prompt_rows"]
        eligible = [p for n in sizes if (p := shape(n))["eligible"]]
        scenarios = []
        for gb, tf in ((450, 989), (450, 400), (225, 800), (225, 400)):
            seconds = LAYERS * sum(overlap_seconds(p, gb, tf) for p in eligible)
            scenarios.append(dict(directional_GB_s=gb, dense_BF16_TFLOP_s=tf,
                                  perfect_overlap_s=seconds, fraction_of_measured_wall=seconds/ref["wall_s"]))
        rows.append(dict(users=ref["concurrent_users"], rep=ref["rep"], parent_wall_s=ref["wall_s"],
                         prefill_calls=len(sizes), eligible_calls=len(eligible), full_chunks=sizes.count(16384),
                         eligible_padded_rows=sum(p["global_padded"] for p in eligible), scenarios=scenarios))
    return dict(source=logged, shape_16384=shape(16384), layers=LAYERS, trace=rows,
                scope="V41 logged prefill sizes rounded up to TP8. Mixed graph padding, backend kernels, launch costs, and achieved concurrent rates are unmeasured. Each scenario overlaps only buddy transfer and first GEMM and assumes a serial parent with equal component rates.",
                hardware_reference="https://www.nvidia.com/en-us/data-center/h200/", measured_candidate=False)


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
