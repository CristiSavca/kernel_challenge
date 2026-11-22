"""Price the modeled W2 byte difference against an archived cold first wave."""
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
cold = here.parent / "v270_wide_prompt_weight_reuse/work.json"
budget = here.parent / "v271_cold_service_envelope/audit.json"
inputs = {p: json.loads(p.read_bytes()) for p in (cold, budget)}
shape = next(s for s in inputs[cold]["samples"] if s["tokens"] == 16384)
legacy = shape["rows"]["ResidualPromptWeightPipeline"]
wide = shape["rows"]["WidePromptWeightPipeline"]
window = next(r for r in inputs[budget]["windows"]["v41"] if r["load"] == 32)
peak = inputs[budget]["geometry"]["HBM_peak_TBps"] * 1e12
saved = (legacy["W2_packed_request_MiB"] - wide["W2_packed_request_MiB"]) * 2**20
extra_a = (wide["A_multicast_full_box_MiB"] - legacy["A_multicast_full_box_MiB"]) * 2**20
assert extra_a > saved > 0
seconds = saved / peak
first_wave = seconds * window["expert_16k_equivalent_layer_calls"]
print(json.dumps(dict(source_sha256={p.parent.name+"/"+p.name:sha256(p.read_bytes()).hexdigest() for p in inputs},
    uniform_16k_prompt_W2_request_MiB=[legacy["W2_packed_request_MiB"],wide["W2_packed_request_MiB"]],
    issued_rows=[legacy["issued_tensor_rows"],wide["issued_tensor_rows"]],
    modeled_activation_full_box_MiB=[legacy["A_multicast_full_box_MiB"],wide["A_multicast_full_box_MiB"]],
    additional_activation_full_box_MiB=extra_a/2**20, equal_miss_fraction_extra_total_MiB=(extra_a-saved)/2**20,
    break_even_incremental_activation_DRAM_fraction_over_weight_DRAM_fraction=saved/extra_a,
    assumed_per_rank_HBM_TB_s=peak/1e12, conditional_microseconds_per_layer=seconds*1e6,
    equivalent_16k_cold_layer_calls=window["expert_16k_equivalent_layer_calls"],
    conditional_seconds_on_32_user_V41_first_wave=first_wave, archived_first_wave_s=window["first_wave_s"],
    fraction=first_wave/window["first_wave_s"], 
    assumptions="Every equivalent full chunk has the modeled uniform routes. Each saved packed request is a DRAM miss transferred at peak. The 0.43 s conversion counts saved weights alone. The full-box activation calculation charges all rows including tile padding. Its break-even ratio assumes each extra A byte and each removed weight byte has a fixed DRAM miss fraction. Actual cache hits, out-of-bounds zero fill, issue work, residency, imbalance and overlap can change the result. This is a units conversion, not a performance bound or forecast."),indent=2))
