"""Sensitivity of the official V64 curve when only the two high-load rungs change."""
import hashlib
import json
from pathlib import Path
import runpy

here = Path(__file__).resolve().parent
calibration = next(here.parent.glob("v265_*/audit.py"))
receipt = here.parents[1] / "runs/abd217baf0ed49fd.json"
estimator, baseline = runpy.run_path(str(calibration))["calibrate"](json.loads(receipt.read_bytes()))

def score(value, interactivity):
    upper = [1, 1, value, value]
    return estimator(upper if interactivity else [1, 1, 1, 1], upper)

def threshold(goal, interactivity):
    # The calibrated frontier is monotone through each of these positive test curves.
    low, high = 1.0, 1.0
    while score(high, interactivity) < goal:
        high *= 1.25
    for _ in range(60):
        mid = (high + low) / 2
        if score(mid, interactivity) >= goal:
            high = mid
        else:
            low = mid
    return high

samples = [1,1.05,1.10,1.25,1.5,2]
types = {"raw_throughput":False, "both_raw_throughput_and_aggregate_user_speed":True}
report = {name: {
    "samples": {str(factor):score(factor, both) for factor in samples},
    "uniform_16_and_32_factor_for": {str(goal):threshold(goal,both) for goal in [3,4]},
} for name,both in types.items()}
assert round(score(1,False),4) == baseline["official_score"]
files=[calibration,receipt,here.parents[2]/"tools/okc_node_loadgen.py"]
print(json.dumps(dict(official_V64=baseline["official_score"],scenarios=report,
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    interpretation="Hold the one- and four-user V64 rungs fixed. Multiply the whole 16- and 32-user rung measurements as specified and interpolate at the stock anchors. Factors apply to end-to-end serving metrics. They are illustrative score sensitivity, never the kernel tile ratio or a V297 prediction."),indent=2))
