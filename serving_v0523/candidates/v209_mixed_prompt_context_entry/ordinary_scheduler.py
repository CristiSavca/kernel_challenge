"""Exercise ordinary mixed cohorts and the native standalone-verifier fallback."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
suite = HERE.parent / "v202_mixed_target_serving/check_scheduler_rounds.py"
spec = importlib.util.spec_from_file_location("ordinary_fixture", suite)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bridge = module.test.bridge
bridge.env["dflash_sp_prefill_scope"] = module.runtime["dflash_sp_prefill_scope"]
rounds = []
with torch.inference_mode():
    for backend in ("FULL", "BREAKABLE"):
        for tails in (1, 2, 4):
            for index in range(8):
                for mirror in (True, False):
                    rounds.append(module.test.replay(backend,tails,index,mirror))
    with patch.object(torch.Tensor, "record_stream", lambda *args: None):
        ordinary = bridge.standalone_fallback()
print(json.dumps(dict(ordinary_mixed_rounds=len(rounds),committed=sum(x[2] for x in rounds),
                      standalone_sampled_selector_replays=ordinary,physical_H200=False),indent=2))
