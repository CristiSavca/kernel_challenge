"""Compare the assembled V270 grouped callback with and without its down projection adapter."""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch
import torch

here = Path(__file__).resolve().parent
fixture = here.parent / "v221_fused_prompt_route_pack/check_dispatch.py"
text = fixture.read_text()
# The shared oracle uses real routing, per-expert activations, BF16 scaling and a CPU finalizer.
base = text[:text.index("groups=[]")]
base = base.replace("manifest=json.loads((subject/'validation.json' if subject.is_dir() else subject).read_bytes())", "")
ns = dict(__name__="v270_host", __file__=str(fixture))
exec(compile(base, str(fixture), "exec"), ns)
old = ns["host"]["env"]
host = ns["host"]
stage = Path(sys.argv[1])
env = dict(old, triton=ns["NS"](next_power_of_2=lambda n: 1 << (n-1).bit_length(), cdiv=lambda n,d:(n+d-1)//d),
           _pack_prompt_routes=ns["Pack"](), humming_expert_outputs=ns["indexed"])
model = stage / ns["base"]["hum"]
for name in ("prompt_route_order", "humming_grouped_expert_outputs"):
    host["add"](host["fn"](model, name), env)
group = env["humming_grouped_expert_outputs"]
layer = ns["layer"]
controls = []
with torch.inference_mode(), patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
    for rows in (8192, 8201, 16384):
        for scale_deferred in (True,False):
            ids = torch.full((rows,8), -1, dtype=torch.int32)
            ids[::17,2] = 0
            ids[::31,6] = 6
            ids[::31,3] = 6
            x = host["rand"](rows,128)
            for gate in ("disabled","paired"):
                layer.humming_prompt_gate_up = host["pstate"](gate)
                outputs = []
                for use_down in (False, True):
                    layer.humming_prompt_down = ns["f"]["bind"]() if use_down else None
                    ns["f"]["events"].clear()
                    values, mapping = group(layer,x,ids,defer_output_scale=scale_deferred)
                    assert ns["f"]["events"][-1][0] == ("W2 custom" if use_down else "w2")
                    outputs.append((values.clone(),mapping.clone()))
                assert torch.equal(outputs[0][0].view(torch.int16), outputs[1][0].view(torch.int16))
                assert torch.equal(outputs[0][1], outputs[1][1])
                controls.append(dict(rows=rows,gate=gate,defer_scale=scale_deferred,
                                     output_words=values.numel(),map_entries=mapping.numel()))
print(json.dumps(dict(cases=controls,compared_bf16=sum(c["output_words"] for c in controls),
                      compared_map_entries=sum(c["map_entries"] for c in controls),gpu=False,
                      patch_sha256=hashlib.sha256((here/"patch.diff").read_bytes()).hexdigest()), indent=2))

