"""Exercise the candidate's actual binding and normal eager/decode functions on CPU buffers."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

here = Path(__file__).resolve().parent
source = Path(sys.argv[1]) / "python/sglang/srt/layers/muse_spark/decode_gate_up.py"
runtime = {"torch": torch}
tree = ast.parse(source.read_bytes())
names = {"bind_prompt_wave", "_bound_gate_up", "run_gate_up"}
exec(compile(ast.Module([n for n in tree.body if getattr(n, "name", "") in names], []), str(source), "exec"), runtime)
bind = runtime["bind_prompt_wave"]
bank, scales = torch.empty(6, dtype=torch.int32), torch.empty(6, dtype=torch.uint8)
maps = torch.arange(256, dtype=torch.uint8)
addresses = tuple((t.data_ptr(), t.shape) for t in (bank, scales))
prepared, invoked = [], []
class Driver:
    accepted = True
    def prepare_wave(self, inputs, banks, target, tile):
        assert banks is maps and target.numel() == 384 and tile == 176
        target[:256].copy_(banks)
        prepared.append((inputs.data_ptr(), target.data_ptr()))
        return self.accepted

    def run_wave(self, inputs, output, ids, offsets, factor, cached, handle, shared, tile, sms):
        assert (inputs.data_ptr(), cached.data_ptr()) == prepared[-1]
        assert torch.equal(cached[:256], maps) and (handle, shared, tile, sms) == (99,128000,176,132)
        assert output.data_ptr() != inputs.data_ptr() and factor is layer.w13_global_scale
        invoked.append((ids.data_ptr(), offsets.data_ptr(), output.data_ptr()))
        return True

    def run_prompt(self, inputs, output, keys, offsets, factors, original, *args):
        assert original is maps and keys.dtype == torch.int32
        return True

    def run(self, inputs, output, keys, offsets, factors, original, *args):
        assert original is maps and keys is raw_ids
        return True

launch = Driver()
state = (launch, maps, addresses, {"tuned": (object(),99,128000,176)}, 132)
layer = SimpleNamespace(w13_weight=bank, w13_weight_scale=scales, w13_global_scale=torch.ones(2,dtype=torch.bfloat16),
    hidden_size=128, intermediate_size_per_partition=256, humming_prompt_gate_up=state,
    humming_decode_gate_up=state, humming_grouped_tunings={"w13": "tuned"})
arena = torch.empty(16 * (128+256), dtype=torch.bfloat16)
inputs, active = arena[:16*128].view(16,128), arena[16*128:].view(16,256)
raw_ids = torch.ones(16,dtype=torch.int64)
offsets = torch.tensor([14,14,16], dtype=torch.int32)
projection = bind(layer,inputs,active)
assert projection
for i in range(4):
    ids = torch.full((16,),i%2,dtype=torch.int32)
    offsets[1] = 14 + i%3
    projection(ids,offsets)
assert len(prepared) == 1 and len(invoked) == 4 and {p[2] for p in invoked} == {active.data_ptr()}
for name,replacement in (("w13_weight",bank.clone()),("w13_weight_scale",scales.clone()),("w13_weight",bank.view(3,2)),
                         ("humming_grouped_tunings",{"w13":"missing"}),("humming_prompt_gate_up",None),
                         ("intermediate_size_per_partition",64)):
    old = getattr(layer,name)
    setattr(layer,name,replacement)
    assert bind(layer,inputs,active) is None, name
    setattr(layer,name,old)
for x,y in ((inputs[:0],active[:0]),(inputs,active[:8]),(inputs.float(),active),(inputs,active.float()),
            (inputs,active.t().contiguous().t()),(inputs[:,:127],active)):
    assert bind(layer,x,y) is None
assert len(prepared) == 1
launch.accepted = False
assert bind(layer,inputs,active) is None and len(prepared) == 2
for prompt in (True,False):
    result = runtime["run_gate_up"](layer,inputs,raw_ids,offsets,"tuned",prompt=prompt)
    assert result.shape == active.shape and result.dtype == active.dtype
    assert runtime["run_gate_up"](layer,inputs,raw_ids,offsets,"missing",prompt=prompt) is None
print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    repeated_wave_launches=len(invoked),map_encodes_per_selected_graph=1,declines=13,
    ordinary_prompt_and_decode=True,scope="Actual Python bindings on CPU, native FFI launch modeled"), indent=2))
