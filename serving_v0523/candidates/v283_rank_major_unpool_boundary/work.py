"""Measure the storage behavior of GroupCoordinator's post-gather reshape on a CPU receive buffer."""
import hashlib
import json
from pathlib import Path
import sys

import torch

stage = Path(sys.argv[1])
here = Path(__file__).resolve().parent
source = (stage/"python/sglang/srt/distributed/parallel_state.py").read_text()
start = source.index("        # Reshape\n", source.index("    def all_gather("))
end = source.index("        return output_tensor", start) + len("        return output_tensor")
# Indent the extracted method tail as a function of its original inputs.
from textwrap import dedent, indent
src = "def finish(output_tensor, world_size, input_size, dim):\n" + indent(dedent(source[start:end]),"    ")
ns = {}
exec(src,ns)
finish = ns["finish"]
work = json.loads((here.parent/"v228_sharded_verify_projections/work.json").read_bytes())
width = 8192
layers = work["sparse_layers"] - 1
cases = []
for rung in work["rungs"]:
    m = rung["rows"]
    for world in (4,8):
        width_local = width//world
        storage = torch.arange(world*m*width_local,dtype=torch.int32).to(torch.bfloat16)
        received = storage.view(world*m,width_local)
        a, b = finish(received,world,(m,width_local),1), finish(received,world,(m,width_local),0)
        a_copy = (a.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr())
        b_alias = (b.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr())
        assert a_copy and b_alias
        raw = b.view(world,m,width_local)
        assert torch.equal(raw.transpose(0,1).reshape(m,width),a)
        bytes_out = m*width*storage.element_size()
        # V229's pre-existing Hopper multicast handles its eight-row unpooler when eligible.
        use = m > 8
        cases.append(dict(users=rung["users"],rows=m,world=world,
            original_allocates_new_storage=a_copy,raw_aliases_receive=b_alias,
            copy_read_write_bytes_per_eligible_boundary=2*bytes_out,
            copy_payload_bytes_per_eligible_boundary=bytes_out,
            modeled_eligible_boundaries=layers if use else 0,
            modeled_copy_launches_avoided=layers if use else 0,
            modeled_copy_read_write_MiB_per_rank_pass=2*bytes_out*layers/2**20 if use else 0))
print(json.dumps(dict(hidden_width=width,sparse_layers=layers+1,bound_layers=layers,torch=torch.__version__,
    source_tail_sha256=hashlib.sha256(source[start:end].encode()).hexdigest(),cases=cases,
    assumptions="Representative equal-length eight-position target verification batches. Native eight-row multicast is available. Remaining 61 nonterminal layer boundaries select the single-CTA JIT and column gather at larger rows.",
    scope="Logical reads and writes in the removed reorder, excluding gather, normalization, cache effects and allocator reuse. No HBM-traffic, device-peak or latency measurement."),indent=2))
