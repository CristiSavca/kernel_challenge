"""Replay V235 sliding cases with the extra V239 role field carried by ForwardBatch."""
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
fixture=here.parent/"v235_peer_mixed_sliding_prefill/independent_causal_attention.py"
source=fixture.read_text()
before="    fb.mixed_verify_logits_indices=role.mixed_verify_logits_indices"
source=source.replace(before,"    fb.mixed_verify_prompt_count=getattr(role, 'mixed_verify_prompt_count', 0)\n"+before)
# V235's original one-call witness models sliding attention. V240's split full case
# runs separately against the integrated tree through the two numerically replayed providers.
full='("accepted",-1,False,4096),'
assert source.count(full)==1 and source.count(before)==1
source=source.replace(full,"")
exec(compile(source,str(fixture),"exec"),{"__file__":str(fixture)})
