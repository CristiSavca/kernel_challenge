"""Pass native fallback operands through the real pinned FA3 provider wrappers."""
import itertools
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
donor=HERE.parent/"v239_split_mixed_full_attention/independent_fa3.py"
src=donor.read_text()
edits={
    "state = dict(batched_prompt_kv=(layout,tail),":
        "state = dict(verify_eligible=False, prompt_storage=None, batched_prompt_kv=(layout,tail),",
    "def pack(k, v, wk, wv, pt, lengths, cuq_head, received_layout):":
        "def pack(k, v, wk, wv, pt, lengths, cuq_head, received_layout, *, storage=None):\n        assert storage is None",
}
for a,b in edits.items():
    assert src.count(a)==1
    src=src.replace(a,b)
scope={"__file__":str(donor),"__name__":"legacy_supplier"}
stage=Path(sys.argv[1])
with patch.object(sys,"argv",[str(donor),str(stage)]):
    exec(compile(src,str(donor),"exec"),scope)
words=retries=unaliased=cases=0
for (prompts,tails),scale,fresh,past,knew in itertools.product(
        [([4096],[8]),([3300,902],[8,8,8]),([3999,91,130],[8,7])],
        ["none","expanded","ranked"],(False,True),(False,True),(False,True)):
    n,r,c=scope["run"](prompts,tails,scale,fresh,past,knew)
    words+=n;retries+=r;unaliased+=c;cases+=1
print(json.dumps(dict(cases=cases,fp8_attention_bf16_result_components=words,missing_out_keyword_retries=retries,
                      fresh_storage_copybacks=unaliased,production_sources={name:hashlib.sha256(text.encode()).hexdigest()
                          for name,text in scope["SOURCES"].items()},physical_H200=False),indent=2))
