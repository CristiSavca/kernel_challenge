"""Run the V255 caller suite on V256, including its retained V188 mode and V248 raster hook."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
prior=HERE.parent/"v255_private_draft_window_metadata"/"serving_replay.py"
src=prior.read_text()
seams={
    "    script = DONOR.read_text()": '''    script = DONOR.read_text().replace('methods(BACK,"FlashAttentionBackend",backend_names)',
        'methods(BACK,"FlashAttentionBackend",backend_names+" _refresh_verify_rasters")').replace(
        'kv_cache_is_mxfp8=False,', '_direct_fp8_prompt=False,kv_cache_is_mxfp8=False,')''',
    '(be, "use_sliding_window_kv_pool", False),': '',
    '        assert static(be, runner) == 2047\n    fixed': '''        assert static(be, runner) == 2047
        with patch.object(be, "use_sliding_window_kv_pool", False):
            assert static(be, runner) == 2047
    fixed''',
}
for old,new in seams.items():
    assert src.count(old)==1,old
    src=src.replace(old,new)
ns={"__name__":"linked_v255","__file__":str(prior)}
exec(compile(src,str(prior),"exec"),ns)
result=ns["main"](Path(sys.argv[1]).resolve(),HERE)
(HERE/"metadata_serving.json").write_text(json.dumps(result,indent=2)+"\n")
