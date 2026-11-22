"""Exercise the prepared bank descriptors and normal Muse medium-density dispatch."""
from pathlib import Path

here = Path(__file__).resolve().parent
fixture = here.parent / "v161_tma_expert_channels/dispatch_check.py"
source = fixture.read_text()
updates = {
    "maps.shape==(512,)": "maps.shape==(520,)",
    "cached=torch.zeros(512,": "cached=torch.zeros(520,",
    'if name=="tma":': 'if name=="medium-tma":\n            assert args[-1] is cached and args[-2].shape==(E,rows*top+1)\n        if name=="tma":',
    'NS(run=select("tma")),cached': 'NS(run=select("tma"),run_medium=select("medium-tma")),cached',
    'else "medium" if rows*top>256': 'else "medium-tma" if rows*top>256 and mode in ("normal","value-update") else "medium" if rows*top>256',
    '(0,1,8,32,33,128,256,257)': '(0,1,8,32,33,64,127,128,129,255,256,257)',
}
for previous, revised in updates.items():
    assert source.count(previous) == 1
    source = source.replace(previous, revised)
exec(compile(source, str(fixture), "exec"), {"__file__": str(fixture), "__name__": "__main__"})
