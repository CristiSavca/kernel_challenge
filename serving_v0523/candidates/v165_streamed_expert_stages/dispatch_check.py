"""Reuse the V161 full setup/dispatch fixture with the two cached residency words."""
from pathlib import Path

prior = Path(__file__).resolve().parent.parent / "v161_tma_expert_channels/dispatch_check.py"
source = prior.read_text()
for part in ("maps.shape==(512,)", "cached=torch.zeros(512,"):
    assert source.count(part) == 1
    source = source.replace(part, part.replace("512", "520"))
exec(compile(source, str(prior), "exec"), {"__file__": str(prior), "__name__": "__main__"})
