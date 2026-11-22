"""Rerun the inherited grouped callback BF16/mapping oracle against the assembled candidate."""
from pathlib import Path
fixture=Path(__file__).resolve().parent.parent/"v270_wide_prompt_weight_reuse/callback.py"
exec(compile(fixture.read_text(),str(fixture),"exec"))
