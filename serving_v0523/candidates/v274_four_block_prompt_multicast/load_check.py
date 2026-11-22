"""Reuse parent production ISA tests for both separately rendered multicast images."""
from pathlib import Path
here=Path(__file__).resolve().parent
parent=here.parent/"v270_wide_prompt_weight_reuse/load_check.py"
src=parent.read_text().replace('here/"candidate.py"','here/"render.py"').replace('role=="WidePromptWeightPipeline"','role.startswith("WidePromptWeightPipeline")').replace('ns["compiler"]','ns["ns"]["compiler"]').replace('ns["HummingKernel"]','ns["ns"]["HummingKernel"]')
exec(compile(src,str(parent),"exec"))
