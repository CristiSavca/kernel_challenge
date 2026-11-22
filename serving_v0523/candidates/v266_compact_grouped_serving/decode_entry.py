"""Run the independent V261 grouped-forward suite against the combined source."""
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
suite = runpy.run_path(str(HERE.parent / "v261_fused_decode_route_staging/independent_plan.py"))
suite["main"].__globals__["HERE"] = HERE
suite["main"]()
