"""Run V248's target graph lifecycle checks against this integrated export."""
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
test = runpy.run_path(str(HERE.parent/"v248_shared_verify_raster/independent_graph.py"))["check"]
test.__globals__["HERE"] = HERE
print(json.dumps(test(Path(sys.argv[1]).resolve()),indent=2))
