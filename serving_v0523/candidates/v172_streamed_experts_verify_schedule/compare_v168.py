"""Run the paired FA3 call and mutable-length comparison on the actual V168 tree."""
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
compare = runpy.run_path(str(HERE.parent/"v151_integrated_verify_scheduler/independent_modes.py"))["run"]
# The inherited fixture otherwise compares its original V147 parent.
compare.__globals__["PARENT"] = "127fafbf1bbaf01796c8c9c18b37c607de47d76e"
print(json.dumps(compare(Path(sys.argv[1]).resolve()),indent=2))
