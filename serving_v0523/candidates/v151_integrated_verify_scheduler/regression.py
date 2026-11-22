"""Exercise V147's baseline and V132's marked metadata path in the V151 sources."""
from pathlib import Path
import json
import runpy
import sys

HERE=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
PARENT="cd089ab0b54e9fa2ee4c1cfdcb3f354370748d85"
v132=runpy.run_path(str(HERE.parent/"v132_verify_scheduler_metadata/verify_dispatch.py"))["main"]
v132.__globals__["PARENT"]=PARENT
suite=v132(stage)
parent_modes=runpy.run_path(str(HERE.parent/"v147_integrated_experts_prefill/independent_modes.py"))["check"](stage)
phase=runpy.run_path(str(HERE.parent/"v147_integrated_experts_prefill/phase_check.py"))["test"](stage)
print(json.dumps(dict(v147_v151_backend_pairs=suite,disabled_modes=parent_modes,disabled_writer=phase),indent=2))
