"""Compile and exercise the integrated W13 CUDA header with the V263 coordinate/BF16 shim."""
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
v263 = HERE.parent/"v263_token_major_expert_decode"
if __name__ == "__main__":
    fn = runpy.run_path(str(v263/"independent_token_writer.py"))["main"]
    fn.__globals__["HERE"] = HERE
    print(json.dumps(fn(),indent=2))
