"""Run V263's ordinary grouped serving entry and its C++ writer against the combined export."""
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
V263 = HERE.parent/"v263_token_major_expert_decode"
if __name__ == "__main__":
    suite = runpy.run_path(str(V263/"dispatch.py"))
    suite["main"].__globals__["HERE"] = HERE
    suite["main"]()
