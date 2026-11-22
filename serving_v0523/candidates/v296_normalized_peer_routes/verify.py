"""Run the shared export verifier for the V296 candidate."""
from pathlib import Path
import runpy
import sys

here=Path(__file__).resolve().parent
if len(sys.argv)!=3:
    raise SystemExit("usage: verify.py CHECKOUT PUBLIC_RULES_JSON")
sys.argv.append(str(here))
runpy.run_path(str(here.parent/"v293_peer_prompt_norm/verify.py"),run_name="__main__")
