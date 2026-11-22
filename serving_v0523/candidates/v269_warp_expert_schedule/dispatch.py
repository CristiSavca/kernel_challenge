"""Exercise the existing public model-routing fixture against the V269 patch."""
from pathlib import Path
import runpy

here = Path(__file__).resolve().parent
check = runpy.run_path(str(here.parent/'v263_token_major_expert_decode/dispatch.py'))['main']
check.__globals__['HERE'] = here
check()
