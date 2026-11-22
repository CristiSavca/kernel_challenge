"""Replay original adverse and exact combined projected draft stores using the V253 owner fixture."""
import importlib.util
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
driver=here.parent/'v253_window_backed_cache_transfers/broad_writer_check.py'
spec=importlib.util.spec_from_file_location('broad_writer_check',driver)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module.run(Path(sys.argv[1]).resolve()),indent=2))
