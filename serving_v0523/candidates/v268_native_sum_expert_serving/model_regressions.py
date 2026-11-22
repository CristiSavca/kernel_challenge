"""Replay native mixed/ordinary target metadata and terminal-model code on the integrated tree."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
torch.set_num_threads(2)
graph = runpy.run_path(str(HERE.parent/"v248_shared_verify_raster/independent_graph.py"))["check"]
graph.__globals__["HERE"] = HERE
graph_result = graph(STAGE)


def outside_model_setup(src):
    module = ast.parse(src)
    cls = next(n for n in module.body if getattr(n, "name", None) == "MuseSpark1xModel")
    cls.body = [n for n in cls.body if getattr(n, "name", None) != "__init__"]
    return ast.dump(module)


path = HERE.parent/"v257_final_residual_gate/model_entry.py"
src = path.read_text()
check = '    assert sources[MODEL].encode() == subprocess.check_output(["git","-C",stage,"show",manifest["parent_tree"]+":"+MODEL])'
assert src.count(check) == 1
# The one intended target-model change is the constructor's QKV admission call.
src = src.replace(check, '    assert outside_model_setup(sources[MODEL]) == outside_model_setup(subprocess.check_output(["git","-C",stage,"show",manifest["parent_tree"]+":"+MODEL]))')
ns = {"__name__":"v268_terminal_fixture", "__file__":str(path), "outside_model_setup":outside_model_setup}
exec(compile(src, str(path), "exec"), ns)
ns["HERE"] = HERE
result = io.StringIO()
with redirect_stdout(result):
    ns["run"](STAGE)
print(json.dumps(dict(target_graph=graph_result, terminal=json.loads(result.getvalue()),
                      only_constructor_changed_in_target_model=True, physical_H200=False),indent=2))
