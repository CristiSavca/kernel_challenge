"""Exercise the integrated serving model, grouped expert entry and mixed forward transitions."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import runpy
import sys

import torch

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()


def function(component, script, name):
    folder, = HERE.parent.glob(f"v{component}_*")
    fn = runpy.run_path(str(folder/script))[name]
    fn.__globals__["HERE"] = HERE
    return fn


if __name__ == "__main__":
    torch.set_num_threads(2)
    if sys.argv[2] == "transition":
        sys.argv = [sys.argv[0], str(STAGE), str(HERE)]
        function(267, "expert_transition.py", "main")()
    elif sys.argv[2] == "model":
        graph = function(248,"independent_graph.py","check")(STAGE)
        text = StringIO()
        with redirect_stdout(text):
            function(257,"model_entry.py","run")(STAGE)
        print(json.dumps(dict(target_graph=graph, terminal=json.loads(text.getvalue())),indent=2))
    elif sys.argv[2] == "decode":
        text = StringIO()
        with redirect_stdout(text):
            function(263,"dispatch.py","main")()
        result = dict(decode=json.loads(text.getvalue()), writer=function(263,"independent_token_writer.py","main")())
        print(json.dumps(result,indent=2))
    else:
        raise ValueError("expected transition, model or decode")
