"""Run the inherited target graph and joined host-cache copy tests on the V256 export."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_control(path, stage, function):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Both parent tests check the patch and tree in this directory.
    module.HERE = HERE
    module.torch.set_num_threads(1)
    return getattr(module, function)(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    for path, function, output in [
        ("v248_shared_verify_raster/independent_graph.py", "check", "target_graph.json"),
        ("v253_window_backed_cache_transfers/copy_replay.py", "main", "integrated_copy.json"),
    ]:
        result = run_control(HERE.parent / path, args.stage, function)
        (HERE / output).write_text(json.dumps(result, indent=2) + "\n")
