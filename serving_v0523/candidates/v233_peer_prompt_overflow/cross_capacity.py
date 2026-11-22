"""Pass actual overflow resolver outputs through the model's peer workspace size call."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import subprocess


def main():
    stage, candidate = map(Path, sys.argv[1:])
    manifest = json.loads((candidate / "validation.json").read_bytes())
    grid = json.loads((candidate / "server_config.json").read_bytes())
    tree = manifest["tree"]

    def source(path):
        return subprocess.check_output(["git", "-C", stage, "show", tree + ":python/sglang/srt/" + path])

    model = ast.parse(source("models/muse_spark.py"))
    work = ast.parse(source("layers/muse_spark/peer_prompt_pack.py"))
    method, = [item for item in work.body if getattr(item, "name", "") == "peer_storage_rows"]
    sizes = {}
    exec(compile(ast.Module([method], []), "<workspace-size>", "exec"), sizes)
    constructor_call, = [node for node in ast.walk(model) if isinstance(node, ast.Call) and
                         isinstance(node.func, ast.Name) and node.func.id == "prepare_peer_prompt_inputs"]
    bind_prefill, = [n for n in ast.walk(model) if isinstance(n, ast.Assign) and
                     ast.unparse(n) == "prefill = get_exec().graph.cuda_graph_config.prefill"]
    probe = ast.Module([bind_prefill, ast.Assign([ast.Name("result", ast.Store())], constructor_call)], [])
    program = compile(ast.fix_missing_locations(probe), "<production-model-capacity-call>", "exec")
    outputs = []
    for case in grid["cases"]:
        # Additional mixed keys are appended later in runner initialization. This grid
        # intentionally has only the ordinary chunk at the time the model allocates.
        chunk_rows = 16384
        world = 8
        hidden = 4096
        opts = SimpleNamespace(bs=[chunk_rows], max_bs=case["configured_graph_max"])
        exec_config = SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(prefill=opts)))
        schedule = SimpleNamespace(chunked_prefill_size=chunk_rows, max_prefill_tokens=case["max_prefill_tokens"])
        def prepare(tp, width, topk, *args):
            assert width == hidden and topk == 8 and tp == world
            return sizes["peer_storage_rows"](width, tp, *args)
        env = dict(get_parallel=lambda: SimpleNamespace(tp_group=world), first=SimpleNamespace(
            expert_hidden_size=hidden, top_k=8), get_schedule=lambda: schedule,
            get_exec=lambda: exec_config, prepare_peer_prompt_inputs=prepare)
        exec(program, env)
        capacity = env["result"]
        assert 0 < capacity * hidden * 2 <= 17 * 1024**2
        keys = [chunk_rows + n for n in case["extras"]]
        assert all((key + world - 1)//world <= capacity for key in keys), (case["name"], keys, capacity)
        outputs.append(dict(case=case["name"], new_graphs=keys, local_rows=capacity, bytes=capacity*hidden*2))
    default = next(row for row in outputs if row["case"] == "intact_chunk_both")
    assert default["new_graphs"] == [16392,16400,16416,16448,16512,16640]
    assert default["local_rows"] == 2080 and default["bytes"] == 17039360
    print(json.dumps(dict(candidate=manifest["candidate"], tree=tree, patch_sha256=manifest["patch_sha256"],
        resolver_sha256=hashlib.sha256((candidate/"server_config.json").read_bytes()).hexdigest(),
        constructor=ast.unparse(constructor_call),
        cases=outputs, physical_gpu=False),indent=2))


if __name__ == "__main__":
    main()
