"""Audit V219's actual BCG allocation, captured source geometry, and public tap type."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent / "v219_verify_graph_sliding_tiles"
util_spec = importlib.util.spec_from_file_location("v217_source", HERE.parent / "v217_captured_verify_context/independent_model_entry.py")
util = importlib.util.module_from_spec(util_spec)
util_spec.loader.exec_module(util)
util.PARENT = (PARENT / "patch.diff").read_text()


def source(path):
    return ast.parse(util.read_source(path, False))


class PPProxyTensors:
    pass


def node(tree, name):
    return next(n for n in tree.body if getattr(n, "name", None) == name)


def method(tree, cls, name):
    return node(node(tree, cls), name)


def counts(tree):
    if torch.is_tensor(tree):
        return tree.numel() * tree.element_size()
    return sum(counts(item) for item in tree)


def flatten(tree):
    if torch.is_tensor(tree):
        return [tree]
    return [t for value in tree for t in flatten(value)]


def main():
    torch.set_num_threads(1)
    runner_tree = source("model_executor/runner/prefill_cuda_graph_runner.py")
    base_tree = source("model_executor/runner/base_cuda_graph_runner.py")
    backend_tree = source("model_executor/runner_backend/breakable_cuda_graph_backend.py")
    model_tree = source("models/muse_spark.py")
    omni_tree = source("models/muse_spark_omni.py")
    assert "MuseSparkForConditionalGeneration" in ast.unparse(omni_tree.body[-1])
    names = {n.name for n in node(runner_tree, "PrefillCudaGraphRunner").body if isinstance(n, ast.FunctionDef)}
    assert not {"cuda_graph_output_capacity_rows", "cuda_graph_output_rows"} & names
    resolve_model = util.bind(source("model_loader/utils.py"), None, "resolve_language_model", {"nn":nn})
    resolve_capture = util.bind(runner_tree, None, "_resolve_transformer_layer_model", {"nn":nn,"resolve_language_model":resolve_model})

    class MuseSpark1xModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()

    class MuseSparkModel(nn.Module):
        forward = util.bind(model_tree, "MuseSparkModel", "forward", {"nn":nn})
        def __init__(self):
            super().__init__()
            self.language_model = MuseSpark1xModel()
        @property
        def layers(self):
            return self.language_model.layers

    class MuseSparkForConditionalGeneration(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MuseSparkModel()

    target = MuseSparkForConditionalGeneration()
    captured = resolve_capture(target)
    assert captured is target.model and captured.language_model is target.model.language_model
    body = method(runner_tree, "PrefillCudaGraphRunner", "_run_forward")
    assert "self.layer_model.forward(" in ast.unparse(body)

    bases = {n: util.bind(base_tree, "BaseCudaGraphRunner", n, {}) for n in ("cuda_graph_output_rows","cuda_graph_output_capacity_rows")}
    runner = type("Runner", (), bases)()
    backend = type("Backend", (), {n:util.bind(backend_tree, "BreakableCudaGraphBackend", n, {"torch":torch, "PPProxyTensors":PPProxyTensors}) for n in
                                   ("_alloc_full_buffer","_output_rows","_slice_output")})()
    backend._cuda_graph_runner = runner

    log = ROOT/"experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__server.log"
    trace = log.read_text()
    m = re.search(r"Capture target prefill CUDA graph begin. backend=(\w+), num_tokens=(\[[^\]]+\])",trace)
    keys = ast.literal_eval(m.group(2))
    assert m.group(1)=="breakable" and len(keys)==74 and max(keys)==16384
    hist = [int(n) for n in re.findall(r"Prefill batch[^\n]+?#new-token: (\d+)[^\n]+?cuda graph: True",trace)]
    official = json.loads((ROOT/"serving_v0523/runs/04b01f958aa5487c.json").read_text())["result"]
    logclip = (ROOT/"serving_v0523/runs/04b01f958aa5487c.log").read_text()
    assert "chunked_prefill_size=16384" in logclip and "MuseSparkForConditionalGeneration" in logclip
    width, tp = 8192,8
    def output(rows):
        return torch.empty((rows,width),device="meta",dtype=torch.bfloat16), [
            torch.empty((rows,width), device="meta", dtype=torch.bfloat16) for _ in range(5)]
    first = max(keys)
    rows = (first+tp-1)//tp
    warmup = output(rows)
    assert runner.cuda_graph_output_capacity_rows(warmup) is None
    # Exact allocation branch and production _output_rows/_slice_output.
    buf = backend._alloc_full_buffer(warmup,first)
    local = backend._alloc_full_buffer(warmup,rows)
    for key in keys:
        k_rows = (key+tp-1)//tp
        produced = backend._output_rows(output(k_rows),key)
        assert produced==k_rows
        sliced = backend._slice_output(buf,produced)
        assert all(t.shape==(k_rows,width) for t in flatten(sliced))
    assert counts(buf) == 1536*2**20 and counts(local)==192*2**20

    # A shape-sensitive forward hook attached after capture observes the wrapper's
    # return during the eager tail as _execute_with_patched_layer installs replay.
    rows, h, ntaps = 3,4,5
    geometry = []
    def observe(module,args,output):
        geometry.append((type(output[1]).__name__,len(output[1]),list(output[1][0].shape)))
        return output
    handle = captured.register_forward_hook(observe)
    t = torch.arange(rows*h,dtype=torch.float32).reshape(rows,h)
    taps = [t + i for i in range(ntaps)]
    try:
        for candidate in (taps,torch.cat(taps,-1)):
            captured.forward = lambda *a:(t,candidate)
            captured(None,None,None)
    finally:
        del captured.forward
        handle.remove()
    assert geometry==[("list",5,[3,4]),("Tensor",3,[20])]
    stats = {"candidate_parent_sha256":hashlib.sha256(util.PARENT.encode()).hexdigest(),
             "source_support":{"resolve_capture":type(captured).__name__,"producer":type(captured.language_model).__name__,
                               "registered":type(target).__name__,"eager_replay_hook_witness":geometry},
             "archive":{"trace":str(log.relative_to(ROOT)), "raw_bytes_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),
                        "universal_newline_text_sha256":hashlib.sha256(trace.encode()).hexdigest(),
                        "backend":m.group(1),"shapes":keys,"graph_prefills":len(hist),"full_chunk_graph_prefills":hist.count(first),
                        "v68_chunk_tokens":16384,"v68_official_peak_over_stock_gb":official["gpu_memory"]["over_stock_gb"]},
             "geometry":{"tp":tp,"width":width,"taps":ntaps,"max_global_rows":first,"max_local_rows":(first+tp-1)//tp,
                         "legacy_allocated_bytes":counts(buf),"six_local_leaves_bytes":counts(local),
                         "avoidably_reserved_bytes":counts(buf)-counts(local),"padded_shape_keys":[n for n in keys if n%tp]},
             "hardware":"PyTorch CPU hooks and meta tensor geometry; V219 source. No CUDA launch or physical peak measurement."}
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
