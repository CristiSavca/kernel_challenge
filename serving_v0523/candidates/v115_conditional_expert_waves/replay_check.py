"""Run the production dispatch and its captured loop body on a CPU graph executor."""
import argparse
import collections
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


DIR = Path(__file__).resolve().parent
BASE = runpy.run_path(str(DIR.parent / "v98_graph_grouped_prefill/cpu_check.py"))


class Graph:
    def __init__(self):
        self.recording = False
        self.iterations = None

    def get_currently_capturing_graph(self):
        return self

    def begin_capture_to_while_node(self, active):
        assert not self.recording
        self.recording = True
        self.active = active
        self.body = []

    def submit(self, fn, args, kwargs):
        if self.recording:
            self.body.append((fn, args, kwargs))
        else:
            fn(*args, **kwargs)

    def set_conditional_handle_for_current_node(self, active):
        assert self.recording and self.active is active
        self.submit(self.update, (active,), {})

    def update(self, active):
        self.again = bool(active)

    def end_capture_to_conditional_node(self):
        self.recording = False
        assert self.body[-1][0] == self.update and len(self.body) == 7
        self.again = bool(self.active)
        self.iterations = 0
        while self.again:
            self.iterations += 1
            assert self.iterations < 64
            for fn, args, kwargs in self.body:
                fn(*args, **kwargs)

    def call(self, fn):
        def wrapped(*args, **kwargs):
            return self.submit(fn, args, kwargs)
        return wrapped

    def kernel(self, fn):
        class Kernel:
            def __getitem__(_, grid):
                return self.call(fn[grid])
        return Kernel()


def main(tree):
    source = tree / "python/sglang/srt/layers/muse_spark/humming_prefill_waves.py"
    ops, _, gemms = BASE["make_ops"](tree)
    pid, (loader,) = BASE["program"](source, ("_load_wave",))
    graph = Graph()
    ops["_load_wave"] = graph.kernel(BASE["Launch"](loader, pid, (2, 3, 4, 5, 6)))
    ops["_pack_wave"] = graph.kernel(ops["_pack_wave"])
    ops["HummingMethod"] = SimpleNamespace(forward_layer=graph.call(ops["HummingMethod"].forward_layer))
    silu = SimpleNamespace(run=graph.call(ops["_expert_scaled_silu_module"]().run))
    ops["_expert_scaled_silu_module"] = lambda: silu
    ops["fused_post_expert_norm_weighted_sum"] = graph.call(ops["fused_post_expert_norm_weighted_sum"])
    rng = torch.Generator().manual_seed(115)
    def weights(shape):
        return torch.randint(-4, 5, shape, generator=rng).float() / 8

    cases = outputs = live_rows = 0
    counts = collections.Counter()
    for tokens, topk, experts, capacity in ((19, 4, 3, 5), (37, 8, 7, 25), (15, 3, 5, 16), (1, 2, 2, 3)):
        h, i = 16, 8
        hidden = weights((tokens, h)).bfloat16()
        scores = weights((tokens, topk))
        layer = SimpleNamespace(num_local_experts=experts, intermediate_size_per_partition=i,
            w13=weights((experts, 2*i, h)), w2=weights((experts, h, i)),
            w13_global_scale=torch.linspace(.5,1.25,experts).bfloat16(),
            w2_global_scale=torch.linspace(.625,1.5,experts).bfloat16(),
            _post_expert_norm_weight=weights((h,)).bfloat16(), _post_expert_norm_eps=1e-5,
            humming_grouped_tunings={"w13":"w13","w2":"w2"})
        for layout in ("all-local", "remote", "mixed", "boundary", "late", "duplicate"):
            ids = torch.randint(-experts, experts * 2, (tokens, topk), generator=rng, dtype=torch.int32)
            if layout == "all-local":
                ids.remainder_(experts)
            if layout == "remote":
                ids.fill_(-1)
            if layout == "boundary":
                ids.fill_(-1)
                ids.flatten()[:capacity] = 0
            if layout == "late":
                ids[:tokens//2] = -1
            if layout == "duplicate":
                ids.fill_(0)
            _, _, _, bounds = ops["plan_waves"](ids, experts, capacity)
            waves = len(bounds) - 1
            for condition, capture in ((False, False), (False, True), (True, False), (True, True)):
                layer.humming_conditional_waves = condition
                for first in (False, True):
                    layer._post_expert_norm_weight_first = first
                    gemms.clear()
                    graph.iterations = None
                    expected = BASE["reference"](layer, hidden, scores, ids)
                    with patch.object(torch.cuda,"is_current_stream_capturing",return_value=capture), \
                         patch.object(torch.cuda,"CUDAGraph",graph):
                        result = ops["grouped_prefill_waves"](layer,hidden,scores,ids,capacity)
                    assert torch.equal(result,expected), (tokens, layout, condition, capture, first)
                    assert gemms[::2] == [("w13", ct) for _, ct in gemms[1::2]]
                    valid = ((ids >= 0) & (ids < experts)).sum().item()
                    assert sum(ct for _, ct in gemms[::2]) == valid
                    bounded = condition and capture and waves >= 4
                    actual = int((bounds[:-1] < tokens).sum()) if bounded else waves
                    assert len(gemms) == 2*actual
                    assert graph.iterations == (actual if bounded else None)
                    if bounded:
                        counts[f"{actual}/{waves}"] += 1
                    cases += 1
                    outputs += expected.numel()
                    live_rows += valid
    print(json.dumps(dict(cpu_cases=cases, bf16_output_values=outputs, live_expert_rows=live_rows,
        executed_vs_max_waves=counts, real_cuda_graph=False),indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("checkout",type=Path)
    main(parser.parse_args().checkout)
