"""Compare V261 and the original two-kernel decode staging on one local H200."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
LOCAL_EXPERTS, GLOBAL_EXPERTS, TOPK, WIDTH = 32, 256, 8, 4096
PATTERNS = ("uniform", "local", "remote", "hot")


def fixtures(rows, device):
    generator = torch.Generator().manual_seed(26100 + rows)
    hidden = torch.randn((rows, WIDTH), dtype=torch.bfloat16, generator=generator)
    # A non-monotonic local map also exercises stable ordering after remapping.
    mapping = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32)
    owner = torch.randperm(GLOBAL_EXPERTS, generator=generator)[:LOCAL_EXPERTS]
    mapping[owner] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32)
    remote = torch.nonzero(mapping < 0).flatten()
    domain = {"uniform": torch.arange(GLOBAL_EXPERTS), "local": owner, "remote": remote,
              "hot": torch.cat((owner[:6], remote[:5]))}
    hidden, device_map = hidden.to(device), mapping.to(device)
    result = []
    for name, ids in domain.items():
        index = torch.rand((rows, len(ids)), generator=generator).topk(TOPK).indices
        routes = ids[index].to(torch.int32)
        assert all(len(torch.unique(row)) == TOPK for row in routes)
        live = int((mapping[routes.long()] >= 0).sum())
        result.append((name, routes.to(device), hidden, device_map, live))
    return result


def compare_order(repetitions):
    return [("parent", "candidate") if step % 2 == 0 else ("candidate", "parent")
            for step in range(repetitions)]


def describe(pair_us):
    ratios = [p / c for p, c in pair_us]
    deltas = [p - c for p, c in pair_us]
    return dict(parent_median_us=statistics.median(p for p, _ in pair_us),
                candidate_median_us=statistics.median(c for _, c in pair_us),
                median_paired_speedup=statistics.median(ratios),
                median_paired_saving_us=statistics.median(deltas), paired_samples_us=pair_us)


def buffer(slots, device):
    def integers(n):
        return torch.full((n,), -991, dtype=torch.int32, device=device)
    return (integers(slots), integers(LOCAL_EXPERTS + 1), integers(slots), integers(slots),
            torch.full((slots, WIDTH), 77, dtype=torch.bfloat16, device=device))


def run_shape(module, rows, repeats, samples):
    slots = rows * TOPK
    span = 1 << (slots - 1).bit_length()
    fused_warps = 8 if span == 256 and WIDTH > 2048 else 4
    device = torch.device("cuda", torch.cuda.current_device())
    data = fixtures(rows, device)
    a, b = buffer(slots, device), buffer(slots, device)
    common = dict(EXPERTS=LOCAL_EXPERTS, GLOBAL_EXPERTS=GLOBAL_EXPERTS, HAS_EXPERT_MAP=True,
                  SLOT_SPAN=span, EXPERT_SPAN=64)
    layout = dict(TOPK=TOPK, HIDDEN=WIDTH, BLOCK=WIDTH)

    def launch(candidate, dst, sample):
        _, ids, hidden, mapping, _ = sample
        local, offsets, keys, inverse, staged = dst
        if candidate:
            module._stage_grouped_decode[(slots + 1,)](
                ids, mapping, local, offsets, keys, inverse, hidden, staged, slots,
                **common, **layout, num_warps=fused_warps)
        else:
            module._plan_grouped_decode[(1,)](
                ids, mapping, local, offsets, keys, inverse, slots, **common, num_warps=4)
            module._gather_grouped_decode[(min(slots, 128),)](
                hidden, keys, offsets, staged, slots, SLOT_SPAN=span, **layout, num_warps=8)

    for sample in data:
        for dst, poison in ((a, -991), (b, -992)):
            for tensor in dst[:-1]:
                tensor.fill_(poison)
            dst[-1].fill_(77)
        launch(False, a, sample)
        launch(True, b, sample)
        for control, actual in zip(a, b):
            assert torch.equal(control, actual), (rows, sample[0])
    # Use separate destinations. Each graph cycles through the same four ID tables.
    graphs = {}
    torch.cuda.synchronize()
    for name, candidate, dst in (("parent", False, a), ("candidate", True, b)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                for sample in data:
                    launch(candidate, dst, sample)
        graphs[name] = graph
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for order in compare_order(samples):
        pair = {}
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            pair[name] = start.elapsed_time(end) * 1000 / (repeats * len(data))
        times.append([pair["parent"], pair["candidate"]])
    return dict(rows=rows, slots=slots, fused_warps=fused_warps, calls_per_replay=repeats * len(data),
                live_routes={x[0]: x[-1] for x in data}, **describe(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-check", action="store_true", help="verify inputs and paired analysis on CPU")
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 9, 16, 32])
    parser.add_argument("--repeats", type=int, default=8, help="four-route cycles in one CUDA graph")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if not (0 < args.repeats <= 256 and 2 <= args.samples <= 128 and all(0 < r <= 32 for r in args.rows)):
        parser.error("use 1–32 rows, 1–256 repeats and 2–128 paired samples")
    if args.fixture_check:
        assert describe([(8., 4.), (6., 3.)])["median_paired_saving_us"] == 3.5
        assert compare_order(2) == [("parent", "candidate"), ("candidate", "parent")]
        cases = {}
        for rows in args.rows:
            cases[rows] = {p: live for p, _, _, _, live in fixtures(rows, "cpu")}
            assert cases[rows]["local"] == rows * TOPK and cases[rows]["remote"] == 0
        return dict(device="CPU", live_routes=cases, paired_summary_checked=True, hardware_timing=False)

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name():
        parser.error("physical H200 required for timings")
    from sglang.srt.layers.muse_spark import muse_spark_humming_moe as module
    source = Path(module.__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != json.loads((HERE / "compiler.json").read_bytes())["source_sha256"]:
        parser.error("import the applied V261 source when running the comparison")
    return dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                source_sha256=digest, rows=[
                    run_shape(module, n, args.repeats, args.samples) for n in args.rows
                ], timing="GPU events around repeated captured route planning and staging. Synthetic inputs are reused; timing excludes model GEMMs, TP communication, allocation, and the rest of serving.")


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
