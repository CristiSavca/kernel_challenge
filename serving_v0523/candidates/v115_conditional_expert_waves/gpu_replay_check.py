"""Replay one real CUDA WHILE graph across changing expert histograms on a CUDA host."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
from compile_decode_sm90 import load_jit_module


def run(source):
    device = torch.device("cuda", torch.cuda.current_device())
    experts, tokens, waves, rows = 32, 16384, 9, 16384
    with tempfile.TemporaryDirectory() as tmp:
        kernel = load_jit_module(source,Path(tmp))._load_wave
        data = [torch.zeros(n,device=device,dtype=torch.int32)
                for n in (waves*experts+1, waves+1, 1, experts+1, experts+1, 2)]
        edges, boundaries, cursor, bounds, offsets, interval = data
        active = torch.empty((),device=device,dtype=torch.bool)
        totals = torch.zeros(3,dtype=torch.int32,device=device)
        def load(first):
            kernel[(1,)](*data,active,TOKENS=tokens,ROWS=rows,EXPERTS=experts,WAVES=waves,
                          BLOCK=triton.next_power_of_2(experts+1),FIRST=first,num_warps=1)
        def body():
            totals[0].add_(interval[1]-interval[0])
            totals[1].add_(offsets[-1]-offsets[0])
            totals[2].add_(1)
            load(False)
        load(True)
        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            totals.zero_()
            load(True)
            current = torch.cuda.CUDAGraph.get_currently_capturing_graph()
            current.begin_capture_to_while_node(active)
            body()
            current.set_conditional_handle_for_current_node(active)
            current.end_capture_to_conditional_node()
        for n,live_groups in ((1,0),(9,9),(2,1),(5,4),(1,1),(9,8)):
            # The last active token interval may contain only remote routes.
            t = [tokens*i//n for i in range(n)] + [tokens]*(waves+1-n)
            counts = torch.zeros(waves,experts,dtype=torch.int32)
            counts[:live_groups] = 7
            e = torch.cat((torch.zeros(1,dtype=torch.int32),counts.flatten().cumsum(0,dtype=torch.int32)))
            boundaries.copy_(torch.tensor(t,dtype=torch.int32))
            edges.copy_(e)
            graph.replay()
            assert totals.tolist() == [tokens,int(counts.sum()),n]
        return dict(cuda=torch.version.cuda,torch=torch.__version__,device=torch.cuda.get_device_name(device),
                    replays=6,compiled_metadata_kernel=True,expert_gemms=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    print(json.dumps(run(p.parse_args().source),indent=2))
