"""Compare possible rank-reduction orders on the same local BF16 GEMM partials."""
import json
import torch


def tree_sum(tensors):
    while len(tensors) > 1:
        tensors = [a+b for a,b in zip(tensors[::2],tensors[1::2])]
    return tensors[0]


def run():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(150)
    totals = dict(values=0,local_partial_comparisons=0,local_partial_differences=0,
                  different_from_fp32_pair_tree=0,different_from_bf16_pair_tree=0)
    max_err = dict(ordered=0.0,fp32_tree=0.0,bf16_tree=0.0)
    for size in (4,8):
        for k in (96,1024):
            rows, channels = size*13, 73
            local, striped = [], []
            for _ in range(size):
                x = torch.randn(rows,k,generator=generator).bfloat16()
                w = torch.randn(channels,k,generator=generator).bfloat16()
                full = torch.nn.functional.linear(x,w)
                pieces = []
                for row_group in x.chunk(size,dim=0):
                    pieces.append(torch.cat([torch.nn.functional.linear(row_group,s)
                                             for s in w.split(31,dim=0)],dim=1))
                recomputed=torch.cat(pieces,dim=0)
                totals["local_partial_comparisons"]+=full.numel()
                totals["local_partial_differences"]+=int((full!=recomputed).sum())
                local.append(full)
                striped.append(recomputed)
            ordered = torch.zeros_like(striped[0],dtype=torch.float32)
            for value in striped:
                ordered+=value.float()
            ordered=ordered.bfloat16()
            fp32_tree=tree_sum([v.float() for v in local]).bfloat16()
            bf16_tree=tree_sum(local)
            exact=torch.stack(local).double().sum(0)
            totals["values"]+=ordered.numel()
            totals["different_from_fp32_pair_tree"]+=int((ordered!=fp32_tree).sum())
            totals["different_from_bf16_pair_tree"]+=int((ordered!=bf16_tree).sum())
            for name,output in (("ordered",ordered),("fp32_tree",fp32_tree),("bf16_tree",bf16_tree)):
                max_err[name]=max(max_err[name],float((output.double()-exact).abs().max()))
    return dict(torch=torch.__version__,**totals,max_abs_error_vs_sum_of_local_bf16=max_err,
                device="CPU",baseline_h200_collective_order="unknown")


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
