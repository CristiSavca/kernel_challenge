"""Count actual waves for synthetic top-eight routes, sampling without replacement."""
import collections
import json

import numpy as np


def main():
    rng=np.random.default_rng(115)
    result=[]
    topk=8
    for tokens in (8192, 16384):
        cap=min(tokens, (384<<20)//(2*(4096+8192))-topk+1)
        limit=(tokens*topk+cap-1)//cap
        counts=collections.Counter()
        for _ in range(512):
            # Marginal local route count for one rank owning 32 of 256 experts.
            local=rng.hypergeometric(32, 224, topk, size=tokens)
            last_start=int(local[:-1].sum())
            counts[min(last_start//cap+1, limit)]+=1
        avg=sum(n*ct for n,ct in counts.items())/sum(counts.values())
        result.append(dict(tokens=tokens,rank_samples=sum(counts.values()),max_waves=limit,
            iterations=counts,average_waves=avg,old_five_kernel_bodies_per_62_layers=limit*5*62,
            average_five_kernel_body_calls_saved_per_chunk=round((limit-avg)*5*62,1)))
    print(json.dumps(dict(seed=115,assumption="Independent uniform top-eight choices per token",
                          synthetic_only=True,shapes=result),indent=2))


if __name__=="__main__":
    main()
