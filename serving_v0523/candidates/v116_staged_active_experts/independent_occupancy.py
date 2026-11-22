"""Exact finite-population occupancy with uniform top-k routing, including all EP ranks."""

import json
from math import ceil, comb, sqrt


def new_local_counts(local, global_experts, topk, tokens):
    # With u marked local experts already seen, a token adds j of the
    # remaining local-u. All remaining topk-j picks can come from the
    # already seen local experts or any nonlocal expert.
    probabilities = [1.0]+[0.0]*local
    denominator=comb(global_experts,topk)
    transitions=[[comb(local-u,j)*comb(global_experts-local+u,topk-j)/denominator
                  for j in range(min(topk,local-u)+1)] for u in range(local+1)]
    for _ in range(tokens):
        new=[0.0]*(local+1)
        for already,prob in enumerate(probabilities):
            if prob:
                for extra,p in enumerate(transitions[already]):
                    new[already+extra]+=prob*p
        probabilities=new
    assert abs(sum(probabilities)-1)<1e-12
    return probabilities


def mul(a,b):
    c=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            c[i+j]+=x*y
    return c


def powers(a,max_power):
    store=[[1]]
    for _ in range(max_power):
        store.append(mul(store[-1],a))
    return store


def estimate(tokens):
    global_experts,local,k=256,32,8
    ranks=global_experts//local
    p=1-(1-k/global_experts)**tokens
    groups=min(local,ceil(local*p+sqrt(local*p*(1-p))))
    one=new_local_counts(local,global_experts,k,tokens)
    total=new_local_counts(global_experts,global_experts,k,tokens)
    # Conditional on the total number D of unique experts, all D-subsets
    # of the 256 experts are equiprobable by permutation symmetry. The
    # coefficient of z^D counts subsets by their occupancy on each rank.
    ok=powers([comb(local,i) if i<=groups else 0 for i in range(local+1)],ranks)
    spill=powers([comb(local,i) if i>groups else 0 for i in range(local+1)],ranks)
    joint=[]
    for r in range(ranks+1):
        choices=mul(ok[ranks-r],spill[r])
        joint.append(sum(mass*comb(ranks,r)*choices[d]/comb(global_experts,d) for d,mass in enumerate(total)))
    assert abs(sum(joint)-1.0)<1e-12
    p_spill=sum(one[groups+1:])
    assert abs(sum(i*q for i,q in enumerate(joint))-ranks*p_spill)<1e-12
    mu=sum(u*q for u,q in enumerate(one))
    excess=sum(max(0,u-groups)*q for u,q in enumerate(one))
    return {"tokens":tokens,"groups":groups,"mean_local_active":mu,
            "sd_local_active":sqrt(sum((u-mu)**2*q for u,q in enumerate(one))),
            "probability_rank_spills":p_spill,"probability_any_of_eight_ranks_spills":1-joint[0],
            "expected_excess_experts_per_rank":excess,
            "additional_experts_given_a_rank_spills":excess/p_spill if p_spill else 0,
            "expected_additional_experts_per_launched_group":excess/groups,
            "rank_spill_count_distribution":joint}


if __name__=="__main__":
    print(json.dumps({"model":"Independent token rows; each chooses 8 distinct experts uniformly from 256. This computes exact subset counts and probabilities to floating-point precision. No benchmark or training routes used.",
        "cases":[estimate(i) for i in (8,16,32)],"adversarial_four_request_cases":{
            "one_expert_per_rank_rotating_across_32_tokens":{"active_on_every_rank":32,"groups":24,"extra_iterations_per_rank":8},
            "identical_eight_experts_for_all_tokens":{"at_most_active_on_any_rank":8,"groups":24,"extra_iterations_per_rank":0}}},
        indent=2))
