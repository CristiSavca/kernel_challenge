"""Check nonzero endpoint scores and all adjacent margins through actual request ownership paths."""
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def check(root):
    test=Path(__file__).resolve().parents[1]/"v99_adaptive_draft_tree/lifecycle_cpu.py"
    sys.path.insert(0,str(test.parent))
    from lifecycle_cpu import load_methods
    runtime=load_methods(root)
    def ix(items):
        return torch.tensor(items,dtype=torch.int64)
    def draft(ids,state=None):
        result=runtime.make_draft_input_v2(bonus_tokens=ix(ids),new_seq_lens=ix(ids)+13)
        result.tree_policy_advantage=state
        return result
    def resolve(relay,slots):
        owner=draft(slots)
        owner.future_indices=ix(slots)
        relay._resolve_spec_extras(SimpleNamespace(spec_info=owner))
        return owner
    rng=torch.Generator().manual_seed(7023)
    results={"direct_mixed_merges":0,"shuffled_relay_resolves":0,"reused_slot_resets":0}
    for width in range(2,9):
        pool=SimpleNamespace(req_to_token=torch.zeros((12,2),dtype=torch.int64))
        algo=SimpleNamespace(is_ngram=lambda:False,is_some=lambda:True)
        relay=runtime.FutureMap("cpu",algo,pool)
        relay._maybe_init_forward_bufs=lambda _:None
        relay.publish_ready=SimpleNamespace(record=lambda:None)
        state=torch.randint(-16,17,(5,width),dtype=torch.int32,generator=rng)
        state[:,0]=torch.tensor([-width,width,0,-2,2])
        for left,right in itertools.product((True,False),repeat=2):
            a=draft([21,22],state[:2] if left else None)
            b=draft([23,24,25],state[2:] if right else None)
            a.merge_batch(b)
            perm=ix([4,1,3,0])
            a.filter_batch(perm,perm.tolist())
            if left or right:
                expected=state.clone()
                if not left: expected[:2]=0
                if not right: expected[2:]=0
                assert torch.equal(a.tree_policy_advantage,expected[perm])
            else:
                assert a.tree_policy_advantage is None
            results["direct_mixed_merges"]+=1
        slots=[10,1,7,4,3]
        expected=state.clone()
        payload=runtime.RelayPayload.from_draft_input(draft([101,102,103,104,105],state))
        relay.publish(ix(slots),payload.bonus_tokens)
        relay.stash(ix(slots),payload)
        state.fill_(51)
        for permutation in itertools.permutations(range(5)):
            handle=resolve(relay,[slots[i] for i in permutation])
            assert torch.equal(handle.tree_policy_advantage,expected[ix(permutation)])
            results["shuffled_relay_resolves"]+=1
        relay.stash(ix([slots[1]]),runtime.RelayPayload(bonus_tokens=ix([998])))
        expected[1]=0
        assert torch.equal(resolve(relay,slots).tree_policy_advantage,expected)
        results["reused_slot_resets"]+=1
    return results


if __name__=="__main__":
    print(json.dumps({"torch":torch.__version__,**check(Path(sys.argv[1]))},indent=2))
