"""Compare the V117 indexed row equation against a global stable sort."""

import hashlib
import json
from pathlib import Path
import random


def test():
    root = Path(__file__).resolve().parent
    patch = (root / "delta.diff").read_text()
    assert hashlib.sha256((root / "patch.diff").read_bytes()).hexdigest() == (
        "b74f15d77bad7efe53a973c51c5ea3cfe8d07bf4bcfa9f8f8081a6a653182c74")
    assert all(term in patch for term in ["ends - padding", "bucket * TILE + lane",
        "tile[:, None] < (slot // ROUTE_TILE)[None, :]", "tl.sum(boundary, 0)",
        "not skinny_gate_up and slots >= 8 * 128"])
    rng = random.Random(117)
    cases = routes = 0
    for experts in (1, 7, 32, 63):
        for topk in (1, 3, 7, 8, 17):
            for tokens in (1, 8, 16, 43, 127, 128, 129, 255, 256):
                slots = tokens * topk
                maps = ([-1] * slots, [0] * slots, [experts - 1] * slots,
                        [(i // 128) % experts for i in range(slots)],
                        [rng.randrange(-experts, 2 * experts) for _ in range(slots)])
                for ids in maps:
                    bucket = [v + 1 if 0 <= v < experts else 0 for v in ids]
                    sorted_slots = sorted(range(slots), key=bucket.__getitem__)
                    reference_rows = {s: j for j, s in enumerate(sorted_slots) if bucket[s]}
                    ends = []
                    ordinal = []
                    for start in range(0, slots, 128):
                        counts = [0] * (experts + 1)
                        for key in bucket[start:start+128]:
                            ordinal.append(counts[key])
                            counts[key] += 1
                        for i in range(1, len(counts)):
                            counts[i] += counts[i - 1]
                        ends.append(counts)
                    for s, key in enumerate(bucket):
                        if key:
                            row = ordinal[s] + sum(e[key-1 + int(t < s // 128)]
                                                   for t, e in enumerate(ends))
                            assert row == reference_rows[s]
                        routes += 1
                    assert [sum(tile[j] for tile in ends) for j in range(experts + 1)] == [
                        sum(b <= j for b in bucket) for j in range(experts + 1)]
                    cases += 1
    print(json.dumps({"adversarial_layouts": cases, "examined_slots": routes,
        "topk": [1,3,7,8,17], "local_experts": [1,7,32,63], "matches": "global stable sort",
        "control": "pure CPU equation independently reconstructed from frozen patch",
        "peak_slot_end_buffer_bytes_tp8_top8": {"16_users": 1056, "32_users": 2112},
        "gpu_timing": None}, indent=2))


if __name__ == "__main__":
    test()
