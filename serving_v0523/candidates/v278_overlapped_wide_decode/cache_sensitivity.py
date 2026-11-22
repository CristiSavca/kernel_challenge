"""Evaluate request counts against alternative L2 capacities and CTA issue orders."""
from collections import Counter, OrderedDict
from pathlib import Path
import hashlib
import json

here = Path(__file__).resolve().parent
profile = here.parents[2]/"experiments/iterations/20261104-moe-hardware-profile-recovered.json"
geometry = here.parent/"v275_wide_decode_output_tiles/assembly.json"
variants = json.loads(geometry.read_bytes())["variants"]
case = next(c for c in json.loads(profile.read_bytes())["cases"] if c["M"] == 256)
counts = case["routing"]["local_counts"]


def simulate(phase, kind, capacity_mib, stages_at_a_time):
    m, n, k = variants[f"{phase}_24_{kind}"]["block"]
    base_n = variants[f"{phase}_24_native"]["block"][1]
    output = 8192 if phase == "w13" else 4096
    tiles = [(e, row, band) for e, rows in enumerate(counts) for row in range(0, rows, m) for band in range(output//n)]
    lru = OrderedDict()
    requested, missed = Counter(), Counter()
    size, capacity = 0, capacity_mib * 2**20

    def read(key, nbytes):
        nonlocal size
        group = key[0]
        requested[group] += nbytes
        if key in lru:
            lru.move_to_end(key)
            return
        missed[group] += nbytes
        if capacity < nbytes:
            return
        while size+nbytes > capacity:
            size -= lru.popitem(last=False)[1]
        lru[key] = nbytes
        size += nbytes

    for start in range(0, len(tiles), 132):
        for first_k in range(0, 4096//k, stages_at_a_time):
            for e, row, band in tiles[start:start+132]:
                for chunk in range(first_k, min(first_k+stages_at_a_time, 4096//k)):
                    read(("A", e, row, chunk), m*k*2)
                    for original_n in range(band*(n//base_n), (band+1)*(n//base_n)):
                        read(("bank", e, original_n, chunk), base_n*k*9//16)
    return dict(jobs=len(tiles), requested_MiB={k:v/2**20 for k,v in requested.items()},
                missed_MiB={k:v/2**20 for k,v in missed.items()})


rows = []
for capacity in (0,4,16,50):
    for step in (1,4,16,32):
        native, wide = {}, {}
        for projection, widened in (("w13","wide"),("w2","wide3")):
            native[projection]=simulate(projection,"native",capacity,step)
            wide[projection]=simulate(projection,widened,capacity,step)
        def total(s, field, group):
            return sum(r[field][group] for r in s.values())
        gains={group:total(native,"missed_MiB",group)-total(wide,"missed_MiB",group) for group in ("A","bank")}
        assert total(native,"requested_MiB","A")-total(wide,"requested_MiB","A")==198
        assert total(native,"requested_MiB","bank")==total(wide,"requested_MiB","bank")==891
        assert capacity or gains=={"A":198,"bank":0}
        rows.append(dict(capacity_MiB=capacity,K_chunks_per_CTA_visit=step,ordinary=native,wide=wide,
                         avoided_cache_miss_MiB=gains))
print(json.dumps(dict(source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (profile,geometry)},
    cases=rows,scope="One fully associative single-rank LRU for activation and packed bank slices, cleared per projection. "
    "N-fast persistent waves of 132 CTAs with simultaneous wave starts. Each row tile has a distinct activation key, "
    "ignoring shared edge rows and tensor-map OOB effects. No output traffic, set mapping, L1, cross-SM variation, "
    "tensor queue or timing. Synthetic issue orders deliberately vary K progress across CTAs. No H200 cache counters."),indent=2))
