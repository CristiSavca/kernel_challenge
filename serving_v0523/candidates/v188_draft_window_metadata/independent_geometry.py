"""Replay pinned SM90 FA4 split and mask bounds before and after a tile-aligned suffix."""
import ast
from collections import Counter, defaultdict, namedtuple
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import subprocess
from types import SimpleNamespace as N

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT / "serving_v0523/sglang"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
CUTE = "python/sglang/kernels/ops/attention/flash_attn/cute/"
W, Q, RATIO, SMS = 2047, 8, 8, 132
hashes = {}
env = dict(FwdConfig=namedtuple("FwdConfig", "m n rs ol"), Int32=int,
           cute=N(ceil_div=lambda a,b: (a+b-1)//b), const_expr=bool,
           cutlass=N(min=min, max=max, const_expr=bool))


def methods(path, names, owner=None):
    data = subprocess.check_output(["git", "-C", str(REPO), "show", PIN + ":" + path])
    hashes[path] = sha256(data).hexdigest()
    nodes = ast.parse(data).body
    if owner:
        nodes = next(n for n in nodes if getattr(n, "name", None) == owner).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    for n in selected:
        n.decorator_list = []
    exec("from __future__ import annotations\n" + ast.unparse(ast.Module(selected, [])), env)
    return data.decode()


interface = methods(CUTE+"interface.py", ["_tile_size_fwd_sm90", "num_splits_heuristic"])
methods(CUTE+"block_info.py", ["get_n_block_min_max", "get_n_idx_left_right"], "BlockInfo")
assert "seqlen_k = num_pages * page_size" in interface and "if arch // 10 == 9 and max_seqlen_q == 1:" in interface
m, n, _, _ = env["_tile_size_fwd_sm90"](128, 128, False, True)
assert (m, n) == (128, 128)
loaded = 2*W + 1 + m
split_counts = {b:env["num_splits_heuristic"](b, SMS, (loaded+n-1)//n, 128) for b in (1,4,16,32)}
assert all(0 < s <= 128 for s in split_counts.values())
archive = ROOT / "experiments/modal_node/20261113T204343Z-bench"
client = next(archive.glob("*v41__load.json"))
log = next(archive.glob("*v41__server.log"))
assert "attention_backend=fa4, model=DFlash2DraftModel, block_size=8, draft_window_size=None, compact_cache=False" in log.read_text()
requests = json.loads(client.read_bytes())["requests"]
by_load = defaultdict(list)
for req in requests:
    by_load[req["load"]].append(req["prompt"])
positions = {item["prompt"] for item in requests}
# Cover first/last physical tile edges, every alignment residue, short paths and the legal context end.
positions.update(range(2305))
positions.update(range(131072-Q-255, 131072-Q+1))
calls = Counter()
for p in sorted(positions):
    start = max(0,p-W)//n*n
    full, compact = (N(seqlen_q=Q, seqlen_k=k) for k in (p+Q,p+Q-start))
    for pack in (1, RATIO):
        info = N(tile_m=m,tile_n=n,is_local=True,is_causal=False,window_size_left=W,window_size_right=W,
                 qhead_per_kvhead_packgqa=pack,is_split_kv=True)
        for q in range(Q):
            lo,hi = env["get_n_idx_left_right"](info,full,q*pack)
            local_lo,local_hi = env["get_n_idx_left_right"](info,compact,q*pack)
            assert (lo,min(hi,full.seqlen_k)) == (start+local_lo,start+min(local_hi,compact.seqlen_k))
            calls["query_key_ranges"] += 1
        for splits in set(split_counts.values()):
            for i in range(splits):
                left,right = env["get_n_block_min_max"](info,full,0,i,splits)
                lo,hi = env["get_n_block_min_max"](info,compact,0,i,splits)
                assert (left,right)==(lo+start//n,hi+start//n)
                calls["physical_split_ranges"] += 1

table = {}
for users, prefixes in by_load.items():
    old = [p+Q for p in prefixes]
    short = [length-max(0,p-W)//n*n for p,length in zip(prefixes,old)]
    table[users] = dict(requests=len(old),full_median_columns=median(old),suffix_median_columns=median(short),
                        suffix_min=min(short),suffix_max=max(short),total_full_columns=sum(old),
                        total_suffix_columns=sum(short),avoided_logical_address_fraction=1-sum(short)/sum(old))
print(json.dumps(dict(pin=PIN, source_hashes=hashes, client=str(client.relative_to(ROOT)), client_sha256=sha256(client.read_bytes()).hexdigest(),
    server=str(log.relative_to(ROOT)), server_sha256=sha256(log.read_bytes()).hexdigest(), native_fa4_geometry=[m,n],
    pack_q_heads_per_kv=RATIO, bound_suffix=W+Q+n-1, generic_local_scheduler_tiles=(loaded+n-1)//n,
    native_num_splits_by_graph_batch=split_counts, distinct_prefix_lengths=len(positions),
    checked=dict(calls), first_block_request_shapes=table, physical_H200_for_candidate=False,
    scope="Request-level prompt lengths from V41, one representative first decode block per completed request. Counts are logical page IDs at page size one. Tests execute native host arithmetic for FA4 intervals and split membership. Neither attention timing, geometry for every replay, nor HBM counters are available."), indent=2))
