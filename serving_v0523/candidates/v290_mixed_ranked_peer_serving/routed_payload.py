"""Count the distinct peer token rows touched on the reconstructed H200 prompt fixtures."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
donor = here.parent / "v274_four_block_prompt_multicast"
draw = donor / "routes.py"
frozen = donor / "routes.json"
archive = here.parents[2] / "experiments/modal_serving/20261113T065259Z/result.json"
prior = json.loads(frozen.read_bytes())
assert sha256(archive.read_bytes()).hexdigest() == prior["archived_sha256"]
module = ast.parse(draw.read_bytes())
philox = next(n for n in module.body if getattr(n,"name",None) == "philox")
env = {"np": np}
exec(compile(ast.Module([philox],[]),str(draw),"exec"),env)
results = []
for fixture in prior["cases"]:
    tokens = fixture["tokens"]
    width, k, ranks = 4096, 8, 8
    elements = tokens * 256
    threads = min(elements, 132*2048)
    draws = np.concatenate([env["philox"](77+tokens,i,threads)
        for i in range((elements+4*threads-1)//(4*threads))])[:elements]
    values = ((draws.astype(np.float64)+0.5)/2**32).astype(np.float32)
    values[values==1] = 0
    values = values.reshape(tokens,256)
    choices = np.argpartition(values,k,axis=1)[:,:k+1]
    pos = np.take_along_axis(values,choices,axis=1).argsort(axis=1)
    ordered = np.take_along_axis(choices,pos,axis=1)
    sorted_values = np.take_along_axis(values,ordered,axis=1)
    assert (sorted_values[:,k-1] < sorted_values[:,k]).all()
    ids = ordered[:,:k]
    hist = np.bincount(ids.ravel(),minlength=256).reshape(ranks,32)
    assert hist.tolist() == fixture["histograms"]
    assert hist[fixture["measured_rank"]].sum() == fixture["measured_local_routes"]
    owner = np.arange(tokens) // (tokens//ranks)
    mask = np.stack([(ids//32==dest).any(axis=1) for dest in range(ranks)])
    volumes = np.stack([np.bincount(owner[needed],minlength=ranks) for needed in mask])
    unique = mask.sum(axis=1)
    remote = volumes.copy()
    np.fill_diagonal(remote,0)
    as_mib = lambda counts: np.asarray(counts)*width*2/2**20
    replicated_remote = float(as_mib(tokens-tokens//ranks))
    incoming = as_mib(remote.sum(axis=1))
    outgoing = as_mib(remote.sum(axis=0))
    results.append(dict(tokens=tokens, routed_slots=tokens*k, local_routes_per_rank=hist.sum(axis=1).tolist(),
        distinct_tokens_by_reader_and_owner=volumes.tolist(), distinct_tokens_per_reader=unique.tolist(),
        baseline_incoming_MiB_per_rank=replicated_remote, peer_incoming_MiB_by_rank=incoming.tolist(),
        peer_outgoing_MiB_by_rank=outgoing.tolist(), peak_incoming_MiB=float(incoming.max()),
        peak_outgoing_MiB=float(outgoing.max()), least_incoming_payload_saved_MiB=float(replicated_remote-incoming.max()),
        median_incoming_payload_saved_MiB=float(replicated_remote-np.median(incoming))))
print(json.dumps(dict(scope="Independent host reconstruction of the archived standalone H200 synthetic top-eight draw. Only rank 3's aggregate route count was physically logged. Each distinct remote owner row is charged once per destination rank. No GPU/link throughput, cache-line retransmission, routing trace from the serving judge or candidate timing is implied.",
    sources={str(p.relative_to(here.parents[2])):sha256(p.read_bytes()).hexdigest() for p in (draw,frozen,archive)},
    cases=results),indent=2))
