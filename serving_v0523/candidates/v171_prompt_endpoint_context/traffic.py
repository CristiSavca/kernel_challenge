"""Read only the archived official serving receipts and quantify hypothetical TP8 buffers."""
import collections
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1]/"runs"
PATTERN = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")


def geometry(rows, requests):
    ranks, hidden, taps, itemsize = 8, 8192, 5, 2
    rows_per_rank = (rows+ranks-1)//ranks
    full = rows_per_rank*ranks*hidden*itemsize
    selected = requests*ranks*hidden*itemsize
    return dict(query_rows=rows,batch=requests,local_rows=rows_per_rank,
                passes_row_count_predicates=rows>=128*ranks and 0<requests<rows_per_rank,
                original_two_gather_output_bytes=full*(taps+1),
                projected_two_gather_output_bytes=full+selected,endpoint_gather_output_bytes=selected)


def check():
    snapshots = []
    for f in sorted(RUNS.glob("*.json")):
        if not re.fullmatch("[a-f0-9]{16}",f.stem):
            continue
        job = json.loads(f.read_text())
        if job.get("board")!="serving" or job.get("id")!=f.stem:
            continue
        result = job.get("result") or {}
        startup = result.get("startup_log") or {}
        observed = [list(map(int,m)) for m in PATTERN.findall(startup.get("text",""))]
        snapshots.append(dict(id=f.stem,mode=job["mode"],status=result.get("status"),
                              cut=startup.get("cut"),observed=observed))
    all_rows = [tuple(row) for job in snapshots for row in job["observed"]]
    v64 = json.loads((RUNS/"abd217baf0ed49fd.json").read_text())["result"]
    return dict(archive_ids=len(snapshots),receipts=snapshots,
                observed_fields=["new_sequences","new_tokens","cached_tokens"],
                observed_counts=[dict(geometry=list(k),count=v) for k,v in sorted(collections.Counter(all_rows).items())],
                observed_passing_cardinality_gate=sum(geometry(n,b)["passes_row_count_predicates"] for b,n,_ in all_rows),
                measured_v64=dict(id="abd217baf0ed49fd",shape=v64["shape"],concurrency=v64["concurrency"],
                                  peak_scored_memory=v64["gpu_memory"],
                                  ttft_rungs=[{k:r[k] for k in ("concurrency","ttft_first_p90_s","ttft_later_p90_s")} for r in v64["rungs"]]),
                conditional_uniform_cold_74160=[geometry(n,1) for n in (16384,16384,16384,16384,8624)],
                conditional_later_equal_lengths=[geometry(753*b,b) for b in (1,2,4,8,16)],
                conditional_max_chunk=[geometry(16384,b) for b in (1,4,16,32)],
                geometry_unit="logical all-gather destination bytes per rank, excluding copies, links and protocol")
if __name__=="__main__":
    print(json.dumps(check(),indent=2))
