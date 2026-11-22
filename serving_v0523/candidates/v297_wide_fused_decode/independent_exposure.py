"""Map archived eight-H200 decode reports and captured graph sizes onto pinned decode bins."""
import ast
from bisect import bisect_left
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
log, = (ROOT/"experiments/modal_node/20261113T204343Z-bench").glob("*v41__server.log")
raw = log.read_bytes()
lines = raw.decode().splitlines()
settings = ast.literal_eval(next(row.split("server_args=",1)[1] for row in lines if "server_args={" in row))
assert settings["tp_size"] == settings["ep_size"] == 8
assert settings["speculative_num_draft_tokens"] == 8 and settings["disable_cuda_graph_padding"] is False
capture, = {tuple(ast.literal_eval(m.group(1))) for row in lines
            if (m:=re.search(r"TP0 EP0\] Capture target verify CUDA graph end\. bs=(\[[\d, ]+\])",row))}
configs = json.loads((HERE.parent/"v275_wide_decode_output_tiles/startup.json").read_bytes())["tests"]
bins = {projection:[(case["max_inclusive"],case["chosen"][0]) for case in configs
                    if case["projection"]==projection and not case["allowed"] and not case["refused"]]
        for projection in ("w13","w2")}
assert bins["w13"] == bins["w2"] == [(112,8),(216,16),(320,24)]
flushes = [i for i,row in enumerate(lines) if "TP0 EP0] Cache flushed successfully!" in row]
waves = [lines[start:end] for start,end in zip(flushes,flushes[1:]+[len(lines)])]
loads = [1,1,1,4,4,16,32]
assert len(loads) == len(waves)
totals = {}
for load,wave in zip(loads,waves):
    seen = totals.setdefault(load,Counter())
    for row in wave:
        m = re.search(r"TP0 EP0\] Decode batch, #running-req: (\d+),.*?cuda graph: (True|False),",row)
        if not m:
            continue
        assert m.group(2) == "True"
        graph = capture[bisect_left(capture,int(m.group(1)))]
        tokens = graph * settings["speculative_num_draft_tokens"]
        estimated = tokens*8*32//256
        seen[next(width for maximum,width in bins["w13"] if estimated<=maximum)]+=1
print(json.dumps(dict(archive=str(log.relative_to(ROOT)),sha256=sha256(raw).hexdigest(),capture_keys=capture,
    bins_max_and_M=bins,scheduler_reports={load:{f"M{m}":counts[m] for m in (8,16,24)}
                                        for load,counts in totals.items()},
    scope="One TP0 decode status report per logged interval in an earlier V41 full-graph DFLASH run. Each active user count is rounded up to the recorded target-verify graph key. The counts describe shape eligibility. Actual V297 kernel selection and elapsed time remain unmeasured."),indent=2))
