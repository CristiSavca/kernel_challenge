"""Ensure all fourteen emitted sources/images match their intended alternate or frozen parent."""
import hashlib,json,sys
from pathlib import Path
here=Path(__file__).resolve().parent;scratch=Path(sys.argv[1]);stage=scratch/"sglang/python/sglang/srt/layers/muse_spark"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
w2=json.loads((here.parent/"v205_steady_decode_weight_pipeline/sm90a.json").read_bytes())["kernels"]
w13=json.loads((here/"sm90a.json").read_bytes())["kernels"]
previous_gate=json.loads((here.parent/"v197_grouped_prompt_gate_up/sm90a.json").read_bytes())["kernels"]
checked={}
for n,x in w2.items():
    for path,mode in (("paired","original"),("paired_twice","candidate")):
        ref=x[mode];p=scratch/path/n
        record=dict(source_sha256=sha(p.with_suffix(".cu")),cubin_sha256=sha(p.with_suffix(".cubin")),sass_sha256=sha(p.with_suffix(".sass")))
        assert all(ref[k]==v for k,v in record.items())
        checked[f"{path}/{n}"]=record
for n in ["decode_8","decode_16","decode_24","prompt_176"]:
    for path,mode in (("old","parent"),("full","candidate")):
        p=scratch/path/n
        rec=dict(source_sha256=sha(p.with_suffix(".cu")),cubin_sha256=sha(p.with_suffix(".cubin")),sass_sha256=sha(p.with_suffix(".sass")))
        ref=w13[n][mode] if n.startswith("decode") else previous_gate[n]
        assert ref["cubin_sha256"]==rec["cubin_sha256"] and ref["sass_sha256"]==rec["sass_sha256"]
        if n.startswith("decode"):assert ref["source_sha256"]==rec["source_sha256"]
        else:assert ref["rendered_source_sha256"]==rec["source_sha256"]
        checked[f"{path}/{n}"]=rec
assert len(checked)==14
print(json.dumps(dict(compiler="12.9.86 SM90a",sources={n:sha(stage/n) for n in ("decode_mma_pair.py","decode_gate_up.py")},images=checked),indent=2))
