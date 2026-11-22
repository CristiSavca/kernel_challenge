"""Verify unchanged V195 gates/geometry, phase identity and the final per-run load/reduction handoff."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=Path(sys.argv[1])
p=here.parent/"v195_fused_expert_affine_verify"
manifest=json.loads((p/"validation.json").read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
def show(path):return subprocess.check_output(["git","-C",root,"show",manifest["tree"]+":"+path])
att="python/sglang/kernels/ops/attention/paged_verify_attention.py"
def methods(data): return {n.name:n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef)}
orig=methods(show(att)); cur=methods((root/att).read_bytes())
same=[n for n in orig if ast.dump(cur[n])==ast.dump(orig[n])]
assert set(orig)-set(same)=={"_partition_attention","_accumulate_tiles"}
assert "_direct_tile_args" in same and "paged_verify_attention" in same
held=set(manifest["paths"])-{att}
assert all(show(n)==(root/n).read_bytes() for n in held)
assert sha((p/"patch.diff").read_bytes())==manifest["patch_sha256"]
body=ast.unparse(cur["_partition_attention"])
assert "_next_tile_run(Origins + request * ORIGIN_STRIDE, cursor, tile_end, CAPACITY - TILE_N)" in body
assert body.count("row_max, row_sum, acc = _accumulate_tiles(")==3 and body.count("row_max, row_sum, acc)")>=2
assert "if tile_end - tile_start <= 32" in body and "cursor = stop" in body
assert "delta_n" not in body
model=show("python/sglang/srt/models/muse_spark.py").decode()
assert "attn_output = self.self_attn(positions, attn_input, forward_batch)" in model
assert "ffn_output = self.mlp(ffn_input, router_input, forward_batch)" in model
assert "grouped_decode=decode_or_verify" in model
print(json.dumps(dict(parent_sha256=manifest["patch_sha256"],parent_tree=manifest["tree"],
    kernel_file_sha256=sha((root/att).read_bytes()),other_parent_files_identical=len(held),
    unchanged_methods=same,segmented_methods=sorted(set(orig)-set(same)),v195_expert_files=manifest["independent_phase_sources"]["191"],
    full_attention_gate_unchanged=True,vmm_guard_unchanged=True,retains_online_state_and_tile_order=True,live_cuda=False),indent=2))
