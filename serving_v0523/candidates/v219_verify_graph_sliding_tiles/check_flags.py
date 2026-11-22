"""Parse the six serving overrides together through the assembled SGLang CLI groups."""
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
helper = HERE.parent/"v176_integrated_prompt_decode/independent_registration.py"
spec = importlib.util.spec_from_file_location("cli_registration",helper)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
group, keys = cli.FLAG_FIELDS["spec"]
cli.FLAG_FIELDS["spec"] = (group, keys + ("speculative_dflash_mixed_prefill",))
manifest_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE/"validation.json"
manifest = json.loads(manifest_path.read_bytes())
tree = manifest["tree"]
expected = dict(cli.EXPECTED, speculative_dflash_sp_prefill_context=True, speculative_dflash_mixed_prefill=True)
declared_order, parser, _, defaults, namespaces = cli.parser(tree)
base_order, _, _, _, _ = cli.parser(manifest["base"])
assert len(base_order) <= len(declared_order)
assert tuple(declared_order[:len(base_order)]) == tuple(base_order)
assert defaults["speculative_dflash_sp_prefill_context"] is False
assert namespaces["speculative_dflash_sp_prefill_context"] == "spec"
parent_order, _, _, _, _ = cli.parser(manifest["parent_tree"])
assert tuple(declared_order) == tuple(parent_order)
assert defaults["speculative_dflash_mixed_prefill"] is False and namespaces["speculative_dflash_mixed_prefill"] == "spec"
for path, name, appended in (
    ("model_executor/forward_batch_info.py","ForwardBatch",("mixed_verify_logits_indices","mixed_verify_sharded_logits","dflash_verify_raw_buffer")),
    ("layers/logits_processor.py","LogitsMetadata",("mixed_verify_logits_indices","mixed_verify_sharded_logits")),
    ("layers/logits_processor.py","LogitsProcessorOutput",("dflash_prefill_context_done","dflash_prefill_kv_projected","dflash_verify_fc_shard")),
):
    # Check every positional value using the real ordered field declarations.
    if name == "LogitsMetadata":
        # Its enum defaults come from the live model runtime, which the worker
        # fixture exercises. Preserve the declared order and complete defaults.
        import ast
        baseline = cli.declarations(cli.source(manifest["base"],path),name)
        current = cli.declarations(cli.source(tree,path),name)
        assert [ast.dump(n) for n in current[:len(baseline)]] == [ast.dump(n) for n in baseline]
        assert tuple(n.target.id for n in current[len(baseline):]) == appended
    else:
        base_sig = cli.public_signature(manifest["base"],path,name)
        assert cli.public_signature(tree,path,name) == (*base_sig, *appended)
for path, name, added in (
    ("speculative/dflash_info.py", "DFlashVerifyInput", "consumes_verify_context"),
    ("layers/attention/flashattention_backend.py", "FlashAttentionMetadata", "swa_verify_tile_origins"),
):
    before = cli.declarations(cli.source(manifest["common_tree"],path),name)
    after = cli.declarations(cli.source(tree,path),name)
    assert [cli.ast.dump(x) for x in before] == [cli.ast.dump(x) for x in after[:-1]]
    assert after[-1].target.id == added
flags = manifest["launch_flags"]
assert set(expected) == {flag[2:].replace("-","_") for flag in flags}
cases = 0
for order in permutations(flags):
    argv = []
    for flag in order:
        argv.append(flag)
        if flags[flag]:
            argv.append(flags[flag])
    data = vars(parser.parse_args(argv))
    assert all(data[k] == v and type(data[k]) is type(v) for k,v in expected.items())
    cases += 1
print(json.dumps(dict(tree=tree,flags=flags,msgspec=cli.msgspec.__version__, parsed_cases=cases,
    parsed_values=expected, original_positional_fields=len(base_order),current_positional_fields=len(declared_order),
    physical_h200=False),indent=2))
