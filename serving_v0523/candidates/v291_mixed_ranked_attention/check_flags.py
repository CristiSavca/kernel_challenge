"""Parse the combined candidate's actual flags and cross-check the V290 schema."""
import ast
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("native_cli",next(here.parent.glob("v176_*/independent_registration.py")))
cli=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
manifest=json.loads((here.parent/"v290_mixed_ranked_peer_serving/validation.json").read_bytes())
components=json.loads((here/"components.json").read_bytes())
tree,parent=components["integrated240"],components["290"]
assert parent==manifest["tree"]
for key,extra in (("spec","speculative_dflash_mixed_prefill"),("schedule","max_prefill_tokens")):
    group,names=cli.FLAG_FIELDS[key]
    cli.FLAG_FIELDS[key]=group,(*names,extra)
stock=cli.parser(manifest["base"])[0]
before=cli.parser(parent)[0]
order,p,types,defaults,namespaces=cli.parser(tree)
assert order==before and tuple(order[:len(stock)])==stock
util=sys.modules["sglang.srt.arg_groups.arg_utils"]
graph=sys.modules["sglang.srt.arg_groups.fields.exec_"].ExecGraph
util.add_cli_args_from_dataclass(p,graph,fields=["cuda_graph_max_bs_prefill"])
flags=manifest["launch_flags"]
expected=dict(cli.EXPECTED,speculative_dflash_sp_prefill_context=True,speculative_dflash_mixed_prefill=True,
              max_prefill_tokens=16640,cuda_graph_max_bs_prefill=16640)
for key in ("speculative_dflash_sp_prefill_context","speculative_dflash_mixed_prefill"):
    assert defaults[key] is False and namespaces[key]=="spec"
assert set(expected)=={k[2:].replace("-","_") for k in flags}
namesets={}
for path,cls in (("model_executor/forward_batch_info.py","ForwardBatch"),("layers/logits_processor.py","LogitsMetadata"),
                 ("layers/logits_processor.py","LogitsProcessorOutput")):
    def fields(rev):
        return [ast.dump(n,include_attributes=False) for n in cli.declarations(cli.source(rev,path),cls)]
    prior,current=fields(parent),fields(tree)
    if cls=="ForwardBatch":
        assert current[:-2]==prior
        actual=cli.declarations(cli.source(tree,path),cls)[-2:]
        assert [(f.target.id,ast.unparse(f.annotation),f.value.value) for f in actual]==[
            ("mixed_verify_is_causal","bool",False),("mixed_verify_prompt_count","int",0)]
        assert cli.public_signature(tree,path,cls)==(*cli.public_signature(parent,path,cls),
            "mixed_verify_is_causal","mixed_verify_prompt_count")
    else:
        assert current==prior
    namesets[cls]=len(prior)
cases=0
for names in permutations(flags):
    argv=[v for k in names for v in ((k,flags[k]) if flags[k] else (k,))]
    values=vars(p.parse_args(argv))
    assert all(values[k]==v and type(values[k]) is type(v) for k,v in expected.items())
    cases+=1
print(json.dumps(dict(tree=tree,parent=parent,msgspec=cli.msgspec.__version__,launch_flags=flags,
    parsed_flag_orderings=cases,verified_values=expected,preserved_public_field_counts=namesets,appended_batch_fields=["mixed_verify_is_causal","mixed_verify_prompt_count"],
    unchanged_V290_server_argument_fields=len(before)),indent=2))
