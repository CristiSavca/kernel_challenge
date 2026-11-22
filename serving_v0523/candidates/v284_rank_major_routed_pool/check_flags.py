"""Cross-check the complete argument parser and the public batch dataclass against V283."""
import ast
import importlib.util
import itertools
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("registration",here.parent/"v176_integrated_prompt_decode/independent_registration.py")
cli=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
parent=json.loads((here.parent/"v283_rank_major_unpool_boundary/validation.json").read_bytes())
v=json.loads(Path(sys.argv[2]).read_bytes()) if len(sys.argv)>2 else json.loads((here/"validation.json").read_bytes())
group,keys=cli.FLAG_FIELDS["spec"]
cli.FLAG_FIELDS["spec"]=(group,keys+("speculative_dflash_mixed_prefill",))
base,tree=parent["tree"],v["tree"]
parent_order,_,_,parent_defaults,parent_namespaces=cli.parser(base)
order,parser,_,defaults,namespaces=cli.parser(tree)
assert order==parent_order and defaults==parent_defaults and namespaces==parent_namespaces
expected=dict(cli.EXPECTED,speculative_dflash_sp_prefill_context=True,speculative_dflash_mixed_prefill=True)
flags=parent["launch_flags"]
assert flags==v["launch_flags"] and set(expected)=={key[2:].replace("-","_") for key in flags}
cases=0
for perm in itertools.permutations(flags):
    argv=[]
    for flag in perm:
        argv.append(flag)
        if flags[flag]:
            argv.append(flags[flag])
    data=vars(parser.parse_args(argv))
    assert all(data[key]==value and type(data[key]) is type(value) for key,value in expected.items())
    cases+=1
path="model_executor/forward_batch_info.py"
parent_fields=cli.declarations(cli.source(base,path),"ForwardBatch")
current_fields=cli.declarations(cli.source(tree,path),"ForwardBatch")
assert [ast.dump(n) for n in parent_fields]==[ast.dump(n) for n in current_fields[:-1]]
last=current_fields[-1]
assert last.target.id=="muse_ranked_pool" and ast.unparse(last.annotation)=="bool" and last.value.value is False
assert cli.public_signature(tree,path,"ForwardBatch")==(*cli.public_signature(base,path,"ForwardBatch"),last.target.id)
assert cli.source(base,"layers/logits_processor.py")==cli.source(tree,"layers/logits_processor.py")
print(json.dumps(dict(parent=base,tree=tree,msgspec=cli.msgspec.__version__,flags=flags,parsed_orderings=cases,
    unchanged_server_argument_fields=len(order),unchanged_ForwardBatch_fields=len(parent_fields),
    appended_batch_field=last.target.id,default=False,logits_schema_unchanged=True),indent=2))
