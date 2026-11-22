"""Check V160's flag with the repository's real Spec declaration and CLI builder."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import msgspec


def positional_order(source):
    module = ast.parse(source)
    for item in module.body:
        if isinstance(item, ast.Assign) and item.targets[0].id == "POSITIONAL_FIELD_ORDER":
            return ast.literal_eval(item.value)
    raise AssertionError("Missing positional field declaration")


if __name__ == "__main__":
    stage = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(stage / "python"))
    from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
    from sglang.srt.arg_groups.fields import collect_input_fields
    from sglang.srt.arg_groups.fields.spec import Spec

    order_path = "python/sglang/srt/arg_groups/field_order.py"
    manifest = json.loads(Path(__file__).with_name("validation.json").read_text())
    baseline = subprocess.check_output(["git", "-C", stage, "show", manifest["base"] + ":" + order_path])
    before = positional_order(baseline)
    after = positional_order((stage / order_path).read_bytes())
    field = "speculative_dflash_mixed_prefill"
    assert after == (*before, field)
    _, defaults, namespaces = collect_input_fields([Spec])
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, Spec, fields=[field])
    flag = "--" + field.replace("_", "-")
    assert defaults[field] is False and namespaces[field] == "spec"
    assert getattr(parser.parse_args([]), field) is False
    assert getattr(parser.parse_args([flag]), field) is True
    tree = subprocess.check_output(["git", "-C", stage, "write-tree"], text=True).strip()
    assert tree == manifest["tree"]
    print(json.dumps(dict(tree=tree, flag=flag, default=False, explicit=True, namespace="spec",
                          original_positional_fields_preserved=len(before), msgspec=msgspec.__version__,
                          original_order_source_sha256=hashlib.sha256(baseline).hexdigest()), indent=2))
