"""CPU source-preservation checks for the isolated dense-chain V9 experiment."""

import ast
import copy
import hashlib
from pathlib import Path


def verify():
    here = Path(__file__).resolve().parent
    parent = here.parent / "submission.py"
    candidate = here / "native_dense_chain_select_v9.py"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == "80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c"
    original = ast.parse(parent.read_text())
    modified = ast.parse(candidate.read_text())
    normalized = copy.deepcopy(modified)
    added = {"_dense_chain_options", "_dense_chain_select"}
    functions = {n.name: n for n in modified.body if isinstance(n, ast.FunctionDef)}
    assert added <= functions.keys()
    normalized.body = [node for node in normalized.body if not (isinstance(node, ast.FunctionDef) and node.name in added)]
    prepare = next(node for node in normalized.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    assert isinstance(prepare.body[0], ast.Import)
    assert [(a.name, a.asname) for a in prepare.body[0].names] == [("time", "_prepare_clock")]
    assert isinstance(prepare.body[1], ast.Assign)
    assert prepare.body[1].targets[0].id == "_prepare_started"
    prepare.body = prepare.body[2:]
    calls = [i for i, node in enumerate(prepare.body) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "_dense_chain_select"]
    assert len(calls) == 1
    del prepare.body[calls[0]]
    assert ast.dump(normalized, include_attributes=False) == ast.dump(original, include_attributes=False), \
        "Unexpected modification outside the two new helpers and the bounded prepare hook"

    # Execute only the pure existing-choice enumerator, without importing Torch,
    # Triton or the submission. It must leave dispatch state unmodified and must
    # never change the inner router backend while considering pooler alternatives.
    state = {"bfp": {"sk2": {"pooler": (), "router": ()}, "choice": {("router", 128): ("wide",)}}}
    namespace = {"bfp__candidates": lambda layer, T, bs: [("torch",), ("rq_tma",)]}
    tree = ast.Module(body=[copy.deepcopy(functions["_dense_chain_options"])], type_ignores=[])
    exec(compile(tree, str(candidate), "exec"), namespace)
    before = copy.deepcopy(state)
    for T in (16, 32, 128, 256):
        for name in ("o", "pr", "un"):
            options = namespace["_dense_chain_options"](state, T, name)
            assert all(layer != "router" for _, layer, _ in options)
            assert all(outer != "bfpf" for outer, _, _ in options) if T > 32 or name != "pr" else True
            assert all(outer is not False for outer, _, _ in options) if T > 32 else True
    assert state == before
    return {"parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "existing_ast_unchanged_after_prepare_hook_removal": True,
            "existing_choice_enumeration_passed": True,
            "cuda_checked": False}


if __name__ == "__main__":
    print(verify())
