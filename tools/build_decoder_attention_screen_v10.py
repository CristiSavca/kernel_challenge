"""Build lightweight attention-only screens of exact decoder native code/flags.

These are local diagnostics, not the active standalone87a9 implementation.
No GPU imports or execution happen in this builder.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c"
CANDIDATE_SHA = "6eb318df58c3ef5576c4ea456b4b494e1ef4548090fa7707c84e247f960c2222"
ATTENTION_SHA = "87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c"
RENAMES = {"_plan7": "_plan", "_num_sms7": "_num_sms", "_workspace7": "_workspace",
           "_launch7": "_launch", "_att7": "_module", "_cluster_bufs7": "_cluster_bufs",
           "_inv_freq7": "_inv_freq", "_TUNED7": "_TUNED", "_supported7": "_supported",
           "_WS7": "_WS", "_CL7": "_CL", "_NUM_SMS7": "_NUM_SMS", "_INV_FREQ7": "_INV_FREQ"}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def functions(tree):
    # First workspace/cluster implementations are the unguarded allocation
    # primitives; later decoder wrappers add guards, which do not alter launch.
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result.setdefault(node.name, node)
    return result


def assignments(tree):
    return {n.targets[0].id: n for n in tree.body if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}


class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = RENAMES.get(node.id, node.id)
        return node
    def visit_FunctionDef(self, node):
        node.name = RENAMES.get(node.name, node.name)
        return self.generic_visit(node)


def build(decoder, expected_sha, kind):
    assert digest(decoder) == expected_sha
    scaffold = (ROOT / "experiments/candidates/muse_attention_decode" / (ATTENTION_SHA + ".py")).read_text()
    assert digest(scaffold) == ATTENTION_SHA
    dt, st = ast.parse(decoder), ast.parse(scaffold)
    df, sf = functions(dt), functions(st)
    da, sa = assignments(dt), assignments(st)
    checked = []
    for old, new in RENAMES.items():
        if old in df and new in sf and old not in ("_att7", "_supported7"):
            assert ast.dump(Rename().visit(copy.deepcopy(df[old])), include_attributes=False) == ast.dump(sf[new], include_attributes=False), old
            checked.append(old)
    for name in ("OKC_FLATCOOP", "OKC_PDL", "OKC_FLAT1B", "OKC_PLAIN8", "OKC_F2", "OKC_S2",
                 "OKC_W0_L1", "OKC_W0_PR", "OKC_W0_L8"):
        assert ast.dump(da[name].value) == ast.dump(sa[name].value)
    assert ast.dump(da["_TUNED7"].value) == ast.dump(sa["_TUNED"].value)
    replacements = []
    for old, new in (("ATT7_CUDA_SRC", "CUDA_SRC"), ("ATT7_CPP_SRC", "CPP_SRC")):
        value = ast.literal_eval(da[old].value)
        replacements.append((ast.get_source_segment(scaffold, sa[new]), new + ' = r"""' + value + '"""'))
    # Use the decoder compiler verbatim, with names/cache variable renamed.
    compiler = ast.get_source_segment(decoder, df["_att7"])
    for old, new in (("def _att7():", "def _build():"), ("_ATT7", "_MOD"),
                     ("ATT7_CPP_SRC", "CPP_SRC"), ("ATT7_CUDA_SRC", "CUDA_SRC")):
        compiler = compiler.replace(old, new)
    compiler = compiler.replace('name=f"', 'name=f"screen_' + kind + '_', 1)
    replacements.append((ast.get_source_segment(scaffold, sf["_build"]), compiler))
    support = ast.get_source_segment(decoder, df["_supported7"]).replace("def _supported7", "def _supported")
    replacements.append((ast.get_source_segment(scaffold, sf["_supported"]), support))
    kernel = '''def kernel_fn(data: Inputs) -> torch.Tensor:
    if not _supported(data["config"]) or not _fits(data) or not _decoder_att_fits(data, data["config"]):
        return _fallback(data)
    return _launch(data, pdl=_pdl(data["config"]["batch"] * data["config"]["q_len"]))'''
    replacements.append((ast.get_source_segment(scaffold, sf["kernel_fn"]), kernel))
    result = scaffold
    for before, after in replacements:
        assert result.count(before) == 1
        result = result.replace(before, after, 1)
    extra = "\n\n" + ast.get_source_segment(decoder, da["PDL"]) + "\n" + ast.get_source_segment(decoder, da["PDL_MAX_T"]) + "\n\n"
    extra += ast.get_source_segment(decoder, df["_pdl"]) + "\n\n"
    extra += ast.get_source_segment(decoder, df["_att_fits7"]).replace("def _att_fits7", "def _decoder_att_fits") + "\n"
    result += extra
    first_line, remainder = result.split("\n", 1)
    result = first_line + "\n# LOCAL DIAGNOSTIC: exact decoder attention source/compiler flags; not the active standalone incumbent.\n" + remainder
    rt = ast.parse(result)
    ra = assignments(rt)
    assert ast.literal_eval(ra["CUDA_SRC"].value) == ast.literal_eval(da["ATT7_CUDA_SRC"].value)
    assert ast.literal_eval(ra["CPP_SRC"].value) == ast.literal_eval(da["ATT7_CPP_SRC"].value)
    return result, {"schema": 1, "kind": kind, "decoder_source_sha256": expected_sha,
                    "standalone_scaffold_sha256": ATTENTION_SHA, "wrapper_sha256": digest(result),
                    "native_sha256": digest(ast.literal_eval(ra["CUDA_SRC"].value)),
                    "native_exact": True, "compiler_from_decoder": True,
                    "equivalent_renamed_functions": checked,
                    "pdl_policy": "exact decoder PDL=True and T<=32; no predecessor in standalone graph",
                    "allocation_difference": "standalone workspaces omit decoder outer guard allocations",
                    "interpretation": "isolated decoder-attention screen, not active standalone incumbent"}


if __name__ == "__main__":
    directory = ROOT / "muse_attention_decode/candidates"
    inputs = ((ROOT / "experiments/candidates/muse_decode_layer" / (PARENT_SHA + ".py"), PARENT_SHA, "decoder_parent_v10"),
              (ROOT / "muse_decode_layer/candidates/native_q8_role_pipeline_v10.py", CANDIDATE_SHA, "decoder_q8_role_v10"))
    reports = []
    for path, expected, kind in inputs:
        source, report = build(path.read_text(), expected, kind)
        output = directory / (kind + ".py")
        output.write_text(source)
        report["path"] = str(output.relative_to(ROOT))
        reports.append(report)
    output = ROOT / "experiments/attention_q8_role_v10_wrapper_provenance.json"
    output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
