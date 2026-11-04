"""Reproducible source-only overlay; never writes the shared serving checkout."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REV = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
PARENT_SHA = "944768d40e380bfb88f917ed5e69fcc1f9470995d3e6d65cfb6267b756e04bb7"
MODULE = "python/sglang/kernels/ops/muse_spark_v12/native_serving_v27.py"
QUANT = "python/sglang/srt/layers/quantization/muse_spark_nvfp4_moe.py"
EXPERT = "python/sglang/srt/layers/muse_spark_fp4_moe.py"
MODEL = "python/sglang/srt/models/muse_spark.py"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build():
    edits = []
    sources = {}
    originals = {}
    for path in (QUANT, EXPERT, MODEL):
        raw = subprocess.check_output([
            "git", "-C", str(ROOT / "serving/sglang"), "show", REV + ":" + path
        ])
        originals[path] = raw.decode()
        sources[path] = raw.decode()
    parent = (HERE.parent / "native_serving_v26/component.py").read_bytes()
    assert sha(parent) == PARENT_SHA
    source = parent.decode()
    # Drop only official-problem prepare/kernel_fn. Keep the public core API.
    stop = source.index("\ndef prepare(weights):")
    source = source[:stop]
    source = source.replace(
        "# okc: problem=muse_moe_w4a16\n",
        "# V27 serving adapter over frozen V26; no official problem wrapper.\n", 1
    )
    source = source.replace("okc_serving_wgs_v26_d", "okc_serving_wgs_v27_d")
    # Only preparation-domain declines get a typed exception. Device failures
    # from extension initialization remain ordinary errors and propagate.
    a = source.index("def prepare_native(")
    b = source.index("\n\nclass Workspace:", a)
    source = source[:a] + source[a:b].replace(
        "raise ValueError(", "raise NativeUnsupported("
    ) + source[b:]
    # All nine long lines are CUDA inline-asm constraint operand lists.
    # Break only after a comma; token sequence and generated PTX text unchanged.
    formatted = []
    formatting = []
    for line in source.splitlines():
        if len(line) > 400:
            assert line.lstrip().startswith((': "=f"', ': "+f"'))
            parts = line.split(", ")
            rows = []
            row = parts[0]
            for part in parts[1:]:
                if len(row) + len(part) + 2 > 180:
                    rows.append(row + ",")
                    row = "          " + part
                else:
                    row += ", " + part
            rows.append(row)
            new = "\n".join(rows)
            formatting.append([line, new])
            formatted.extend(rows)
        else:
            formatted.append(line)
    source = "\n".join(formatted) + "\n\n"
    source += (HERE / "integration.py.inc").read_text()
    sources[MODULE] = source

    def edit(path, old, new):
        assert sources[path].count(old) == 1, (path, old[:80])
        sources[path] = sources[path].replace(old, new, 1)
        edits.append({"path": path, "old": old, "new": new})

    edit(QUANT, "    def _process_weights_for_marlin(self, layer):\n",
         "    def _process_weights_for_marlin(self, layer):\n"
         "        from sglang.kernels.ops.muse_spark_v12.native_serving_v27 import (\n"
         "            try_install_native,\n"
         "        )\n\n"
         "        if try_install_native(layer):\n"
         "            return\n")
    edit(EXPERT, "                device=self.gate_up_proj.device,\n",
         "                device=(self._native_v27_bank['device']\n"
         "                        if getattr(self, '_native_v27_bank', None) is not None\n"
         "                        else self.gate_up_proj.device),\n")
    old = "        if pre_expert_norm is not None:\n            hidden_states = pre_expert_norm(hidden_states)\n        expert_outputs, local_topk_ids = self._marlin_expert_outputs(\n"
    new = (
        "        if getattr(self, '_native_v27_bank', None) is not None:\n"
        "            from sglang.kernels.ops.muse_spark_v12.native_serving_v27 import (\n"
        "                native_forward,\n"
        "            )\n\n"
        "            return native_forward(\n"
        "                self, hidden_states, topk_scores, topk_ids,\n"
        "                post_expert_norm, pre_expert_norm,\n"
        "            )\n" + old
    )
    edit(EXPERT, old, new)
    edit(MODEL, "        self.layers = nn.ModuleList()\n",
         "        self.layers = nn.ModuleList()\n"
         "        from sglang.kernels.ops.muse_spark_v12.native_serving_v27 import (\n"
         "            NativeModelOwner,\n"
         "        )\n\n"
         "        self._native_v27_owner = NativeModelOwner()\n")
    edit(MODEL, "                layer.mlp.side_stream = side_stream\n",
         "                layer.mlp.side_stream = side_stream\n"
         "                if layer.mlp.nvfp4_checkpoint:\n"
         "                    layer.mlp.experts._native_v27_owner = self._native_v27_owner\n")

    pieces = []
    entries = {}
    for path, text in sources.items():
        ast.parse(text)
        target = HERE / "candidate" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        before = originals.get(path, "")
        if path in originals:
            target = HERE / "baseline" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(before)
        diff = ''.join(difflib.unified_diff(
            before.splitlines(True), text.splitlines(True),
            fromfile='a/' + path if before else '/dev/null', tofile='b/' + path,
        ))
        header = 'diff --git a/' + path + ' b/' + path + '\n'
        if not before:
            header += 'new file mode 100644\n'
        pieces.append(header + diff)
        entries[path] = {'baseline_sha256': sha(before.encode()) if before else None,
                         'candidate_sha256': sha(text.encode()), 'bytes': len(text.encode())}
    patch = ''.join(pieces)
    assert len(patch.encode()) <= 409600
    assert all(len(line[1:]) <= 400 for line in patch.splitlines()
               if line.startswith('+') and not line.startswith('+++'))
    (HERE / "native_serving_v27.patch").write_text(patch)
    (HERE / "edits.json").write_text(json.dumps(edits, indent=2) + '\n')
    (HERE / "formatting.json").write_text(json.dumps(formatting, indent=2) + '\n')
    manifest = {'serving_revision': REV, 'parent_component_sha256': PARENT_SHA,
                'files': entries, 'patch_sha256': sha(patch.encode()),
                'patch_bytes': len(patch.encode()), 'formatting_only_lines': len(formatting),
                'builder_sha256': sha(Path(__file__).read_bytes()),
                'adapter_fragment_sha256': sha((HERE / 'integration.py.inc').read_bytes()),
                'workspace_contract': 'One model/device logical serial forward lane; unsupported overlap declines before conversion.',
                'no_gpu_validation': True}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
