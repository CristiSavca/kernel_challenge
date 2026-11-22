"""Load-gate each real prompt variant with matching and adverse SM90a disassembly."""
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

here = Path(__file__).resolve().parent
subject = Path(sys.argv[2]).resolve()
destination = Path(sys.argv[3]).resolve()
compiled = Path(sys.argv.pop(4)).resolve()
with redirect_stdout(io.StringIO()) as buf:
    ns = runpy.run_path(str(here/"candidate.py"))
startup = json.loads(buf.getvalue())
archive = json.loads((here/"sm90a.json").read_bytes())["variants"]
accepted, rejected, zero_kernel = {}, {}, {}
for role, kernel in ns["kernels"].items():
    source = (destination/(role+".cu")).read_bytes()
    sas = (compiled/(role+".sass")).read_text()
    cubin = compiled/(role+".cubin")
    assert sha256(source).hexdigest() == archive[role]["source_sha256"]
    assert sha256(cubin.read_bytes()).hexdigest() == archive[role]["cubin_sha256"]
    kernel.cubin_loaded = False
    kernel.kernel_filename = str(cubin)
    choices = dict(matched=sas, missing_matrix=sas.replace("HGMMA.", "ABSENT."),
                   missing_transport=sas.replace("UTMALDG.2D", "ABSENT.2D"),
                   missing_wait=sas.replace("WARPGROUP.DEPBAR.LE", "ABSENT.WAIT"))
    if role=="WidePromptWeightPipeline":
        choices["ordinary_down"] = (compiled/"PromptWeightPipeline.sass").read_text()
        choices["missing_multicast"] = sas.replace("MULTICAST","ABSENT")
        choices["missing_full_width"] = sas.replace("HGMMA.64x256","HGMMA.64x248")
    rejects = []
    for key,assembly in choices.items():
        with (patch.object(ns["compiler"].NVCCCompiler, "_get_env", create=True, return_value={"binaries":{"cuobjdump":"cuobjdump"}}),
              patch("subprocess.check_output", return_value=assembly),
              patch.object(ns["HummingKernel"],"load_cubin", autospec=True) as register):
            try:
                kernel.load_cubin()
            except RuntimeError:
                assert key != "matched" and register.call_count==0
                rejects.append(key)
            else:
                assert key == "matched" and register.call_count==1
    accepted[role] = dict(source_sha256=archive[role]["source_sha256"], cubin_sha256=archive[role]["cubin_sha256"],
                          shared_bytes=archive[role]["shared_bytes"])
    rejected[role] = rejects
# The production grouped callback may use its own prepared W2 image.
# Ordinary decode resolves a different table keyed by the decode tuning string.
model = (subject/"python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py").read_text()
import ast
nodes = {f.name:f for f in ast.parse(model).body if isinstance(f,ast.FunctionDef)}
for name in ["humming_grouped_expert_outputs","humming_grouped_decode_outputs"]:
    fn = ast.unparse(nodes[name])
    prompt = name == "humming_grouped_expert_outputs"
    assert ("humming_prompt_down" in fn) == prompt
    assert ("humming_decode_down" in fn) != prompt
    zero_kernel[name] = {"prompt_table":prompt, "decode_table":not prompt}
patch = subject.parent / "patch.diff"
print(json.dumps(dict(official_patch_sha256=sha256(patch.read_bytes()).hexdigest(), startup=startup, emitted_images=accepted, rejected_assembly=rejected, dispatch=zero_kernel,
                      gpu=False, scope="Pinned production constructor, generator and pre-registration checks. Real cuobjdump text; GPU context and final module registration are simulated."), indent=2))

