"""Generate exactly the exported entry points declared by the production launcher."""
import ast
from pathlib import Path
import sys

pin, stage = map(Path, sys.argv[1:])
src = stage/"python/sglang/srt/layers/muse_spark/decode_gate_up.py"
call = next(n for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id=="load_jit")
wrappers = ast.literal_eval(next(x.value for x in call.keywords if x.arg=="cuda_wrappers"))
spec = pin/"python/sglang/kernels/jit/utils/compile/spec.py"
source = spec.read_text()
ns = {"Tuple":tuple}
exec(source[source.index("_FFI_INCLUDES ="):source.index("def resolve_sources(")], ns)
print(ns["_wrapper_source"](("decode_gate_up_launch.cuh",), wrappers), end="")
