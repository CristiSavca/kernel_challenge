"""Compile V176/V177 with Humming's published NVRTC flags, name expression and shim headers."""
import ast
import ctypes as c
import sys
from pathlib import Path

vendor, generated, library, cuda_include = map(Path, sys.argv[1:])
nv = c.CDLL(str(library))
compiler = ast.parse((vendor/"jit/compiler.py").read_text())
cls = next(x for x in compiler.body if getattr(x, "name", "") == "NVRTCCompiler")
shims = ast.literal_eval(next(x.value for x in cls.body if isinstance(x, ast.AnnAssign)
                              and getattr(x.target, "id", None) == "_STD_HEADER_SHIMS"))
arr = lambda seq: (c.c_char_p * len(seq))(*(s.encode() for s in seq))
headers, includes = arr(list(shims.values())), arr(list(shims))
flags = ["--gpu-architecture=sm_90a", "-std=c++17", "-DHUMMING_DEBUG_KERNEL=0", "--use_fast_math",
         "--dopt=on", "-extra-device-vectorization", "--ptxas-options=-O3", "--ptxas-options=--register-usage-level=10",
         "--diag-suppress=39,161,174,177,940,1444", "-default-device", "-I"+str(vendor/"include"), "-I"+str(cuda_include)]
for m in (136,152,176):
    for role in ("8192","gate"):
        path = generated / f"{role}_{m}.cu"
        source, expr = path.read_text().rsplit("auto ptr = reinterpret_cast<void*>(&",1)
        expr = " ".join(expr.strip().removesuffix(");").split()).encode()
        prog = c.c_void_p()
        rc = nv.nvrtcCreateProgram(c.byref(prog),source.encode(),b"kernel.cu",len(shims),headers,includes)
        assert rc == 0 and nv.nvrtcAddNameExpression(prog,expr) == 0
        rc = nv.nvrtcCompileProgram(prog,len(flags),arr(flags))
        size = c.c_size_t()
        nv.nvrtcGetProgramLogSize(prog,c.byref(size))
        log = c.create_string_buffer(size.value)
        nv.nvrtcGetProgramLog(prog,log)
        path.with_suffix(".nvrtc.log").write_bytes(log.value)
        if rc:
            raise RuntimeError((rc,log.value.decode()[:3000]))
        assert nv.nvrtcGetCUBINSize(prog,c.byref(size)) == 0
        cubin = c.create_string_buffer(size.value)
        assert nv.nvrtcGetCUBIN(prog,cubin) == 0
        path.with_suffix(".nvrtc.cubin").write_bytes(cubin.raw)
        assert nv.nvrtcDestroyProgram(c.byref(prog)) == 0
