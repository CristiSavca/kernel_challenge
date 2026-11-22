"""Execute both pinned C++ copy selectors against a stream-recording CUDA ABI."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


FILE = "python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh"
HERE = Path(__file__).resolve().parent
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def body(source):
    return source[source.index("inline void copy_page_first_pages_fallback("):
                  source.index("template <int64_t kElementSize")]


def main(stage):
    candidate = (stage / FILE).read_bytes()
    original = subprocess.check_output(["git","-C",stage,"show",BASE + ":" + FILE])
    support, main = (HERE / "copy_fixture.cpp").read_text().split("// INSERT STOCK AND CANDIDATE HERE")
    code = support + "\nnamespace original {\n" + body(original.decode()) + "\n}\n"
    code += "\nnamespace candidate {\n" + body(candidate.decode()) + "\n}\n" + main
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        (tmp/"test.cpp").write_text(code)
        for label,flags in (("cuda12_7",["-DCUDA_VERSION=12070"]),("cuda12_9",["-DCUDA_VERSION=12090"]),
                            ("cuda13",["-DCUDA_VERSION=13000"]),("rocm",["-DUSE_ROCM","-DCUDA_VERSION=12090"])):
            subprocess.run(["clang++","-std=c++20","-O2","-Werror","-fsanitize=undefined","-fno-sanitize-recover=all",
                            *flags,str(tmp/"test.cpp"),"-o",str(tmp/"test")],check=True)
            rows.append(dict(path=label,**json.loads(subprocess.check_output([tmp/"test"]))))
    return dict(fixture="production header functions, queued CPU memcpy in stream order",
                base=BASE,parent_header_sha256=hashlib.sha256(original).hexdigest(),
                source_sha256=hashlib.sha256(candidate).hexdigest(),cases=rows,hardware_gpu=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",type=Path)
    print(json.dumps(main(p.parse_args().stage),indent=2))
