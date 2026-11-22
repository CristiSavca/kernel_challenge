"""Compile the pinned FA3 metadata allocation formula for each physical attention kind."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
EXPECTED = json.loads((HERE.parent/"v132_verify_scheduler_metadata/splits.json").read_text())["immutable_sha256"]["flash_api.cpp"]
src = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(src).hexdigest()==EXPECTED
code = src.decode()
builder = code[code.index("mha_fwd_get_scheduler_metadata("):]
begin = builder.index("    auto round_multiple =")
end = builder.index("        // printf(\"(Metadata)",begin)
expressions = builder[begin:end]
assert 'params.varlen_sort_batches = !params.is_local;' in expressions
assert 'params.head_swizzle = params.is_causal || params.is_local;' in expressions
program = """
#include <iostream>
struct Params {int b; bool is_local,is_causal,varlen_sort_batches,head_swizzle;};
int main() {
for(int b: {1,4,16,32}) for(bool local: {false,true}) {
  Params params{b,local,!local,false,false}; bool scheduler_needs_semaphore=true,use_prepare_varlen=true;
%s
  std::cout << b << " " << local << " " << (tile_count_semaphore_offset+int(scheduler_needs_semaphore))
            << " " << head_swizzle_offset << " " << tile_count_semaphore_offset << "\\n";
  }
}
}""" % expressions
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp)
    (path/"shape.cc").write_text(program)
    subprocess.run(["c++","-std=c++17","-O2",str(path/"shape.cc"),"-o",str(path/"shape")],check=True)
    result=subprocess.check_output([str(path/"shape")],text=True)
records=[]
for line in result.splitlines():
    batch,local,words,head,semaphore=map(int,line.split())
    records.append(dict(batch=batch,windowed=bool(local),native_int32_words=words,head_swizzle_offset=head,semaphore_offset=semaphore))
assert all(records[i]["native_int32_words"]>records[i+1]["native_int32_words"] for i in (0,2,4,6))
print(json.dumps(dict(FA3_revision="f89bc2306632d1ec5f97b014dded4254f5b4a907",flash_api_sha256=EXPECTED,rows=records,
    finding="The full and windowed consumers require distinct sort, swizzle and semaphore storage even at equal static split counts."),indent=2))
