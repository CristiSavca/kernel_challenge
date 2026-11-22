"""Run the pinned FA3 layout/TMA selectors and SGLang's BCG replay-metadata wrapper."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from dataclasses import dataclass
from unittest.mock import patch
from urllib.request import urlopen


SHA = "f89bc2306632d1ec5f97b014dded4254f5b4a907"


def cpp_function(source, name):
    start = source.index(name)
    start = source.rindex("\n", 0, start) + 1
    begin = source.index("{", start)
    depth = 1
    for i in range(begin + 1, len(source)):
        depth += (source[i] == "{") - (source[i] == "}")
        if not depth:
            return source[start:i + 1]
    raise ValueError(name)


def native():
    files = {}
    for name in ("flash_api.cpp", "tile_size.h", "heuristics.h"):
        with urlopen(f"https://raw.githubusercontent.com/sgl-project/sgl-attn/{SHA}/hopper/{name}", timeout=15) as response:
            files[name] = response.read().decode()
    api = files["flash_api.cpp"]
    formulas = [next(line.strip() for line in api.splitlines() if line.strip().startswith("int const " + key + " ="))
                for key in ("max_num_pages_per_seq", "num_pages", "page_size", "seqlen_k", "total_k")]
    source = """
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <tuple>
struct Flash_fwd_params {
    int arch=90, d=64, d_rounded=64, dv_rounded=64, page_size=1, seqlen_q=512, h=16, h_k=2;
    bool page_table=true, leftpad_k=false, knew_ptr=false, is_local=false, is_causal=true, is_e4m3=true;
    float softcap=0;
};
struct TensorShape {
    int64_t first, second;
    int64_t size(int i) const { return i == 0 ? first : second; }
};
"""
    source += cpp_function(files["tile_size.h"], "tile_size_fwd_sm90(") + "\n"
    source += cpp_function(files["heuristics.h"], "use_one_mma_wg(") + "\n"
    source += cpp_function(api, "get_pagedkv_tma(") + "\n"
    source += """
int main() {
    int eligibility=0, shapes=0;
    for (int arch : {80, 90}) for (int dimq : {1, 8, 24, 25, 512, 16384})
    for (int ratio : {1, 2, 8, 16}) for (int page : {1, 128, 160, 320})
    for (int mode=0; mode<5; ++mode) {
        Flash_fwd_params p;
        p.arch=arch; p.seqlen_q=dimq; p.h=ratio*p.h_k; p.page_size=page;
        p.leftpad_k=mode==1; p.knew_ptr=mode==2; p.is_local=mode==3; p.page_table=mode!=4;
        bool expected = arch==90 && mode==0 && page%160==0 && dimq*ratio>192;
        assert(expected==get_pagedkv_tma(p)); ++eligibility;
    }
    for (int capacity : {511, 640, 3430401, 3456001}) for (int residue : {0, 1, 47, 159})
    for (int batch_size : {1, 2, 4, 32, 128, 512}) for (int width : {512, 639, 640, 16384, 74000})
    for (int layout=0; layout<3; ++layout) for (bool highest_in_last_page : {true, false}) {
        if (int64_t(batch_size)*width>262144 || width>capacity) continue;
        bool paged_KV=layout!=0, is_varlen_k=false;
        int physical_pages=(capacity-residue)/160;
        int highest_page=highest_in_last_page ? physical_pages-1 : physical_pages/2;
        int highest=residue+(highest_page+1)*160-1;
        int coarse_count=(highest-residue+1)/160;
        assert(coarse_count<=physical_pages && highest<capacity);
        TensorShape k=layout==0?TensorShape{batch_size,width}:layout==1?TensorShape{capacity,1}:TensorShape{coarse_count,160};
        TensorShape page_table{batch_size, layout==2?(width+159)/160:width};
        std::optional<int> max_seqlen_k_;
"""
    source += "\n".join(formulas)
    source += """
        assert(total_k == batch_size*(layout==0?width:layout==1?1:160));
        assert(num_pages==(layout==0?0:layout==1?capacity:coarse_count));
        assert(seqlen_k>0 && seqlen_k<=width+159 && seqlen_k>=width);
        ++shapes;
    }
    std::cout << eligibility << " " << shapes;
}
"""
    with tempfile.TemporaryDirectory(prefix="okc-v143-independent-") as path:
        root = Path(path)
        file = root / "check.cpp"
        file.write_text(source)
        subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(file),
                        "-o", str(root / "check")], check=True)
        eligibility, shapes = map(int, subprocess.check_output([str(root / "check")], text=True).split())
    return dict(geometry=eligibility, native_shape_equations=shapes,
                abi_source_sha256={f: hashlib.sha256(s.encode()).hexdigest() for f, s in files.items()})


def replay(stage):
    path = "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
    text = subprocess.check_output(["git", "-C", stage, "show", "HEAD:" + path], text=True)
    method = next(node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.FunctionDef)
                  and node.name == "_prepare_forward_metadata_for_replay")
    scope = {}
    exec("from __future__ import annotations\n" + ast.unparse(method), scope)
    prepare = scope[method.name]
    calls = []
    backend = SimpleNamespace(skips_prefill_metadata_on_idle_dp_rank=False)
    backend.init_forward_metadata = lambda fb: calls.append(("live", fb))
    backend.resolve_prefill_shared_read_ends = lambda fb, num_qo_tokens, build_snapshot: (fb, num_qo_tokens, build_snapshot)
    obj = SimpleNamespace(model_runner=SimpleNamespace(attn_backend=backend), _is_full_backend=False,
                          use_captured_attn_metadata=False)
    stale = object()
    count = 0
    for iteration in range(8):
        for bucket in (512, 1024, 16384):
            first, second = object(), object()
            for live in (first, second, first):
                result = prepare(obj, live, stale, SimpleNamespace(size=bucket))
                assert result == (live, bucket, True)
                assert calls[-1] == ("live", live)
                count += 1
    return dict(bcg_live_refreshes=count, prefill_runner_sha256=hashlib.sha256(text.encode()).hexdigest())


def stream_lifetime(stage):
    import torch
    source = (stage / "python/sglang/srt/layers/attention/prefill_kv_span.py").read_text()
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name=="PrefillKVSpan")
    env = dict(torch=torch, dataclass=dataclass, _PAGE=160)
    exec(ast.unparse(cls), env)
    span_cls = env[cls.name]
    backing = torch.arange(160 * 11 * 128, dtype=torch.int32).byte().view(torch.float8_e4m3fn)
    key = backing.view(-1, 1, 2, 64)
    value = key.clone()
    records = checks = 0
    for inference in (False, True):
        with torch.inference_mode(inference):
            # Selected page IDs may skip or alias physical tiles between requests.
            for residue in (0, 1, 47, 159):
                indices = torch.tensor([[5, 3, 2], [3, 2, 5]], dtype=torch.int32)
                table = (residue + 160 * indices[:, :, None] + torch.arange(160)).flatten(1)
                lens = torch.tensor([480, 480], dtype=torch.int32)
                highest = int(table.max())
                host = torch.tensor([0, 1, int(table[0, 0]), 0, 480, highest], dtype=torch.int64)
                recorded = []
                pages = SimpleNamespace(is_cuda=True, device="tested_device", record_stream=recorded.append)
                event = SimpleNamespace(query=lambda: False)
                span = span_cls(table, lens, 512, pages, host, event)
                with patch.object(torch.cuda, "current_stream", side_effect=lambda device: stream):
                    stream = object()
                    assert span.view(key, value, table, lens, 512)[2] is table
                    event.query = lambda: True
                    for _ in range(3):
                        stream = object()
                        k, v, result = span.view(key, value, table, lens, 512)
                        assert result is pages and recorded[-1] is stream
                        assert k.shape[0] == v.shape[0] == int(indices.max()) + 1
                        for i in range(2):
                            old = key.view(torch.uint8)[table[i].long(), 0]
                            new = k.view(torch.uint8)[indices[i].long()].flatten(0, 1)
                            assert torch.equal(old, new)
                        checks += 1
                    assert span.view(key, value, table.clone(), lens, 512)[2] is not pages
                    assert span.view(key, value, table, lens.clone(), 512)[2] is not pages
                    if not inference:
                        table[0, 0] += 1
                        assert span.view(key, value, table, lens, 512)[2] is table
                    records += len(recorded)
                    assert len(recorded) == 3
    return dict(consumer_stream_records=records, interleaved_view_checks=checks,
                span_source_sha256=hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("stage", type=Path)
    stage = args.parse_args().stage
    print(json.dumps({**native(), **replay(stage), **stream_lifetime(stage), "sgl_attn_commit": SHA,
                      "physical_gpu": False}, indent=2))
