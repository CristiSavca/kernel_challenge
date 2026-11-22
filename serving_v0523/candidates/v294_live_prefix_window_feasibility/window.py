"""Test the actual released FA3 left boundary and both V291 cache-pack entry points."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from urllib.request import urlopen

import torch


HERE = Path(__file__).resolve().parent
CAND = HERE.parent / "v291_mixed_ranked_attention"
SHA = "f89bc2306632d1ec5f97b014dded4254f5b4a907"


def sha(blob):
    return sha256(blob).hexdigest()


def archive_file(patch, path):
    sect = next(x for x in patch.split("diff --git ") if x.startswith(f"a/{path} b/{path}\n"))
    assert "new file mode" in sect and sect.count("\n@@") == 1
    added = [s[1:] for s in sect.splitlines() if s.startswith("+") and not s.startswith("+++")]
    assert f"@@ -0,0 +1,{len(added)} @@" in sect
    return "\n".join(added) + "\n"


def packing():
    validation = json.loads((CAND / "validation.json").read_bytes())
    patch = (CAND / "patch.diff").read_bytes()
    assert sha(patch) == validation["patch_sha256"]
    path = "python/sglang/srt/layers/attention/direct_prompt_kv.py"
    src = archive_file(patch.decode(), path)
    batched = archive_file(patch.decode(), "python/sglang/kernels/ops/attention/batched_prompt_kv.py")
    abi = json.loads((CAND / "attention_abi.json").read_bytes())
    prior = json.loads((HERE.parent / "v223_batched_direct_prompt/interpreter.json").read_bytes())
    assert sha(src.encode()) == abi["sources"][path]
    assert sha(batched.encode()) == prior["source_sha256"]
    assert "halo = min(prefix, width)" in src and "min(p, left)" in src
    assert "qlens + (lengths - qlens).clamp(max=left)" in src
    assert "old = row < halo" in batched and "(length - used + row), old" in batched
    assert "old[:, None], 0)" in batched
    target = next(n for n in ast.parse(src).body if getattr(n, "name", None) == "complete_direct_prompt_kv")
    env = {"torch": torch}
    exec(compile(ast.Module([target], []), path, "exec"), env)
    torch.set_num_threads(1)
    cases = rows = 0
    for p in (0, 1, 150, 2046, 2047, 2048, 16385, 65536, 114688):
        for q in (1, 15, 4096, 16384):
            for left in (0, 2047, -1):
                # Full-attention control checks every key for small prefixes.
                if left < 0 and p > 2048:
                    continue
                halo = p if left < 0 else min(p, left)
                tables = torch.full((1, p + q), -9999, dtype=torch.int32)
                indices = (torch.arange(halo, dtype=torch.int32) * 7 + 3) % (halo + 13)
                tables[0, p-halo:p] = indices
                base = torch.arange((halo+13)*128, dtype=torch.int32).to(torch.uint8).reshape(-1,1,2,64)
                caches = (base, base.bitwise_xor(173))
                dests = [torch.full((halo+q,2,64), val, dtype=torch.uint8) for val in (17, 97)]
                lengths = torch.tensor([p+q],dtype=torch.int32)
                k, v, out = env[target.name](SimpleNamespace(), (*dests, p, halo), *caches, tables, lengths)
                assert out.tolist() == [halo+q]
                for got, source, val in zip((k[0],v[0]), caches, (17,97)):
                    assert torch.equal(got[:halo], source[indices.long(), 0]) and (got[halo:] == val).all()
                cases += 1
                rows += halo
    return dict(v291_tree=validation["tree"], direct_source=sha(src.encode()),
                host_torch=str(torch.__version__), direct_cases=cases, cached_kv_rows_checked=rows,
                batched_source=sha(batched.encode()), byte_identical_v223_triton_interpreter_cases=prior["interpreter_cases"],
                byte_identical_v223_triton_interpreter_bytes=prior["checked_kv_bytes"])


def native():
    files = {}
    for name in ("block.h", "tile_size.h", "mainloop_fwd_sm90_tma_gmma_ws.hpp", "paged_kv.h"):
        with urlopen(f"https://raw.githubusercontent.com/sgl-project/sgl-attn/{SHA}/hopper/{name}", timeout=20) as resp:
            files[name] = resp.read()
    prior = json.loads((HERE.parent / "v170_long_prefill_attention_feasibility/check.json").read_bytes())["source_sha256"]
    for name in ("block.h", "tile_size.h"):
        assert sha(files[name]) == prior[name]
    mainloop, paged = files["mainloop_fwd_sm90_tma_gmma_ws.hpp"], files["paged_kv.h"]
    assert b"seqlen_info.seqlen_k - n_offset, seqlen_info.leftpad_k + n_offset" in mainloop
    assert b"row_idx + leftpad_k" in paged
    assert b"seqlen_info.offset_k + n_offset, _0{}, _0{}), mK_TMA" in mainloop
    assert b"_0{}, seqlen_info.offset_k + n_offset, _0{}), mVt_TMA" in mainloop
    base = Path(tempfile.mkdtemp(prefix="okc-v294-bounds-")).resolve()
    try:
        for name in ("block.h", "tile_size.h"):
            (base / name).write_bytes(files[name])
        subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Werror", "-O2",
                        "-I", base, HERE/"window.cc", "-o", base/"bounds"], check=True)
        output = subprocess.check_output([base/"bounds"], text=True)
    finally:
        members = sorted(base.iterdir())
        assert all(p.is_file() and not p.is_symlink() and p.name in ("block.h","tile_size.h","bounds") for p in members)
        for name in ("block.h", "tile_size.h"):
            if (base/name).exists():
                assert (base/name).read_bytes() == files[name]
        assert subprocess.run(["git","-C",base,"rev-parse","--is-inside-work-tree"],capture_output=True).returncode != 0
        assert subprocess.run(["lsof","-nP","+D",base],capture_output=True).returncode == 1
        retired = {p.name:p.stat().st_size for p in members}
        before = shutil.disk_usage(base).free
        for p in members:
            p.unlink()
        base.rmdir()
        after = shutil.disk_usage(base.parent).free
        if len(sys.argv) > 1 and sys.argv[1] == "--ledger":
            ledger=dict(removed_path=str(base), retired_files_bytes=retired, git_worktree=False,
                        open_files=False,free_bytes_before=before,free_bytes_after=after,
                        observed_volume_free_change=after-before, remaining=base.exists())
            (HERE/"cleanup.json").write_text(json.dumps(ledger,indent=2)+"\n")
    configs = []
    for line in output.splitlines():
        m,n,pack,paged,tests,right = map(int, line.split())
        configs.append(dict(tile=[m,n], pack_gqa=bool(pack), paged_non_tma=bool(paged),
                            split_boundaries=tests, expired_left_rows=0,
                            max_padded_right_rows_before_predication=right))
    return dict(sgl_attn_commit=SHA, sources={n:sha(f) for n,f in files.items()}, configs=configs,
                total_split_boundaries=sum(c["split_boundaries"] for c in configs))


print(json.dumps(dict(packing=packing(), native=native(), physical_gpu=False), indent=2))
