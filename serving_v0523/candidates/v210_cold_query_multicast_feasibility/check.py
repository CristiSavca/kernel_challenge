"""Replay native Hopper block bounds and inspect the pinned FA3 packaging and launch ABI."""

import ast
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
PIN = "f89bc2306632d1ec5f97b014dded4254f5b4a907"
SGL = "297e54fd105e974cdf401e5b4d20253a22c16f90"
SG = HERE.parents[1] / "sglang"
ORIGIN = f"https://raw.githubusercontent.com/sgl-project/sgl-attn/{PIN}"


def curl(url):
    return subprocess.check_output(["curl", "-fsSL", "--max-time", "25", url])


def sgl(path):
    return subprocess.check_output(["git", "-C", SG, "show", f"{SGL}:python/sglang/{path}"]).decode()


def wheel_inventory():
    # Inspect the archive index by byte range without downloading the binary payload.
    meta = json.loads(curl("https://pypi.org/pypi/sglang-kernel/0.4.9/json"))
    wheel = next(item for item in meta["urls"] if item["filename"].endswith("x86_64.whl"))
    def read_range(lo, hi):
        data = subprocess.check_output(["curl", "-fsSL", "--max-time", "25",
                "--max-filesize", str(hi-lo+1), "-r", f"{lo}-{hi}", wheel["url"]])
        assert len(data) == hi-lo+1
        return data
    end = wheel["size"]
    tail = read_range(end-65536,end-1)
    marker = tail.rindex(b"PK\x05\x06")
    _, _, _, entries, size, start = struct.unpack_from("<HHHHII",tail,marker+4)
    index = read_range(start,start+size-1)
    offset = 0
    names = []
    while offset < size:
        assert index[offset:offset+4] == b"PK\x01\x02"
        n,extra,comment = struct.unpack_from("<HHH",index,offset+28)
        names.append(index[offset+46:offset+46+n].decode())
        offset += 46+n+extra+comment
    assert offset == size and len(names) == entries
    ops = [n for n in names if n.rsplit("/",1)[-1].startswith("flash_ops") and n.endswith(".so")]
    assert ops and not any(n.endswith("mainloop_fwd_sm90_tma_gmma_ws.hpp") for n in names)
    return dict(filename=wheel["filename"],sha256=wheel["digests"]["sha256"],
                archived_paths=entries,range_read_bytes=65536+size,native_binaries=ops,
                contains_hopper_mainloop=False)


def sources():
    tree = json.loads(curl(f"https://api.github.com/repos/sgl-project/sgl-attn/git/trees/{PIN}?recursive=1"))
    available = {row["path"] for row in tree["tree"] if row["type"] == "blob"}
    seen = {}
    todo = {"hopper/flash_fwd_launch_template.h"}
    while todo:
        paths = sorted(todo)
        with ThreadPoolExecutor(6) as pool:
            seen.update(zip(paths, pool.map(lambda p: curl(f"{ORIGIN}/{p}"), paths)))
        deps = set()
        for p in paths:
            for name in re.findall(rb'^\s*#\s*include\s*"([^"]+)"', seen[p], re.MULTILINE):
                dep = str(PurePosixPath(p).parent / name.decode())
                if dep in available:
                    deps.add(dep)
        todo = deps - seen.keys()
    return seen


def block_ranges(header, cold):
    # Reuse the existing host definitions of the CUDA integer helpers.
    fixture = HERE.parent / "v146_zero_copy_window_prefill/independent_fa3.py"
    prog = next(node.value.value for node in ast.parse(fixture.read_text()).body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "program")
    prefix = prog[:prog.index("template<bool Packed,bool Split>")]
    body = """
int main() {
  using Bounds = flash::BlockMN<Info,192,160,true,false,true,false>;
  int q,k;
  while (std::cin >> q >> k) {
    for (int m=0; m<cute::ceil_div(q*8,192); ++m) {
      auto [lo,hi,off] = Bounds::get_n_block_min_max({k,q},m,0,0,1,k-1,0,{0},{8});
      std::cout << q << " " << k << " " << m << " " << lo << " " << hi << " " << off << "\\n";
    }
  }
}
"""
    with tempfile.TemporaryDirectory(prefix="okc-v210-host-") as raw:
        path = Path(raw)
        (path / "block.h").write_bytes(header)
        src = path / "range.cpp"
        src.write_text(prefix + body)
        exe = path / "range"
        subprocess.run(["c++", "-std=c++17", "-O2", str(src), "-o", str(exe)], check=True)
        inp = "".join(f"{row['q']} {row['k']}\n" for row in cold)
        return [tuple(map(int,line.split())) for line in
                subprocess.check_output([exe], input=inp, text=True).splitlines()]


def account(cold, rows):
    chunks = []
    witness = None
    for row in cold:
        q, k = row["q"], row["k"]
        tiles = [(m, lo, hi, off) for q1,k1,m,lo,hi,off in rows if (q1,k1)==(q,k)]
        assert all(lo == 0 and off == 0 for _,lo,hi,off in tiles)
        assert [x[0] for x in tiles] == list(range((q+23)//24))
        before = 2 * sum(hi-lo for _,lo,hi,_ in tiles)
        assert before == row["full_packed_unpacked_swa_packed_unpacked_tile_pairs"][0]
        pairs = list(zip(tiles[::2], tiles[1::2]))
        matched = [a for a,b in pairs if a[1:]==b[1:]]
        saved = 2 * sum(hi-lo for _,lo,hi,_ in matched)
        for a,b in pairs:
            if a[1:] != b[1:] and witness is None:
                witness = dict(q=q,k=k,m_blocks=[a[0],b[0]],first_k_tiles=[a[2]-1,b[2]-1])
        chunks.append(dict(q=q,k=k,pairs_per_head=len(pairs),matched_per_head=len(matched),
                           unpaired_m_tiles_per_head=len(tiles)%2,live_kv_tile_reads=before,
                           saved_kv_tile_reads=saved))
    return chunks,witness


def main():
    src = sources()
    launch = src["hopper/flash_fwd_launch_template.h"].decode()
    native = src["hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"].decode()
    assert "(kHeadDim == 192)) && !Is_causal && !Is_local && !Split && !PagedKVNonTMA && !Varlen" in launch
    assert "VarlenDynamicPersistentTileScheduler" in launch
    assert all(t in native for t in ("mcast_mask_kv", "get_slice(cluster_local_block_id.x)",
               "convert_type_out", "softmax.finalize(v_descale)", "permute_output_fp8"))
    cmake = sgl("kernels/aot/CMakeLists.txt")
    assert PIN in cmake and 'install(TARGETS flash_ops LIBRARY DESTINATION "sgl_kernel")' in cmake
    backend = sgl("srt/layers/attention/flashattention_backend.py")
    assert "from sgl_kernel.flash_attn import" in backend
    assert "from sgl_kernel import flash_ops" in sgl("kernels/aot/python/sgl_kernel/flash_attn.py")
    assert "FP8 is only supported on SM100" in sgl("kernels/ops/attention/flash_attn/cute/interface.py")
    assert "self.cluster_shape_mn = (1, 1)" in sgl("kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py")
    listed = subprocess.check_output(["git","-C",SG,"ls-tree","-r","--name-only",SGL],text=True).splitlines()
    assert not any(p.endswith("/mainloop_fwd_sm90_tma_gmma_ws.hpp") for p in listed)
    v170 = json.loads((HERE.parent / "v170_long_prefill_attention_feasibility/check.json").read_text())
    block = src["hopper/block.h"]
    assert sha256(block).hexdigest() == v170["source_sha256"]["block.h"]
    chunks,witness = account(v170["cold"], block_ranges(block,v170["cold"]))
    saved = sum(x["saved_kv_tile_reads"] for x in chunks)
    all_reads = sum(x["live_kv_tile_reads"] for x in chunks)
    total_bytes = sum(map(len,src.values()))
    result = dict(
        native_pin=PIN,sglang_pin=SGL,wheel=wheel_inventory(),chunks=chunks,incompatible_pair=witness,
        operand_reads_eliminated_fraction=saved/all_reads,
        ideal_full_layer_operand_load_only_factor=all_reads/(all_reads-saved),
        all_full_layers_saved_logical_bytes=saved*16*160*64*2,
        all_full_layers_before_logical_bytes=all_reads*16*160*64*2,
        header_closure_files=len(src),header_closure_verbatim_bytes=total_bytes,
        header_closure_sha256={p:sha256(b).hexdigest() for p,b in sorted(src.items())},
        v206_export_bytes=(HERE.parent / "v206_steady_decode_gate_up_pipeline/patch.diff").stat().st_size,
        serving_ttft_first_p90_s=v170["one_user"]["ttft_first_p90_s"],
        hardware_speedup=None,conclusion="native public ABI cannot select this cluster schedule")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
