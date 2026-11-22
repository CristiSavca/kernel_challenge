"""Exercise the private sliding-window cache through the production page-first backup functions."""
import argparse
import ctypes
import importlib.util
import json
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
HEADER = "python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh"

COPY_ABI = r"""
extern "C" size_t backup(int grouped, void* sk, void* sv, void* hk, void* hv,
                         const int64_t* host_rows, int64_t count, int64_t row_words) {
  using tvm::ffi::TensorView;
  std::vector<TensorView> src{{sk, row_words, 2}, {sv, row_words, 2}};
  std::vector<TensorView> dst{{hk, row_words, 2}, {hv, row_words, 2}};
  assert(pending.empty());
  auto batched = grouped ? candidate::try_copy_page_first_pages_batch
                         : original::try_copy_page_first_pages_batch;
  auto regular = grouped ? candidate::copy_page_first_pages_fallback
                         : original::copy_page_first_pages_fallback;
  if (!batched(src, dst, host_rows, count, 1, device_id, stream_id)) {
    regular(src, dst, host_rows, count, 1, stream_id);
  }
  size_t descriptors = pending.size();
  for (const auto& transfer : pending) std::memcpy(transfer.dst, transfer.src, transfer.bytes);
  pending.clear();
  return descriptors;
}
"""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture_restored_planes(stage):
    owner = load_module("windowed_draft", HERE.parent / "v249_private_draft_swa/replay.py")
    original_method = owner.method
    sidecars = []

    def method(stage, path, cls, name, scope):
        function = original_method(stage, path, cls, name, scope)
        if name != "build_swa_draft_pools":
            return function

        def collect(*args, **kwargs):
            spec, entries = function(*args, **kwargs)
            assert len(entries) == 1 and spec[0].indices_from_pool == "swa"
            sidecars.extend(entries)
            return spec, entries

        return collect

    owner.method = method
    prior = owner.run(stage)
    entry, = sidecars
    assert entry.layer_mapping == {i: i for i in range(5)}
    layers = [entry.device_pool.get_kv_buffer(entry.layer_mapping[i]) for i in range(5)]
    # Select data written by the owner-table and accepted-prefix recovery fixture.
    occupied = torch.where((layers[0][0] != -913).flatten(1).any(1))[0]
    assert len(occupied) >= 64
    source_rows = occupied.flip(0)[:64].roll(13)
    planes = [torch.stack([layer[p][source_rows] for layer in layers], dim=1) for p in range(2)]
    return [p.reshape(len(source_rows), 5, -1) for p in planes], prior


def build_native_copy(stage, current, temp):
    host = load_module("host_copy", HERE.parent / "v154_coalesced_hicache_backup/check_copy.py")
    original = subprocess.check_output(["git", "-C", stage, "show", host.BASE + ":" + HEADER]).decode()
    support = (host.HERE / "copy_fixture.cpp").read_text().split("// INSERT STOCK AND CANDIDATE HERE")[0]
    code = (support + "\nnamespace original {\n" + host.body(original)
            + "\n}\nnamespace candidate {\n" + host.body(current) + "\n}\n" + COPY_ABI)
    source = temp / "copy.cc"
    library = temp / "copy.so"
    source.write_text(code)
    subprocess.run(["clang++", "-std=c++20", "-O2", "-DCUDA_VERSION=12090", "-shared", "-fPIC",
                    str(source), "-o", str(library)], check=True)
    copy = ctypes.CDLL(str(library)).backup
    copy.argtypes = [ctypes.c_int, *([ctypes.c_void_p] * 5), ctypes.c_int64, ctypes.c_int64]
    copy.restype = ctypes.c_size_t
    return copy


def check_order(copy, planes, geometry, order):
    count = len(planes[0])
    indices = torch.arange(count, dtype=torch.int64)
    if order == "runs":
        indices += torch.arange(count) // 7
    elif order == "fragmented":
        indices = (indices * 17) % count
    indices += 3
    host = [torch.full((128, *p.shape[1:]), -819.0, dtype=p.dtype) for p in planes]
    expected = [h.clone() for h in host]
    for h, p in zip(expected, planes):
        h[indices] = p

    counts = []
    for grouped in (False, True):
        output = [h.clone() for h in host]
        pointers = [a.data_ptr() for a in (*planes, *output, indices)]
        counts.append(copy(grouped, *pointers, count, planes[0][0].numel()))
        for h, p, wanted in zip(output, planes, expected):
            assert torch.equal(h, wanted)
            # Model reassignment to a new, nonidentity set of device slots on load.
            destination = (torch.arange(count) * 19) % count
            restored = torch.empty_like(p)
            restored[destination] = h[indices]
            assert torch.equal(restored[destination], p)

    runs = 1 + int((indices[1:] != indices[:-1] + 1).sum())
    assert counts == [2 * count, 2 * runs], counts
    return dict(geometry=geometry, order=order, pages=count,
                bytes_per_K_or_V_page=planes[0][0].numel() * planes[0].element_size(),
                parent_copies=counts[0], candidate_copies=counts[1],
                roundtrip_bf16_words=sum(p.numel() for p in planes))


def main(stage):
    manifest = json.loads((HERE / "validation.json").read_bytes())
    assert sha256((HERE / "patch.diff").read_bytes()).hexdigest() == manifest["patch_sha256"]
    assert subprocess.check_output(["git", "-C", stage, "write-tree"]).decode().strip() == manifest["tree"]
    planes, prior = capture_restored_planes(stage)
    current = (stage / HEADER).read_text()
    assert sha256(current.encode()).hexdigest() == manifest["header_sha256"]
    # Expand the projected test values to native H200 K or V row width on this model.
    native_planes = [p.repeat(1, 1, (128 + p.shape[2] - 1) // p.shape[2])[:, :, :128].contiguous()
                     for p in planes]
    with tempfile.TemporaryDirectory() as temp:
        copy = build_native_copy(stage, current, Path(temp))
        cases = [check_order(copy, data, geometry, order)
                 for geometry, data in (("fixture", planes), ("native_width", native_planes))
                 for order in ("consecutive", "runs", "fragmented")]
    return dict(patch_sha256=manifest["patch_sha256"], header_sha256=manifest["header_sha256"],
                draft_source_sha256=prior["source_sha256"],
                owner_replay_words=prior["matched_and_preserved_bf16_words"], cases=cases,
                physical_H200=False,
                scope="Executed the native private-pool and host sidecar builder, then actual page-first C++ "
                      "copy selectors with a queued CPU memcpy ABI. The restore and transport hardware are "
                      "modeled. Descriptor counts assume the indicated host-page permutations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    print(json.dumps(main(parser.parse_args().stage), indent=2))
