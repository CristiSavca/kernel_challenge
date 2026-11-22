"""Execute the staged CUDA branch-address expressions on the host, with bounds-checked loads."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HEADER = "python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh"
PREFIX = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using bf16_t = uint16_t;
const bf16_t* lowest;
const bf16_t* limit;
struct BF16Vec {
  std::array<uint16_t, 8> bits;
  void load(const uint16_t* ptr, int vector_index) {
    const auto* begin = ptr + 8 * vector_index;
    if (begin < lowest || begin + 8 > limit || (begin - lowest) % 8 != 0) std::abort();
    std::memcpy(bits.data(), begin, 16);
  }
};
struct Params { const void* branch; int num_rows; };
struct Axis { int x; };
constexpr int kHidden = 8192, kVec = 8;
uint16_t word(int row, int col) {
  return ((col + uint32_t(row % 8)*8192u) * 2053u + (row / 8)*1711u + 37u) & 65535u;
}
template<int kColumnRanks, int kNumThreads>
uint64_t verify(int rows) {
  const int shard = kHidden / kColumnRanks;
  std::vector<uint16_t> data(size_t(rows)*kHidden);
  for (int r=0; r<kColumnRanks; ++r)
    for (int m=0; m<rows; ++m)
      for (int c=0; c<shard; ++c)
        data[(r*rows+m)*shard+c] = word(m,r*shard+c);
  lowest = data.data();
  limit = lowest + data.size();
  Params params{lowest,rows};
  uint64_t count = 0;
  for (int row=0; row<rows; ++row) {
    Axis blockIdx{row};
    // BASE
    constexpr int kPacks = kHidden / (kNumThreads*kVec);
    for (int t=0; t<kNumThreads; ++t) {
      Axis threadIdx{t};
      for (int p=0; p<kPacks; ++p) {
        // LOAD
        for (int i=0; i<8; ++i) {
          const int expected_col = 8*(t+p*kNumThreads)+i;
          if (branch_v.bits[i] != word(row,expected_col)) std::abort();
          ++count;
        }
      }
    }
  }
  return count;
}
template<int R> void suite() {
  uint64_t counts[4]{};
  for (int rows : {1, 3, 8, 21, 32, 93, 128, 255, 256, 257}) {
    counts[0] += verify<R,128>(rows);
    counts[1] += verify<R,256>(rows);
    counts[2] += verify<R,512>(rows);
    counts[3] += verify<R,1024>(rows);
  }
  std::printf("%d %llu %llu %llu %llu\n", R, (unsigned long long)counts[0],
      (unsigned long long)counts[1], (unsigned long long)counts[2], (unsigned long long)counts[3]);
}
int main() { suite<1>(); suite<4>(); suite<8>(); }
"""


def extracted(source):
    base = source[source.index("const int64_t base ="):source.index("  const auto stream =")]
    load = source[source.index("const int idx = threadIdx.x + p * kNumThreads;"):]
    load = load[:load.index("#pragma unroll")]
    assert "params.branch" in base and "BF16Vec branch_v;" in load and "params.num_rows" in load
    return base, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=Path)
    ap.add_argument("--full-width", type=Path, help="precomputed output from the V228 full_width.py fixture")
    args = ap.parse_args()
    header = (args.stage / HEADER).read_text()
    base, load = extracted(header)
    with tempfile.TemporaryDirectory(prefix="v283-host-address-") as td:
        scratch = Path(td)
        code = scratch / "verify.cpp"
        exe = scratch / "verify"
        code.write_text(PREFIX.replace("// BASE", base).replace("// LOAD", load))
        subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(code), "-o", str(exe)], check=True)
        counts = {}
        for line in subprocess.check_output([str(exe)], text=True).splitlines():
            ranks, *words = list(map(int, line.split()))
            counts[str(ranks)] = dict(zip((128,256,512,1024),words))
    fixture = Path(__file__).resolve().parents[1] / "v228_sharded_verify_projections/full_width.py"
    data = (args.full_width.read_text() if args.full_width else
            subprocess.check_output([sys.executable, "-B", str(fixture)], text=True))
    full = json.loads(data)
    assert full["differing_bf16"] == 0 and full["total_compared"] == 10420224
    print(json.dumps(dict(source=HEADER,
                          header_sha256=sha256(header.encode()).hexdigest(),
                          address_expressions_sha256=sha256((base+load).encode()).hexdigest(),
                          rows=[1,3,8,21,32,93,128,255,256,257], words_by_tp_and_threads=counts,
                          words_total=sum(sum(d.values()) for d in counts.values()),
                          words_mismatched=0, payload_set="all BF16 bit patterns at M>=8",
                          full_width_host_gemm=dict(fixture=fixture.name,
                               fixture_sha256=sha256(fixture.read_bytes()).hexdigest(),
                               rows=[8,32,128,256], tp=[4,8], method=full["method"],
                               dimensions=[[8192,4096],[4096,8192]], torch=full["torch"],
                               word_count=full["total_compared"], word_mismatches=0)),indent=2))


if __name__ == "__main__":
    main()
