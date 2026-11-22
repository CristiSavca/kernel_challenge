"""Host-compile the shipped token-major A loader and epilogue, including indirect controls."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
OLDER = HERE.parent / "v261_fused_decode_route_staging"
ABI = runpy.run_path(str(HERE.parent / "v191_decode_gate_up_epilogue/independent_epilogue.py"))
HEADER = "python/sglang/kernels/jit/csrc/muse_spark/decode_gate_up.cuh"


def shipped(folder):
    diff = (folder / "patch.diff").read_bytes()
    lines = next(p for p in diff.decode().split("diff --git ")
                 if p.startswith(f"a/{HEADER} b/{HEADER}\n")).splitlines()
    start = next(i for i, v in enumerate(lines) if v.startswith("@@ "))
    count = int(re.fullmatch(r"@@ -0,0 \+1,(\d+) @@", lines[start])[1])
    assert len(lines[start + 1:]) == count and all(v.startswith("+") for v in lines[start + 1:])
    return "\n".join(v[1:] for v in lines[start + 1:]) + "\n"


LOADER = r"""
template<class C> class G2SMemoryLoaderA;
constexpr unsigned kMuseGateK = 128;
struct Access { int4* destination; void* barrier; uint32_t col, row; };
std::vector<Access> reads;
static const CUtensorMap* expected_map;
void tma_load_2d(const CUtensorMap* m, int4* dest, void* bar, uint32_t x, uint32_t y) {
  assert(m == expected_map);
  reads.push_back({dest, bar, x, y});
}
"""

FIXTURE = r"""
uint16_t pattern(unsigned token, unsigned col, unsigned band, unsigned side, unsigned expert) {
  return static_cast<uint16_t>(3389 * token + 127 * col + 2969 * band + 1941 * side + 787 * expert);
}
uint16_t expected(unsigned token, unsigned col, unsigned expert, uint16_t factor) {
  auto g = pattern(token, col % 128, col / 128, 0, expert);
  auto u = pattern(token, col % 128, col / 128, 1, expert);
  float scale = __bfloat162float(factor);
  auto scaled = [scale](uint16_t v) {
    return __bfloat162float(__float2bfloat16_rn(__bfloat162float(v) * scale));
  };
  float x = scaled(g), y = scaled(u);
  return __float2bfloat16_rn((x / (1.f + expf(-x))) * y);
}

int main() {
  uint64_t words=0, guards=0, addresses=0, duplicate_rows=0, multi_tiles=0, partial=0, empty=0, cases=0;
  constexpr unsigned M=MUSE_GATE_M, I=HUMMING_SHAPE_N/2, E=4;
  const uint16_t factors[E] = {0x8000, 0x3e81, 0xc110, 0x3f43};
  for (unsigned tokens=1; tokens<=8; ++tokens) for (unsigned k: {2u,4u,8u,16u})
    for (unsigned mode=0; mode<4; ++mode) for (bool direct: {false,true}) {
      const unsigned slots=tokens*k; unsigned shift=0, logk=0;
      for (unsigned tmp=slots-1; tmp; tmp>>=1) ++shift;
      for (unsigned tmp=k; tmp>1; tmp>>=1) ++logk;
      std::vector<int32_t> packed(slots);
      for (unsigned j=0;j<slots;++j) {
        unsigned b = mode==0 ? 0 : mode==1 ? 2 : mode==2 ? (j*7+j/k+3)%(E+1) : 1+(j/k)%E;
        packed[j]=(b<<shift)+j;
      }
      std::sort(packed.begin(),packed.end());
      std::vector<int4> out(slots*I/8+4);
      for (auto& v: out) for (auto& c: v.v) c=0xbeef;
      MuseGateCtx ctx{}; CUtensorMap map{};
      ctx.params.c=out.data()+2; ctx.params.bs2=factors; ctx.params.expert_ids_ptr=packed.data();
      ctx.params.top_k=shift | (direct ? logk<<16 : 0); ctx.params.a=&map;
      expected_map=&map;
      EpilogueArithmetic<MuseGateCtx> ar;
      EpilogueGmemWriter<MuseGateCtx,EpilogueArithmetic<MuseGateCtx>> write(ctx,ar);
      for (unsigned e=0;e<E;++e) {
        unsigned lo=std::lower_bound(packed.begin(),packed.end(),int((e+1)<<shift))-packed.begin();
        unsigned end=std::lower_bound(packed.begin(),packed.end(),int((e+2)<<shift))-packed.begin();
        if (lo==end) ++empty;
        if (end-lo>M) ++multi_tiles;
        for (unsigned start=lo;start<end;start+=M) {
          unsigned live=std::min(M,end-start); if (live<M) ++partial;
          alignas(16) int4 scratch[M*16]{}; int barrier=0;
          for (unsigned lane=0; lane<4; ++lane) {
            threadIdx.x=lane; reads.clear(); G2SMemoryLoaderA<MuseGateCtx> load(ctx);
            load.seek(0,3,0,start); load.template load<false>(scratch,&barrier);
            load.load(scratch,&barrier); load.load(scratch,&barrier);
            assert(reads.size()==(lane<2 ? 3 : 0));
            for (unsigned i=0; i<reads.size(); ++i) {
              auto& v=reads[i]; assert(v.destination==scratch+lane*M*8 && v.barrier==&barrier);
              assert(v.col==(3u+(i==2))*128+lane*64 && v.row==(direct ? 0 : start)); ++addresses;
            }
          }
          for (unsigned band=0;band<I/128;++band) {
            for (unsigned row=0;row<M;++row) for (unsigned col=0;col<256;++col) {
              unsigned token=direct ? row : (row<live ? (unsigned(packed[start+row]) % (1u<<shift))/k : 99);
              unsigned linear=col/64*M+row;
              unsigned dst=linear*8+(((col%64)/8)^((linear+offsetof(SharedStorageType,reduce)/128)%8));
              ctx.smem.reduce[dst].v[col%8]=pattern(token,col%128,band,col/128,e);
            }
            write.seek(0,band,end,start);
            for (threadIdx.x=0;threadIdx.x<512;++threadIdx.x) write.write(0,1,0);
          }
        }
      }
      for (unsigned r=0; r<slots; ++r) {
        unsigned bucket=unsigned(packed[r])>>shift, token=(unsigned(packed[r])%(1u<<shift))/k;
        if (r && bucket && (packed[r-1]>>shift)==int(bucket) &&
            (unsigned(packed[r-1])%(1u<<shift))/k==token) ++duplicate_rows;
        for (unsigned c=0;c<I;++c) {
          auto value=out[2+r*I/8+c/8].v[c%8];
          if (bucket) { assert(value==expected(token,c,bucket-1,factors[bucket-1])); ++words; }
          else { assert(value==0xbeef); ++guards; }
        }
      }
      for (auto p: {size_t(0),size_t(1),out.size()-2,out.size()-1})
        for (auto v: out[p].v) { assert(v==0xbeef); ++guards; }
      ++cases;
    }
  std::printf("%llu %llu %llu %llu %llu %llu %llu %llu\n",(unsigned long long)cases,
              (unsigned long long)words,(unsigned long long)guards,(unsigned long long)addresses,
              (unsigned long long)duplicate_rows,(unsigned long long)multi_tiles,
              (unsigned long long)partial,(unsigned long long)empty);
}
"""


def main():
    original = shipped(OLDER)
    actual = shipped(HERE)
    epilogue = actual[actual.index("template <> class EpilogueGmemWriter"):]
    old = original[original.index("template <> class EpilogueGmemWriter"):]
    assert ABI["function"](old, "CUDA_INLINE static float scaled") == ABI["function"](epilogue, "CUDA_INLINE static float scaled")
    epilogue, n = re.subn(r'asm\("mul\.rn\.f32[^;]+;[^\n]+\);',
                          "product = __bfloat162float(value) * scale;", epilogue)
    assert n == 1
    loader = ABI["function"](actual, "template <> class G2SMemoryLoaderA") + ";\n"
    prelude = ABI["CPP"].replace("uint32_t top_k;", "uint32_t top_k; const void* a;")
    prelude = prelude.replace("SharedStorageType smem;", "uint32_t load_thread_id(){return threadIdx.x;} SharedStorageType smem;")
    records = []
    with tempfile.TemporaryDirectory() as folder:
        src = Path(folder)/"oracle.cpp"
        binary = Path(folder)/"oracle"
        src.write_text(prelude + LOADER + loader + epilogue + FIXTURE)
        for width, pad in ((128,1),(384,7),(4096,3)):
            subprocess.run(["c++","-std=c++17","-O2","-fno-strict-aliasing","-DMUSE_GATE_M=8",
                            f"-DHUMMING_SHAPE_N={2*width}", f"-DPAD={pad}",str(src),"-o",str(binary)],check=True)
            values = list(map(int,subprocess.check_output([binary], text=True).split()))
            assert all(values)
            records.append(dict(width=width,shared_padding_128B=pad,**dict(zip(
                ("cases","live_bf16","guard_words","TMA_coordinates","duplicate_source_rows",
                 "multitile_experts","partial_tiles","empty_experts"),values,strict=True))))
    return dict(patch_sha256=hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest(),
                header_sha256=hashlib.sha256(actual.encode()).hexdigest(),cases=records,
                physical_H200=False,scope="Host-compiled production A-loader and epilogue. TMA intrinsic records addresses, WGMMA fragments are populated independently, CPU expf and software BF16 rounding.")


if __name__ == "__main__":
    print(json.dumps(main(),indent=2))
