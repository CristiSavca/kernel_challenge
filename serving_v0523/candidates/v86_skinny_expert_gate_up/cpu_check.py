"""Execute the patched CUDA CTA with software BF16, warp MMA and barrier instructions."""

import argparse
import ctypes
import json
import pathlib
import subprocess
import tempfile

import numpy as np


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def floats(x):
    return (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)


# The adapter keeps CUDA arithmetic, pointer indexing, shared-memory dataflow,
# launch geometry and stores from the candidate. Coroutines suspend all lanes
# at actual barrier and MMA sites before the instruction simulator proceeds.
PRELUDE = r"""
#include <bit>
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
using bf16_t = uint16_t;
struct float4 { float x, y, z, w; };
struct Index { unsigned x, y; };
Index threadIdx, blockIdx;
#define SGL_DEVICE inline
#define __grid_constant__
#define __shared__ static
#define __launch_bounds__(...)
float __uint_as_float(uint32_t bits) { return std::bit_cast<float>(bits); }
namespace device {
template <typename T, int N> struct alignas(sizeof(T)*N) AlignedVector {
  T values[N];
  T operator[](int i) const { return values[i]; }
};
template <typename T, typename S> T cast(S value) {
  if constexpr (std::is_same_v<T,bf16_t> && std::is_same_v<S,float>) {
    auto bits = std::bit_cast<uint32_t>(value);
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
  } else if constexpr (std::is_same_v<T,float> && std::is_same_v<S,bf16_t>) {
    return std::bit_cast<float>(uint32_t(value) << 16);
  } else { return static_cast<T>(value); }
}
}
using namespace device;
enum class ActivationKind { kSiLU };
template <ActivationKind K> float apply_activation_f32(float x) { return x / (1.0f + expf(-x)); }
namespace muse_expert_scaled_silu {
float scale_and_round(bf16_t x, float g) { return cast<float>(cast<bf16_t>(cast<float>(x) * g)); }
}
struct Op { float4* accum = nullptr; uint32_t a[4] = {}, b[2] = {}; };
struct Routine {
  struct promise_type {
    Op op{};
    Routine get_return_object() {
      return {std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept {return {};}
    std::suspend_always final_suspend() noexcept {return {};}
    std::suspend_always yield_value(Op request) { op = request; return {}; }
    void return_void() {}
    void unhandled_exception() { std::abort(); }
  };
  std::coroutine_handle<promise_type> h;
};
"""

POSTLUDE = r"""
void warp_mma(Routine* lanes) {
  float a[16][16] = {}, b[16][8] = {};
  for (int r = 0; r < 16; ++r) for (int k = 0; k < 16; ++k) {
    auto& op = lanes[(r % 8) * 4 + (k % 8) / 2].h.promise().op;
    uint32_t word = op.a[r / 8 + 2 * (k / 8)];
    a[r][k] = cast<float>(bf16_t(word >> ((k % 2) * 16)));
  }
  for (int k = 0; k < 16; ++k) for (int c = 0; c < 8; ++c) {
    auto& op = lanes[c * 4 + (k % 8) / 2].h.promise().op;
    b[k][c] = cast<float>(bf16_t(op.b[k / 8] >> ((k % 2) * 16)));
  }
  for (int i = 0; i < 32; ++i) {
    auto& d = *lanes[i].h.promise().op.accum;
    float* out[4] = {&d.x, &d.y, &d.z, &d.w};
    for (int f = 0; f < 4; ++f) {
      int r = i / 4 + (f / 2) * 8, c = (i % 4) * 2 + f % 2;
      for (int k = 0; k < 16; ++k) *out[f] += a[r][k] * b[k][c];
    }
  }
}
extern "C" void simulate(const bf16_t* in, bf16_t* out, const uint32_t* weights,
                         const uint8_t* scales, const bf16_t* globals, const int32_t* offsets,
                         uint32_t hidden, uint32_t intermediate, uint32_t experts) {
  using namespace sglang::muse_skinny_gate_up;
  Params p{in,out,weights,scales,globals,offsets,hidden,intermediate};
  for (blockIdx.y=0; blockIdx.y<experts; ++blockIdx.y) {
    for (blockIdx.x=0; blockIdx.x<intermediate/64; ++blockIdx.x) {
      Routine lanes[32*kWarps];
      bool ready[kWarps]={};
      for (auto& lane : lanes) lane=run_kernel(p);
      bool more;
      do {
        more=false; bool all_ready=true;
        for (uint32_t w=0; w<kWarps; ++w) {
          if (ready[w] || lanes[w*32].h.done()) continue;
          for (int l=0; l<32; ++l) {threadIdx.x=w*32+l; lanes[threadIdx.x].h.resume();}
          if (!lanes[w*32].h.done()) {
            if (lanes[w*32].h.promise().op.accum) warp_mma(lanes + w*32);
            else ready[w]=true;
          }
        }
        for (uint32_t w=0; w<kWarps; ++w) {
          more |= !lanes[w*32].h.done();
          all_ready &= ready[w] || lanes[w*32].h.done();
        }
        if (all_ready) for (auto& r : ready) r=false;
      } while(more);
      for (auto& lane : lanes) lane.h.destroy();
    }
  }
}
"""


def load_kernel(folder, checkout):
    header = checkout / "python/sglang/kernels/jit/csrc/muse_spark/expert_skinny_gate_up.cuh"
    body = header.read_text().split('namespace sglang {', 1)[1].split('struct MuseSkinnyGateUp', 1)[0]
    body = body.replace('asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pair) : "f"(x), "f"(y));',
                        'pair = cast<bf16_t>(x) | (uint32_t(cast<bf16_t>(y)) << 16);')
    body = body.replace('void accumulate(', 'Op accumulate(')
    start = body.index('  asm volatile(')
    end = body.index('\n}', start)
    body = body[:start] + '  return Op{&acc,{a0,a1,a2,a3},{b0,b1}};' + body[end:]
    body = body.replace('__global__ __launch_bounds__(32 * kWarps) void', 'Routine')
    body = body.replace('if (first == end) return;', 'if (first == end) co_return;')
    body = body.replace('        accumulate(', '        co_yield accumulate(')
    body = body.replace('__syncthreads();', 'co_yield Op{};')
    src = folder / 'simulate.cpp'
    src.write_text(PRELUDE + 'namespace sglang {\n' + body + '}\n' + POSTLUDE)
    lib = folder / 'simulate.so'
    subprocess.run(['c++', '-std=c++20', '-O2', '-fno-strict-aliasing', '-shared', '-fPIC',
                    str(src), '-o', str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    sim = dll.simulate
    sim.restype = None
    sim.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 3
    return sim


def pack_codes(codes):
    # Forward Humming process.cuh schedule, independently following its nested
    # lane and 8x8 subblock loops and interleave_mode=3 pack permutation.
    experts, n, k = codes.shape
    tiles = codes.reshape(experts, n // 64, 64, k // 16, 16).transpose(0, 3, 1, 2, 4)
    packed = np.zeros((experts, k // 16, n // 64, 32, 4), dtype=np.uint32)
    for lane in range(32):
        for part in range(4):
            entries = []
            for k8 in range(2):
                for n8 in range(2):
                    for sub in range(2):
                        entries.append(tiles[..., part*16 + n8*8 + lane//4, k8*8 + lane%4*2 + sub])
            for i in range(8):
                source = (i % 4) * 2 + (i // 4)
                packed[..., lane, part] |= entries[source].astype(np.uint32) << (4*i)
    return packed


def convert_scales(scale):
    return np.where(scale < 8, 0, (1 + ((scale % 8)/8)) * 2.0**((scale//8).astype(int)-7)).astype(np.float32)


def run_case(sim, rng, hidden, inner, counts, prefix, integral):
    experts = len(counts)
    rows = prefix + sum(counts)
    n = inner * 2
    codes = rng.integers(0, 16, (experts, n, hidden), dtype=np.uint8)
    choices = [0, 56, 64, 72] if integral else [0, 8, 15, 26, 37, 53, 55, 56, 61, 69]
    scales = rng.choice(np.array(choices, dtype=np.uint8), (experts, n, hidden//16))
    global_scales = bf16(rng.choice([0, .25, .5, 1], experts)) if integral else bf16(rng.uniform(.01, 1.5, experts))
    x = bf16(rng.integers(-2, 3, (rows, hidden)) if integral else rng.normal(0, .2, (rows, hidden)))
    packed = pack_codes(codes)
    transposed = scales.transpose(0, 2, 1).copy()
    scale_bank = transposed.reshape(experts, hidden//16, n//64, 8, 8).swapaxes(-1,-2).copy()
    boundaries = np.array([prefix, *(prefix + np.cumsum(counts))], np.int32)
    poison = 0x7FC1
    output = np.full((rows, inner), poison, np.uint16)
    arrays = [x, output, packed, scale_bank, global_scales, boundaries]
    sim(*(a.ctypes.data for a in arrays), hidden, inner, experts)
    assert np.all(output[:prefix] == poison)
    lut = np.array([0,.5,1,1.5,2,3,4,6], np.float32)
    expected = []
    for e,(first,end) in enumerate(zip(boundaries, boundaries[1:])):
        w = np.where(codes[e] & 8, -1., 1.).astype(np.float32) * lut[codes[e] & 7]
        w *= np.repeat(convert_scales(scales[e]), 16, axis=-1)
        v = x[first:end]
        # Match the four independent FP32 stripes and final FP32 combination.
        parts = np.zeros((4,len(v),n), dtype=np.float32)
        for g in range(hidden//16):
            for j in range(16):
                k=g*16+j
                parts[g%4] += floats(v[:,k,None]) * w[None,:,k]
        sums = ((parts[0] + parts[1]) + parts[2]) + parts[3]
        scaled = floats(bf16(floats(bf16(sums)) * floats(global_scales[e])))
        gate,up = np.split(scaled, 2, axis=-1)
        with np.errstate(over='ignore'):
            expected.append(bf16(gate/(1+np.exp(-gate))*up))
    expected = np.concatenate(expected)
    actual = output[prefix:]
    assert np.array_equal(actual,expected), np.argwhere(actual!=expected)[:10]
    return int(actual.size), int(prefix*inner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkout', type=pathlib.Path)
    args = ap.parse_args()
    rng = np.random.default_rng(61186)
    cases = [
        (64,64,[0,0,0,0],7), (128,64,[1,0,0,1],4), (128,128,[2,3,7,8],9),
        (256,128,[0,0,9,18],2), (128,192,[0,17,1,8,3,0,1,7],1),
        (4096,64,[0,1,0],7), (128,4096,[0,1],3), (64,64,[32,0,0,0],0),
    ]
    with tempfile.TemporaryDirectory() as path:
        sim = load_kernel(pathlib.Path(path), args.checkout)
        comparisons = padding = 0
        for integral in [True, False]:
            for params in cases:
                live, untouched = run_case(sim,rng,*params,integral)
                comparisons += live
                padding += untouched
    print(json.dumps(dict(cases=len(cases)*2, compared_bf16_values=comparisons,
                          untouched_prefix_values=padding, kernel_simulated=True),indent=2))


if __name__ == '__main__':
    main()
