// Compiled after the production compact writer and host BF16 primitives.
#include <fstream>
#include <iostream>
#include <iterator>

static uint16_t sample(unsigned position, unsigned channel, bool up) {
  unsigned sequence = position * (HUMMING_SHAPE_N / 2) + channel;
  return static_cast<uint16_t>(sequence * (up ? 2969u : 3389u) + (up ? 11239 : 48731));
}

int main(int argc, char** argv) {
  constexpr unsigned m = MUSE_GATE_M, width = TEST_N, band = width / 2;
  constexpr unsigned intermediate = HUMMING_SHAPE_N / 2, rows = m + 13, first = 7;
  std::vector<int32_t> mapping(m * width);
  std::ifstream input(argv[1], std::ios::binary);
  input.read(reinterpret_cast<char*>(mapping.data()), mapping.size() * sizeof(int32_t));
  assert(argc == 2 && input.gcount() == mapping.size() * sizeof(int32_t));
  std::vector<int32_t> keys(rows, -1);
  const unsigned shift = 19;
  for (unsigned row = first; row < first + m; ++row)
    keys[row] = (4u << shift) | ((row * 135337u) % (1u << shift));
  int4 guard{{0x5213,0xf112,0x3f7e,0x0,0x8001,0x1753,0xf2e1,0x1112}};
  uint64_t words = 0, guards = 0, errors = 0;
  bool seen_gate[65536] = {}, seen_up[65536] = {};
  for (unsigned live : {0u, 1u, m / 2, m - 1, m}) {
    for (uint16_t gain : {0u, 0x8000u, 1u, 0x3e81u, 0xc110u, 0x7f7fu}) {
      uint16_t gains[] = {0x7fc1, 0x7fc1, 0x7fc1, gain};
      std::vector<int4> actual(rows * intermediate / 8 + 4, guard);
      MuseGateCtx ctx{};
      ctx.params = {actual.data() + 2, gains, keys.data(), shift};
      EpilogueArithmetic<MuseGateCtx> arithmetic;
      EpilogueGmemWriter<MuseGateCtx, EpilogueArithmetic<MuseGateCtx>> writer(ctx, arithmetic);
      for (unsigned tile = 0; tile < intermediate / band; ++tile) {
        for (unsigned offset = 0; offset < mapping.size(); ++offset) {
          unsigned logical = mapping[offset], row = logical / width, col = logical % width;
          ctx.smem.reduce[offset / 8].v[offset % 8] = sample(row, tile * band + col % band, col >= band);
        }
        writer.seek(0, tile, first + live, first);
        for (threadIdx.x = 0; threadIdx.x < 512; ++threadIdx.x) writer.write(0, 1, 0);
      }
      for (unsigned vec = 0; vec < actual.size(); ++vec) {
        if (vec < 2 + first * intermediate / 8 || vec >= 2 + (first + live) * intermediate / 8) {
          errors += std::memcmp(&actual[vec], &guard, sizeof(guard)) != 0;
          guards += 8;
          continue;
        }
        unsigned row = (vec - 2) / (intermediate / 8) - first;
        for (unsigned lane = 0; lane < 8; ++lane) {
          unsigned col = (vec - 2) % (intermediate / 8) * 8 + lane;
          auto g = sample(row, col, false), u = sample(row, col, true);
          seen_gate[g] = seen_up[u] = true;
          auto scaled = [gain](uint16_t v) {
            return __bfloat162float(__float2bfloat16_rn(__bfloat162float(v) * __bfloat162float(gain)));
          };
          float x = scaled(g), y = scaled(u);
          uint16_t expected = __float2bfloat16_rn((x / (1.f + expf(-x))) * y);
          errors += actual[vec].v[lane] != expected;
          ++words;
        }
      }
    }
  }
  std::cout << words << ' ' << guards << ' ' << errors << ' '
            << std::count(seen_gate, seen_gate + 65536, true) << ' '
            << std::count(seen_up, seen_up + 65536, true) << '\n';
}
